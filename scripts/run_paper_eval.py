#!/usr/bin/env python3
"""
Paper evaluation suite runner (RQ1 → RQ2 → RQ3).

Usage:
  python scripts/run_paper_eval.py --list
  python scripts/run_paper_eval.py --list-mods seaquest
  python scripts/run_paper_eval.py --rq 1 --only seaquest_skills_ego
  python scripts/run_paper_eval.py --rq 1 --phase train --shard 1/2
  python scripts/run_paper_eval.py --rq 2   # gated on RQ1 skill ckpts
  python scripts/run_paper_eval.py --smoke --trial   # full wiring smoke test
  python scripts/materialize_paper_eval_configs.py   # regenerate RQ2/RQ3 YAMLs

  # DESTROY ablation (3 GPUs: one variant per shard):
  python scripts/run_paper_eval.py --suite destroy_ablation --gpus 0,1,2 --wait-deps --lean

  # DESTROY urgency (contact+living-tax, no shot ego):
  python scripts/run_paper_eval.py --suite destroy_urgency --wait-deps --lean

  # COLLECT urgency + DESTROY shot-between (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite collect_shot_probes --gpus 0,1 --wait-deps --lean

  # Pose-only vs DESTROY-only extras (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite ego_pose_probes --gpus 0,1 --wait-deps --lean

  # COLLECT death-off vs DESTROY-first heuristic (2 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite collect_combat_probes --gpus 0,1 --wait-deps --lean

  # Distance clip 50px × 8 overlapping world-θ bins (4 GPUs, one arm each):
  python scripts/run_paper_eval.py --suite ego_clip_bins --gpus 0,1,2,3 --wait-deps --lean

  # Skill-reshape validation (COLLECT approach + AVOID escape; cluster RTPT):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --eval-only
  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py --suite skill_reshape --trial --lean

  # One-shot overnight on two GPUs (workers poll for RQ1 ckpts before gated jobs):
  python scripts/run_paper_eval.py --gpus 0,1 --wait-deps

Fail-fast guards (avoid burning hours / idling forever):
  - Abort if free disk on the project volume drops below --min-free-gb (default 2).
  - Treat ENOSPC / "No space left on device" as fatal for the whole suite.
  - With --wait-deps, do not wait for stems this worker already failed to produce.

Disk-lean overnight (shared /home nearly full):
  python scripts/run_paper_eval.py --gpus 0,1 --wait-deps --lean
  # --lean: no videos, no training_metrics.csv, wandb local dirs under $TMPDIR

Resume remaining work on 3 GPUs (skip finished trains+evals; drop cancelled pixel RQ1):
  python scripts/run_paper_eval.py --gpus 0,1,2 --wait-deps --lean --resume \\
    --exclude seaquest_skills_pixel seaquest_skills_pixel_eval

  # Preview what is still missing (includes incomplete/missing *_mods.json evals):
  python scripts/run_paper_eval.py --resume --list \\
    --exclude seaquest_skills_pixel seaquest_skills_pixel_eval
"""

from __future__ import annotations

import argparse
import copy
import errno
import gc
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Abort when free space on the project volume drops below this.
# Modest default: shared /home is often nearly full; ckpts << videos/wandb.
DEFAULT_MIN_FREE_GB = 2.0

# Avoid XLA:GPU Triton GEMM fusion CHECK failures
# (triton_tiling_propagation.cc: dst_remaining_size == 1) seen on some
# jaxlib/CUDA stacks during first compile of large train graphs.
_DEFAULT_XLA_FLAGS = (
    "--xla_gpu_strict_conv_algorithm_picker=false "
    "--xla_gpu_enable_triton_gemm=false"
)
_existing = os.environ.get("XLA_FLAGS", "").strip()
if not _existing:
    os.environ["XLA_FLAGS"] = _DEFAULT_XLA_FLAGS
elif "--xla_gpu_enable_triton_gemm" not in _existing:
    os.environ["XLA_FLAGS"] = f"{_existing} --xla_gpu_enable_triton_gemm=false"
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_JAXATARI_FORK = Path("/home/paul/Documents/JAXAtari-seaquest-robustness")
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "paper_eval"
MODS_DIR = CONFIG_DIR / "mods"
RTPT_BASE = "HRL_paper_eval"
DEFAULT_SEED = 44
TRIAL_STEPS_DEFAULT = 16384  # one update at 128×128
# Full-game horizon for paper / skill-reshape evals (CLI default remains 800).
SUITE_EVAL_MAX_FRAMES = 27000

# After COLLECT approach + AVOID escape reshaping: train ego Seaquest, then
# standard mod eval, Pong transfer, BeamRider (idle FIRE).
SKILL_RESHAPE_SUITE = "skill_reshape"
SKILL_RESHAPE_STEMS = (
    "seaquest_skills_ego",
    "seaquest_skills_ego_eval",
    "pong_transfer_ego",
    "beamrider_transfer_ego",
)
SKILL_RESHAPE_EVAL_STEMS = tuple(
    s for s in SKILL_RESHAPE_STEMS if s != "seaquest_skills_ego"
)
DESTROY_ABLATION_SUITE = "destroy_ablation"
# Order is shard-critical: 3 GPUs get (train, eval, pong) of one variant each.
DESTROY_ABLATION_STEMS = (
    "seaquest_skills_ego_destroy_yalign",
    "seaquest_skills_ego_destroy_standoff",
    "seaquest_skills_ego_destroy_contact",
    "seaquest_skills_ego_destroy_yalign_eval",
    "seaquest_skills_ego_destroy_standoff_eval",
    "seaquest_skills_ego_destroy_contact_eval",
    "pong_transfer_ego_destroy_yalign",
    "pong_transfer_ego_destroy_standoff",
    "pong_transfer_ego_destroy_contact",
)
DESTROY_URGENCY_SUITE = "destroy_urgency"
# Shot-ego arm dropped: it underperformed; suite is urgency (reward+heuristic) only.
DESTROY_URGENCY_STEMS = (
    "seaquest_skills_ego_destroy_urgency",
    "seaquest_skills_ego_destroy_urgency_eval",
    "pong_transfer_ego_destroy_urgency",
)
COLLECT_SHOT_PROBES_SUITE = "collect_shot_probes"
# Shard-critical for 2 GPUs: GPU i gets train/eval/pong of one arm.
COLLECT_SHOT_PROBES_STEMS = (
    "seaquest_skills_ego_collect_urgency",
    "seaquest_skills_ego_destroy_shot_between",
    "seaquest_skills_ego_collect_urgency_eval",
    "seaquest_skills_ego_destroy_shot_between_eval",
    "pong_transfer_ego_collect_urgency",
    "pong_transfer_ego_destroy_shot_between",
)
EGO_POSE_PROBES_SUITE = "ego_pose_probes"
# Pose-only (12-dim all heads) vs DESTROY-only extras (DESTROY 24 / others 12).
EGO_POSE_PROBES_STEMS = (
    "seaquest_skills_ego_pose",
    "seaquest_skills_ego_destroy_extras",
    "seaquest_skills_ego_pose_eval",
    "seaquest_skills_ego_destroy_extras_eval",
    "pong_transfer_ego_pose",
    "pong_transfer_ego_destroy_extras",
)
COLLECT_COMBAT_PROBES_SUITE = "collect_combat_probes"
# Isolated urg-v2 clones: COLLECT death penalty off vs DESTROY-first heuristic.
COLLECT_COMBAT_PROBES_STEMS = (
    "seaquest_skills_ego_collect_nodeath",
    "seaquest_skills_ego_destroy_first",
    "seaquest_skills_ego_collect_nodeath_eval",
    "seaquest_skills_ego_destroy_first_eval",
    "pong_transfer_ego_collect_nodeath",
    "pong_transfer_ego_destroy_first",
)
NO_SURFACE_DEATH_SUITE = "no_surface_death"
NO_SURFACE_DEATH_STEMS = (
    "seaquest_skills_ego_no_surface_death",
    "seaquest_skills_ego_no_surface_death_eval",
    "pong_transfer_ego_no_surface_death",
)
# urg-v2 + new heuristic: 2×2 of distance clip 50px and 8 world-θ bins.
# Shard-critical for 4 GPUs: GPU i gets train/eval/pong of one arm.
EGO_CLIP_BINS_SUITE = "ego_clip_bins"
EGO_CLIP_BINS_STEMS = (
    "seaquest_skills_ego_clip_bins_base",
    "seaquest_skills_ego_clip50",
    "seaquest_skills_ego_angbins",
    "seaquest_skills_ego_clip50_angbins",
    "seaquest_skills_ego_clip_bins_base_eval",
    "seaquest_skills_ego_clip50_eval",
    "seaquest_skills_ego_angbins_eval",
    "seaquest_skills_ego_clip50_angbins_eval",
    "pong_transfer_ego_clip_bins_base",
    "pong_transfer_ego_clip50",
    "pong_transfer_ego_angbins",
    "pong_transfer_ego_clip50_angbins",
)
SUITE_STEMS = {
    SKILL_RESHAPE_SUITE: SKILL_RESHAPE_STEMS,
    DESTROY_ABLATION_SUITE: DESTROY_ABLATION_STEMS,
    DESTROY_URGENCY_SUITE: DESTROY_URGENCY_STEMS,
    COLLECT_SHOT_PROBES_SUITE: COLLECT_SHOT_PROBES_STEMS,
    EGO_POSE_PROBES_SUITE: EGO_POSE_PROBES_STEMS,
    COLLECT_COMBAT_PROBES_SUITE: COLLECT_COMBAT_PROBES_STEMS,
    NO_SURFACE_DEATH_SUITE: NO_SURFACE_DEATH_STEMS,
    EGO_CLIP_BINS_SUITE: EGO_CLIP_BINS_STEMS,
}


def _default_batch_tag() -> str:
    """Dated suite id so W&B filters don't mix overnight batches."""
    return datetime.now().strftime("paper_eval_%Y%m%d_%H%M")


def _resolve_batch_tag(cli_value: str | None) -> str:
    if cli_value and str(cli_value).strip():
        return str(cli_value).strip()
    env = os.environ.get("PAPER_EVAL_BATCH_TAG", "").strip()
    if env:
        return env
    return _default_batch_tag()


def _attach_batch_wandb_tag(cfg: dict[str, Any], batch_tag: str) -> None:
    tags = list(cfg.get("WANDB_TAGS") or [])
    # Keep suite:paper_eval; add dated batch for grouping this overnight.
    if "suite:paper_eval" not in tags:
        tags.append("suite:paper_eval")
    batch_key = f"batch:{batch_tag}"
    if batch_key not in tags:
        tags.append(batch_key)
    cfg["WANDB_TAGS"] = tags
    cfg["PAPER_EVAL_BATCH_TAG"] = batch_tag

# Representative smoke matrix (every KIND / PHASE exercised once).
SMOKE_STEMS = (
    # RQ1 train
    "seaquest_skills_ego",
    "seaquest_pqn",
    # RQ1 eval
    "seaquest_skills_ego_eval",
    "seaquest_heuristic_random",
    "seaquest_pqn_eval",
    # RQ2
    "pong_transfer_ego",
    "pong_finetune_ego",
    "pong_from_scratch_ego",
    # RQ3
    "seaquest_meta_pixel_ego_skills",
    "seaquest_tandem_pixel_ego",
)

RQ1_SKILL_STEMS = (
    "seaquest_skills_ego",
    "seaquest_skills_pixel",
    "seaquest_skills_oc",
)


def _prepend_jaxatari_root(root: Path | None) -> None:
    if root is None:
        return
    src = root / "src"
    if not src.is_dir():
        raise SystemExit(f"JAXAtari root missing src/: {root}")
    sys.path.insert(0, str(src))
    os.environ["PAPER_EVAL_JAXATARI_ROOT"] = str(root)
    print(f"Using JAXAtari from {src}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    alg = raw.pop("alg", {}) or {}
    return {**raw, **alg}


def _parse_shard(spec: str) -> tuple[int, int]:
    parts = spec.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard {spec!r}; expected INDEX/TOTAL")
    shard_one_based, num_shards = int(parts[0]), int(parts[1])
    if num_shards < 1 or not 1 <= shard_one_based <= num_shards:
        raise ValueError(f"Invalid shard {spec!r}")
    return shard_one_based - 1, num_shards


def _is_runnable_config(path: Path) -> bool:
    if path.name.startswith("_") or path.name.startswith("manifest"):
        return False
    try:
        path.relative_to(MODS_DIR)
        return False
    except ValueError:
        pass
    return path.suffix in {".yaml", ".yml"}


def _discover_configs(
    *,
    rq: int | None = None,
    only: list[str] | None = None,
    phase: str | None = None,
) -> list[Path]:
    paths = sorted(p for p in CONFIG_DIR.rglob("*.yaml") if _is_runnable_config(p))
    if rq is not None:
        paths = [p for p in paths if int(_load_yaml(p).get("PAPER_EVAL_RQ", -1)) == rq]
    if phase is not None:
        paths = [
            p
            for p in paths
            if str(_load_yaml(p).get("PAPER_EVAL_PHASE", "train")).lower() == phase.lower()
        ]
    if only:
        wanted = {name.removesuffix(".yaml") for name in only}
        paths = [p for p in paths if p.stem in wanted]
        missing = wanted - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"Config(s) not found: {sorted(missing)}")
    return paths


def _shard_paths(paths: list[Path], *, shard_idx: int, num_shards: int) -> list[Path]:
    return [p for i, p in enumerate(paths) if i % num_shards == shard_idx]


def _rtpt_name(
    shard: str | None,
    rq: int | None,
    *,
    suite: str | None = None,
) -> str:
    if suite:
        base = f"{RTPT_BASE}_{suite}"
    elif rq is not None:
        base = f"{RTPT_BASE}_rq{rq}"
    else:
        base = RTPT_BASE
    if shard is None:
        return base
    return f"{base}_s{_parse_shard(shard)[0] + 1}"


def _align_trial_timesteps(cfg: dict[str, Any]) -> None:
    batch = int(cfg.get("NUM_ENVS", 128)) * int(cfg.get("NUM_STEPS", 128))
    if batch <= 0:
        return
    total = int(cfg.get("TOTAL_TIMESTEPS", 0))
    cfg["TOTAL_TIMESTEPS"] = max(1, total // batch) * batch
    if "TOTAL_TIMESTEPS_DECAY" in cfg:
        cfg["TOTAL_TIMESTEPS_DECAY"] = cfg["TOTAL_TIMESTEPS"]


def _configure_lean_disk() -> Path:
    """Put wandb local files on $TMPDIR (often a separate volume from /home)."""
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    tmp_root = Path(os.environ.get("TMPDIR") or "/tmp") / user / "cbl_paper_eval"
    wandb_dir = tmp_root / "wandb"
    cache_dir = tmp_root / "wandb_cache"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    # Avoid large local artifact media if anything slips through.
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    try:
        tmp_free = shutil.disk_usage(tmp_root).free / (1024.0**3)
        tmp_note = f"{tmp_free:.1f} GiB free on tmp"
    except OSError:
        tmp_note = "tmp free unknown"
    print(
        f"lean disk: WANDB_DIR={wandb_dir}  WANDB_CACHE_DIR={cache_dir}  ({tmp_note})",
        flush=True,
    )
    return tmp_root


def _apply_run_defaults(
    cfg: dict[str, Any],
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
    seed: int | None,
    batch_tag: str | None = None,
    save_training_metrics_csv: bool = True,
) -> None:
    if wandb_mode is not None:
        cfg["WANDB_MODE"] = wandb_mode
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        cfg.setdefault("WANDB_TAGS", [])
        if "trial" not in cfg["WANDB_TAGS"]:
            cfg["WANDB_TAGS"] = [*cfg["WANDB_TAGS"], "trial"]
        _align_trial_timesteps(cfg)
    if seed is not None:
        cfg["SEED"] = int(seed)
    cfg.setdefault("ENTITY", "paul-seitz-tu-darmstadt")
    cfg.setdefault("PROJECT", "concept_based_learning")
    cfg.setdefault("SEED", DEFAULT_SEED)
    cfg.setdefault("NUM_SEEDS", 1)
    cfg["RECORD_VIDEO"] = record_video
    cfg["VIDEO_AT_END"] = record_video
    cfg["SAVE_TRAINING_METRICS_CSV"] = bool(save_training_metrics_csv)
    if batch_tag:
        _attach_batch_wandb_tag(cfg, batch_tag)


def _skill_ckpt_from_save(save_path: str | Path, env_name: str, seed: int) -> Path:
    root = Path(save_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / env_name / f"hierarchical_pqn_{env_name}_seed{seed}.safetensors"


def _eval_json_path(stem: str) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{stem}_mods.json"
    )


def _flat_pqn_ckpt_from_cfg(cfg: dict[str, Any], *, seed: int) -> Path:
    from concept_based_learning.paper_eval.checkpoints import resolve_flat_pqn_checkpoint

    return resolve_flat_pqn_checkpoint(
        object_centric=bool(cfg.get("OBJECT_CENTRIC", True)),
        env_name=str(cfg.get("ENV_NAME", "seaquest")).lower(),
        seed=int(cfg.get("SEED", seed)),
        save_root=cfg.get("SAVE_PATH"),
        vmap_index=0,
    )


def _skills_train_artifacts(cfg: dict[str, Any], *, seed: int, kind: str) -> list[Path]:
    """Expected checkpoint files produced by a skills / meta / tandem train."""
    env_name = str(cfg.get("ENV_NAME", "seaquest")).lower()
    seed_i = int(cfg.get("SEED", seed))
    save_root = cfg.get("SAVE_PATH", "./outputs/models")
    root = Path(save_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    env_dir = root / env_name
    alg = str(cfg.get("ALG_NAME", "hierarchical_pqn"))
    if kind in {"meta", "tandem"}:
        return [
            env_dir / f"{alg}_{env_name}_seed{seed_i}_skills.safetensors",
            env_dir / f"{alg}_{env_name}_seed{seed_i}_meta.safetensors",
        ]
    # skills_only / finetune: single stacked skill ckpt (no _skills suffix)
    return [env_dir / f"{alg}_{env_name}_seed{seed_i}.safetensors"]


def _eval_json_complete(
    path: Path,
    *,
    env_name: str | None = None,
    expected_mod_keys: list[str] | None = None,
) -> tuple[bool, str]:
    """
    True only if the mods JSON exists, parses, and has a non-empty ``results``
    map that covers expected mod keys (at least ``base``).

    Partial writes from earlier disk-full failures are treated as incomplete.
    """
    if not path.is_file() or path.stat().st_size < 8:
        return False, f"missing {path}"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable {path.name}: {exc}"
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        return False, f"empty results in {path.name}"
    if "base" not in results:
        return False, f"{path.name} missing results['base']"

    keys = expected_mod_keys
    if keys is None and env_name:
        try:
            from concept_based_learning.paper_eval.mods import mod_specs_for_env

            keys = [m.key for m in mod_specs_for_env(env_name)]
        except Exception:
            keys = None
    if keys:
        missing = [k for k in keys if k not in results]
        if missing:
            return (
                False,
                f"{path.name} incomplete ({len(results)}/{len(keys)} mods; "
                f"missing e.g. {missing[:3]})",
            )
    return True, f"{path.name} ({len(results)} mods)"


def experiment_done(path: Path, *, seed: int) -> tuple[bool, str]:
    """
    Return (done, reason) based on on-disk artifacts.

    Train → checkpoint(s) present.
    Eval / transfer / heuristic → complete mods JSON (all allowlisted mods).
    """
    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    status = str(raw.get("PAPER_EVAL_STATUS", "ready")).lower()
    if status in {"stub", "planned"}:
        return True, f"status={status} (skipped)"

    if phase == "eval" or kind in {"heuristic_random", "transfer"}:
        env_name = str(raw.get("ENV_NAME") or raw.get("PAPER_EVAL_MODS") or "seaquest")
        if isinstance(env_name, list):
            env_name = env_name[0] if env_name else "seaquest"
        env_name = str(env_name).lower()
        out = _eval_json_path(path.stem)
        ok, detail = _eval_json_complete(out, env_name=env_name)
        if ok:
            return True, f"eval {detail}"
        return False, f"eval incomplete: {detail}"

    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if kind == "flat_pqn":
        ckpt = _flat_pqn_ckpt_from_cfg(cfg, seed=seed)
        if ckpt.is_file():
            return True, f"ckpt {ckpt.name}"
        return False, f"missing {ckpt}"

    arts = _skills_train_artifacts(cfg, seed=seed, kind=kind)
    missing = [p for p in arts if not p.is_file()]
    if not missing:
        return True, "ckpt " + ", ".join(p.name for p in arts)
    return False, "missing " + ", ".join(str(p) for p in missing)


def _filter_remaining_paths(
    paths: list[Path],
    *,
    seed: int,
    exclude: set[str],
    skip_done: bool,
) -> list[Path]:
    """Drop completed runs and explicitly excluded stems (+ dependents)."""
    exclude_l = {s.removesuffix(".yaml") for s in exclude}
    remaining: list[Path] = []
    skipped_done = 0
    skipped_excl = 0
    skipped_dep = 0
    for path in paths:
        stem = path.stem
        if stem in exclude_l:
            print(f"  SKIP (excluded): {stem}", flush=True)
            skipped_excl += 1
            continue
        requires = _required_rq1_stems(path)
        blocked = [r for r in requires if r in exclude_l]
        if blocked:
            print(
                f"  SKIP (depends on excluded {', '.join(blocked)}): {stem}",
                flush=True,
            )
            skipped_dep += 1
            continue
        if skip_done:
            done, reason = experiment_done(path, seed=seed)
            if done:
                print(f"  SKIP (done: {reason}): {stem}", flush=True)
                skipped_done += 1
                continue
        remaining.append(path)
    print(
        f"resume filter: keep {len(remaining)} / {len(paths)} "
        f"(done={skipped_done}, excluded={skipped_excl}, dep-excluded={skipped_dep})",
        flush=True,
    )
    return remaining


def _rq1_checkpoint_ready(stem: str, *, seed: int) -> tuple[bool, Path]:
    path = CONFIG_DIR / "rq1" / f"{stem}.yaml"
    if not path.is_file():
        return False, path
    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    ckpt = _skill_ckpt_from_save(
        cfg.get("SAVE_PATH", f"./outputs/models/paper_eval/rq1/{stem}"),
        str(cfg.get("ENV_NAME", "seaquest")),
        int(cfg.get("SEED", seed)),
    )
    return ckpt.is_file(), ckpt


def _check_rq_gate(paths: list[Path], *, seed: int) -> None:
    missing: list[str] = []
    for path in paths:
        raw = _load_yaml(path)
        requires = list(raw.get("PAPER_EVAL_REQUIRES_RQ1") or [])
        rq = int(raw.get("PAPER_EVAL_RQ", 1))
        kind = str(raw.get("PAPER_EVAL_KIND", "")).lower()
        phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
        # Default gate for RQ>=2 train/finetune/transfer (not from_scratch with empty requires)
        if not requires and rq >= 2 and phase != "train":
            requires = list(RQ1_SKILL_STEMS[:1])  # at least ego
        if kind in {"meta", "tandem"} and not requires:
            requires = ["seaquest_skills_ego"]
        for stem in requires:
            ok, ckpt = _rq1_checkpoint_ready(str(stem), seed=seed)
            if not ok:
                missing.append(f"{path.stem} requires {stem} → missing {ckpt}")
    if missing:
        raise SystemExit(
            "RQ gate failed (train RQ1 first, or pass --skip-rq-gate):\n  - "
            + "\n  - ".join(missing)
        )


def _required_rq1_stems(path: Path) -> list[str]:
    raw = _load_yaml(path)
    requires = list(raw.get("PAPER_EVAL_REQUIRES_RQ1") or [])
    rq = int(raw.get("PAPER_EVAL_RQ", 1))
    kind = str(raw.get("PAPER_EVAL_KIND", "")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    if not requires and rq >= 2 and phase != "train":
        requires = list(RQ1_SKILL_STEMS[:1])
    if kind in {"meta", "tandem"} and not requires:
        requires = ["seaquest_skills_ego"]
    return [str(s) for s in requires]


def _is_no_space_error(exc: BaseException) -> bool:
    """True for ENOSPC and common chained / string forms of disk-full failures."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, OSError) and cur.errno == errno.ENOSPC:
            return True
        msg = str(cur).lower()
        if "no space left on device" in msg or "[errno 28]" in msg:
            return True
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None and cur.__context__ is not cur.__cause__:
            stack.append(cur.__context__)
    return False


def _free_gb(path: Path | None = None) -> float:
    """Free space (GiB) on the filesystem that holds ``path`` (default: project root)."""
    target = path if path is not None else PROJECT_ROOT
    try:
        usage = shutil.disk_usage(target)
    except FileNotFoundError:
        usage = shutil.disk_usage(PROJECT_ROOT)
    return float(usage.free) / (1024.0**3)


def _assert_disk_space(
    *,
    min_free_gb: float,
    where: str,
    path: Path | None = None,
) -> None:
    """Crash immediately if the project volume is too full to continue safely."""
    if min_free_gb <= 0:
        return
    free = _free_gb(path)
    if free < float(min_free_gb):
        raise SystemExit(
            f"Aborting ({where}): only {free:.2f} GiB free on "
            f"{path or PROJECT_ROOT} (need >= {min_free_gb:.2f} GiB). "
            "Free disk space (e.g. wandb/run-* , old outputs) and rerun."
        )


def _wait_for_path_deps(
    path: Path,
    *,
    seed: int,
    poll_s: float = 60.0,
    timeout_h: float = 36.0,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    failed_local_stems: set[str] | None = None,
) -> None:
    """Block until PAPER_EVAL_REQUIRES_RQ1 checkpoints exist (multi-GPU overnight)."""
    requires = _required_rq1_stems(path)
    if not requires:
        return
    failed_local = failed_local_stems or set()
    already_failed = [s for s in requires if s in failed_local]
    if already_failed:
        raise SystemExit(
            f"Aborting wait for {path.stem}: required dep(s) already failed in "
            f"this worker (will never appear): {', '.join(already_failed)}"
        )
    deadline = time.time() + max(60.0, timeout_h * 3600.0)
    last_msg = ""
    while True:
        _assert_disk_space(
            min_free_gb=min_free_gb,
            where=f"wait-deps for {path.stem}",
        )
        missing: list[str] = []
        for stem in requires:
            if stem in failed_local:
                raise SystemExit(
                    f"Aborting wait for {path.stem}: required dep {stem} failed "
                    "in this worker while waiting."
                )
            ok, ckpt = _rq1_checkpoint_ready(stem, seed=seed)
            if not ok:
                missing.append(f"{stem} → {ckpt}")
        if not missing:
            print(f"  Deps ready for {path.stem}", flush=True)
            return
        msg = f"  Waiting on deps for {path.stem}: {', '.join(missing)}"
        if msg != last_msg:
            print(msg, flush=True)
            last_msg = msg
        if time.time() >= deadline:
            raise SystemExit(
                f"Timed out after {timeout_h}h waiting for deps of {path.stem}:\n  - "
                + "\n  - ".join(missing)
            )
        time.sleep(poll_s)


_PHASE_SORT = {"train": 0, "finetune": 1, "eval": 2}


def _schedule_key(path: Path) -> tuple:
    """Prefer ungated RQ1 trains first so other GPUs can unblock sooner."""
    raw = _load_yaml(path)
    requires = _required_rq1_stems(path)
    rq = int(raw.get("PAPER_EVAL_RQ", 1))
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    return (1 if requires else 0, rq, _PHASE_SORT.get(phase, 9), path.stem)


def _launch_multi_gpu(argv: list[str], gpu_ids: list[str]) -> int:
    """
    Re-exec this script once per GPU with ``--shard i/N --wait-deps``.

    Parent blocks until all workers exit. Returns max worker return code.
    """
    n = len(gpu_ids)
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--gpus":
            i += 2  # skip flag + value
            continue
        if a.startswith("--gpus="):
            i += 1
            continue
        # Workers get their own --shard; drop a parent-provided one if any.
        if a == "--shard":
            i += 2
            continue
        if a.startswith("--shard="):
            i += 1
            continue
        cleaned.append(a)
        i += 1

    if "--wait-deps" not in cleaned:
        cleaned.append("--wait-deps")

    procs: list[subprocess.Popen] = []
    for wi, gpu in enumerate(gpu_ids):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *cleaned,
            "--shard",
            f"{wi + 1}/{n}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        # Share one batch tag across workers (parent sets PAPER_EVAL_BATCH_TAG).
        if "PAPER_EVAL_BATCH_TAG" not in env:
            env["PAPER_EVAL_BATCH_TAG"] = _default_batch_tag()
        print(f"\n=== multi-GPU worker {wi + 1}/{n}  CUDA_VISIBLE_DEVICES={gpu} ===")
        print(" ", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT)))

    codes = [p.wait() for p in procs]
    worst = max(codes) if codes else 1
    if worst != 0:
        print(f"\nmulti-GPU: worker exit codes {codes}", flush=True)
    else:
        print("\nmulti-GPU: all workers finished successfully.", flush=True)
    return int(worst)


def _n_eval_mods(raw: dict[str, Any], *, trial: bool) -> int:
    env = str(raw.get("ENV_NAME", "seaquest"))
    try:
        from concept_based_learning.paper_eval.mods import mod_specs_for_env

        return max(1, len(mod_specs_for_env(env, trial=trial)))
    except Exception:
        return 1


def _updates_for_config_path(
    path: Path, *, trial_steps: int | None, trial: bool = False
) -> int:
    """Weighted RTPT units for one config (pixel train ≫ ego; evals are non-zero)."""
    from concept_based_learning.training.rtpt_progress import (
        RTPT_EVAL_UNITS_PER_MOD,
        num_updates_from_timesteps,
        rtpt_weight_for_config,
    )

    raw = _load_yaml(path)
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if phase == "eval" or kind == "heuristic_random":
        return _n_eval_mods(raw, trial=trial) * RTPT_EVAL_UNITS_PER_MOD
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        _align_trial_timesteps(cfg)
    updates = num_updates_from_timesteps(
        int(cfg.get("TOTAL_TIMESTEPS", 0)),
        num_steps=int(cfg.get("NUM_STEPS", 128)),
        num_envs=int(cfg.get("NUM_ENVS", 128)),
    )
    return updates * rtpt_weight_for_config(cfg, kind=kind)


def _bind_rtpt_job(path: Path) -> None:
    """Scale jax.debug.callback ticks for the job about to run (evals tick per mod)."""
    from concept_based_learning.training.rtpt_progress import (
        rtpt_weight_for_config,
        set_rtpt_job_weight,
    )

    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    if phase == "eval" or kind == "heuristic_random":
        set_rtpt_job_weight(1)
        return
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    weight = rtpt_weight_for_config(cfg, kind=kind)
    set_rtpt_job_weight(weight)
    if weight > 1:
        print(f"  RTPT job weight={weight} (pixel/oc wall-time scaling)", flush=True)


def _init_rtpt_for_paths(
    paths: list[Path],
    *,
    experiment_name: str,
    trial_steps: int | None,
    trial: bool = False,
) -> None:
    from concept_based_learning.training.rtpt_progress import start_rtpt

    total = max(
        1,
        sum(
            _updates_for_config_path(p, trial_steps=trial_steps, trial=trial)
            for p in paths
        ),
    )
    if start_rtpt(
        {
            "RTPT_ENABLED": True,
            "RTPT_NAME_INITIALS": "PS",
            "RTPT_EXPERIMENT_NAME": experiment_name,
            "RTPT_STEP_EVERY": 10,
            "TOTAL_TIMESTEPS": total,
            "NUM_STEPS": 1,
            "NUM_ENVS": 1,
        }
    ):
        print(
            f"RTPT initialized: {total:,} weighted units across {len(paths)} run(s) "
            f"(evals included; pixel×10, oc×2)\n"
        )


def _visible_gpu_indices() -> list[int] | None:
    """Physical GPU ids this process may use, or None if unrestricted."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    if raw == "-1":
        return []
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids


def _gpu_mem_brief() -> str:
    """One-line nvidia-smi used/total for *this process's* visible GPUs."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible = _visible_gpu_indices()
        if visible is not None:
            if not visible:
                return "(cpu / no visible gpu)"
            cmd.extend(["-i", ",".join(str(i) for i in visible)])
        out = subprocess.check_output(cmd, text=True, timeout=5)
        parts = []
        for line in out.strip().splitlines():
            idx, used, total = [p.strip() for p in line.split(",")]
            parts.append(f"gpu{idx}:{used}/{total}MiB")
        return " ".join(parts) if parts else "(no gpu)"
    except Exception:
        return "(nvidia-smi unavailable)"


def _pin_jaxatari_action_sets_to_host() -> None:
    """
    Copy JAXAtari ``ACTION_SET`` class attrs to host numpy.

    Those sets are module-level ``jnp.array``s. Deleting live device buffers
    (or ``jax.clear_backends``) tombstones them, and the next ``AtariWrapper``
    then fails with ``Array has been deleted with shape=int32[18]``.
    """
    try:
        import inspect

        import jax
        import jax.numpy as jnp
        import numpy as np
        from jaxatari.core import GAME_MODULES
        from jaxatari.environment import JaxEnvironment
    except Exception:
        return

    for module_path in GAME_MODULES.values():
        try:
            module = __import__(module_path, fromlist=["*"])
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, JaxEnvironment) or obj is JaxEnvironment:
                continue
            arr = getattr(obj, "ACTION_SET", None)
            if arr is None:
                continue
            try:
                if isinstance(arr, jax.Array) and getattr(arr, "is_deleted", lambda: False)():
                    continue
                host = np.asarray(arr, dtype=np.int32)
            except Exception:
                continue
            # Numpy stays off-device; jnp.take / np.array still accept it.
            try:
                obj.ACTION_SET = host
            except Exception:
                try:
                    obj.ACTION_SET = jnp.asarray(host, dtype=jnp.int32)
                except Exception:
                    pass


def _clear_jax_memory() -> None:
    """Drop compile caches. Do not delete live buffers or the JAX backend.

    Overnight trains used ``live_buffers().delete()`` / ``clear_backends()`` to
    reclaim VRAM. That also tombstones JAXAtari class constants (``ACTION_SET``),
    so the next eval in the same process crashes in ``AtariWrapper.__init__``.
    ``clear_caches`` + GC is enough for eval→eval; the CUDA pool is reused.
    """
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _release(*, label: str = "") -> None:
    """Best-effort VRAM reclaim between paper_eval jobs in the same process."""
    before = _gpu_mem_brief()
    _pin_jaxatari_action_sets_to_host()
    gc.collect()
    _clear_jax_memory()
    gc.collect()
    _pin_jaxatari_action_sets_to_host()
    after = _gpu_mem_brief()
    tag = f" after {label}" if label else ""
    print(f"  GPU release{tag}: {before} → {after}", flush=True)


def _print_mods(env: str | None) -> None:
    from concept_based_learning.paper_eval.mods import load_mod_allowlist, mod_specs_for_env

    envs = [env] if env else sorted(p.stem for p in MODS_DIR.glob("*.yaml"))
    for name in envs:
        raw = load_mod_allowlist(name)
        print(f"\n=== mods/{name}.yaml  status={raw.get('status', '?')} ===")
        for spec in mod_specs_for_env(name):
            mods_s = ",".join(spec.mods) if spec.mods else "(base)"
            print(f"  {spec.key:28s} [{spec.category:10s}]  {mods_s}")


def _print_matrix(paths: list[Path]) -> None:
    print(f"\n=== paper_eval ({CONFIG_DIR.relative_to(PROJECT_ROOT)}) ===")
    print(f"{'Config':<36} {'RQ':>2} {'phase':<8} {'kind':<16} status")
    print("-" * 90)
    for path in paths:
        raw = _load_yaml(path)
        print(
            f"{path.stem:<36} "
            f"{str(raw.get('PAPER_EVAL_RQ', '?')):>2} "
            f"{str(raw.get('PAPER_EVAL_PHASE', 'train')):<8} "
            f"{str(raw.get('PAPER_EVAL_KIND', 'skills')):<16} "
            f"{raw.get('PAPER_EVAL_STATUS', 'ready')}"
        )
    print(f"\n{len(paths)} configs. W&B filter: suite:paper_eval")


# Eval YAMLs may set PAPER_EVAL_ARCHITECTURE_OF to pin network/obs shape to a
# train config while still overriding policy/eval knobs locally.
_EVAL_OVERRIDE_KEYS = (
    "HEURISTIC_MODE",
    "NO_TARGET_GAME_ACTION",
    "SKILL_DEATH_PENALIZE_COLLECT",
    "EVAL_GAME_MODS",
    "GAME_MODS",
    "SURFACE_DANGER_CLOSE_RANGE",
    "DESTROY_INTERRUPT_RANGE",
    "AVOID_INTERRUPT_RANGE",
    "DIVER_LANE_Y_PX",
    "COLLECT_REQUIRES_NO_DESTROY",
)


def _architecture_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Load arch YAML referenced by PAPER_EVAL_ARCHITECTURE_OF if present.

    When an architecture ref is used, still apply eval-local overrides from
    ``raw`` (e.g. ``HEURISTIC_MODE: avoid_only`` on an AVOID probe).
    """
    ref = raw.get("PAPER_EVAL_ARCHITECTURE_OF")
    if not ref:
        return _flatten_experiment_cfg(copy.deepcopy(raw))
    ref_path = CONFIG_DIR / "rq1" / f"{str(ref).removesuffix('.yaml')}.yaml"
    if not ref_path.is_file():
        # search
        matches = list(CONFIG_DIR.rglob(f"{str(ref).removesuffix('.yaml')}.yaml"))
        if not matches:
            raise FileNotFoundError(f"Architecture config not found: {ref}")
        ref_path = matches[0]
    cfg = _flatten_experiment_cfg(copy.deepcopy(_load_yaml(ref_path)))
    eval_flat = _flatten_experiment_cfg(copy.deepcopy(raw))
    for key in _EVAL_OVERRIDE_KEYS:
        if key in eval_flat:
            cfg[key] = eval_flat[key]
    return cfg


def _run_skills_train(path: Path, **kwargs) -> None:
    import main as train_main

    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    _apply_run_defaults(cfg, **kwargs)
    print(f"\n{'=' * 70}\n  TRAIN skills: {path.relative_to(CONFIG_DIR)}")
    print(
        f"  ENV={cfg.get('ENV_NAME')} MODE={cfg.get('TRAIN_MODE')} "
        f"INPUT={cfg.get('SKILL_INPUT_MODE')} STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}\n"
        f"{'=' * 70}\n"
    )
    try:
        train_main.single_run(cfg)
    finally:
        _release(label=path.stem)


def _run_flat_pqn_train(path: Path, **kwargs) -> None:
    from concept_based_learning.baselines.flat_pqn_train import train_flat_pqn

    raw = _load_yaml(path)
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    _apply_run_defaults(cfg, **kwargs)
    print(f"\n{'=' * 70}\n  TRAIN flat_pqn: {path.relative_to(CONFIG_DIR)}")
    print(
        f"  ENV={cfg.get('ENV_NAME')} OC={cfg.get('OBJECT_CENTRIC')} "
        f"STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}\n{'=' * 70}\n"
    )
    try:
        train_flat_pqn(cfg)
    finally:
        _release(label=path.stem)


def _run_mod_eval(
    path: Path,
    *,
    trial: bool,
    seed: int | None,
    eval_episodes: int,
    eval_max_frames: int,
) -> None:
    from concept_based_learning.paper_eval.eval_harness import run_mod_eval

    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    env_name = str(raw.get("ENV_NAME", "seaquest"))
    cfg = _architecture_config(raw)
    if seed is not None:
        cfg["SEED"] = int(seed)
    cfg.setdefault("SEED", DEFAULT_SEED)
    base_seed = int(cfg["SEED"])

    skills_path = raw.get("PAPER_EVAL_SKILLS_CKPT")
    meta_path = raw.get("PAPER_EVAL_META_CKPT")
    flat_path = raw.get("PAPER_EVAL_FLAT_PQN_CKPT")
    object_centric = bool(raw.get("PAPER_EVAL_OBJECT_CENTRIC", raw.get("OBJECT_CENTRIC", True)))

    # Resolve default skill ckpt from architecture SAVE_PATH when needed.
    if kind in {"skills", "transfer", "hierarchical"} and not skills_path:
        skills_path = str(
            _skill_ckpt_from_save(
                cfg.get("SAVE_PATH", raw.get("SAVE_PATH", ".")),
                env_name if kind != "transfer" else "seaquest",
                base_seed,
            )
        )
        # Transfer always loads Seaquest skills
        if kind == "transfer":
            skills_path = raw.get("PAPER_EVAL_SKILLS_CKPT") or skills_path

    agent_kind = {
        "heuristic_random": "heuristic_random",
        "flat_pqn": "flat_pqn",
        "transfer": "transfer",
        "skills": "skills",
        "hierarchical": "skills",
    }.get(kind, kind)

    out_json = (
        PROJECT_ROOT
        / "outputs"
        / "models"
        / "paper_eval"
        / "eval"
        / f"{path.stem}_mods.json"
    )
    print(f"\n{'=' * 70}\n  EVAL mods: {path.relative_to(CONFIG_DIR)} kind={kind}\n{'=' * 70}")
    _pin_jaxatari_action_sets_to_host()
    try:
        run_mod_eval(
            agent_kind=agent_kind,
            config=cfg,
            env_name=env_name,
            skills_path=skills_path,
            meta_path=meta_path,
            flat_pqn_path=flat_path,
            object_centric=object_centric,
            trial=trial,
            mod_keys=list(raw["PAPER_EVAL_MOD_KEYS"]) if raw.get("PAPER_EVAL_MOD_KEYS") else None,
            num_episodes=eval_episodes,
            num_seeds=1,
            base_seed=base_seed,
            max_frames=eval_max_frames,
            output_json=out_json,
            extra_meta={
                "run_name": str(raw.get("WANDB_RUN_NAME", path.stem)),
                "config_stem": path.stem,
                "paper_eval_rq": raw.get("PAPER_EVAL_RQ", ""),
                "paper_eval_phase": raw.get("PAPER_EVAL_PHASE", "eval"),
                "paper_eval_kind": kind,
                "heuristic_mode": cfg.get("HEURISTIC_MODE"),
                "no_target_game_action": cfg.get("NO_TARGET_GAME_ACTION"),
            },
        )
    finally:
        _release(label=f"eval:{path.stem}")


def _run_experiment(
    path: Path,
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
    seed: int | None,
    trial: bool,
    eval_episodes: int,
    eval_max_frames: int,
    enforce_rq_gate: bool,
    post_eval: bool,
    batch_tag: str | None = None,
    save_training_metrics_csv: bool = True,
) -> None:
    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    status = str(raw.get("PAPER_EVAL_STATUS", "ready")).lower()
    if status in {"stub", "planned"}:
        print(f"\n  SKIP (status={status}): {path.stem}")
        return

    _bind_rtpt_job(path)

    if enforce_rq_gate and (
        raw.get("PAPER_EVAL_REQUIRES_RQ1")
        or kind in {"meta", "tandem", "transfer"}
        or (int(raw.get("PAPER_EVAL_RQ", 1)) >= 2 and phase == "finetune")
    ):
        _check_rq_gate([path], seed=int(seed if seed is not None else DEFAULT_SEED))

    train_kwargs = dict(
        wandb_mode=wandb_mode,
        trial_steps=trial_steps,
        record_video=record_video,
        seed=seed,
        batch_tag=batch_tag,
        save_training_metrics_csv=save_training_metrics_csv,
    )

    if phase == "eval" or kind == "heuristic_random":
        _run_mod_eval(
            path,
            trial=trial,
            seed=seed,
            eval_episodes=eval_episodes,
            eval_max_frames=eval_max_frames,
        )
        return

    if kind == "flat_pqn":
        _run_flat_pqn_train(path, **train_kwargs)
        return

    if kind in {"skills", "meta", "tandem"}:
        # meta_only / tandem / skills_only all go through main.single_run
        _run_skills_train(path, **train_kwargs)
        # Optional frozen mod eval after train (weights unchanged — not finetune-on-mods).
        # Prefer explicit ``--phase eval`` for the paper protocol; use ``--post-eval``
        # only when you want train+eval in one shot (e.g. --smoke).
        if (
            post_eval
            and phase in {"finetune", "train"}
            and raw.get("PAPER_EVAL_MODS")
            and kind != "meta"
        ):
            from concept_based_learning.paper_eval.eval_harness import run_mod_eval

            cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
            env_name = str(cfg.get("ENV_NAME", "seaquest"))
            seed_i = int(cfg.get("SEED", seed or DEFAULT_SEED))
            save_root = Path(cfg.get("SAVE_PATH", "."))
            if not save_root.is_absolute():
                save_root = PROJECT_ROOT / save_root
            env_dir = save_root / env_name
            # skills_only: hierarchical_pqn_{env}_seed{N}.safetensors
            # tandem:      hierarchical_pqn_{env}_seed{N}_skills.safetensors
            candidates = [
                env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}.safetensors",
                env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}_skills.safetensors",
            ]
            ckpt = next((p for p in candidates if p.is_file()), None)
            meta_ckpt = env_dir / f"hierarchical_pqn_{env_name}_seed{seed_i}_meta.safetensors"
            if ckpt is not None:
                print(f"\n  Post-train mod eval for {path.stem} → {ckpt.name}")
                try:
                    run_mod_eval(
                        agent_kind="skills",
                        config=cfg,
                        env_name=env_name,
                        skills_path=str(ckpt),
                        meta_path=str(meta_ckpt) if meta_ckpt.is_file() else None,
                        trial=trial,
                        num_episodes=eval_episodes,
                        num_seeds=1,
                        base_seed=seed_i,
                        max_frames=eval_max_frames,
                        output_json=PROJECT_ROOT
                        / "outputs"
                        / "models"
                        / "paper_eval"
                        / "eval"
                        / f"{path.stem}_post_mods.json",
                        extra_meta={
                            "run_name": str(raw.get("WANDB_RUN_NAME", path.stem)),
                            "config_stem": path.stem,
                            "paper_eval_rq": raw.get("PAPER_EVAL_RQ", ""),
                            "paper_eval_phase": f"{phase}+post_eval",
                            "paper_eval_kind": kind,
                        },
                    )
                finally:
                    _release(label=f"post_eval:{path.stem}")
            else:
                print(f"  WARN: expected checkpoint missing after train under {env_dir}")
        return

    if kind == "transfer":
        # Transfer configs are phase=eval; handled above. Fallback:
        _run_mod_eval(
            path,
            trial=trial,
            seed=seed,
            eval_episodes=eval_episodes,
            eval_max_frames=eval_max_frames,
        )
        return

    raise SystemExit(f"Unhandled PAPER_EVAL_KIND={kind!r} phase={phase!r} in {path}")


def _resolve_wandb_mode(*, trial: bool, wandb_mode: str | None) -> str:
    if wandb_mode is not None:
        return wandb_mode
    return "disabled" if trial else "online"


def _resolve_jaxatari_root(cli_value: str | None) -> Path | None:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("PAPER_EVAL_JAXATARI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_JAXATARI_FORK.is_dir():
        return DEFAULT_JAXATARI_FORK
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rq", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--phase", choices=["train", "eval", "finetune"], default=None)
    parser.add_argument("--only", nargs="*", metavar="NAME")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_STEMS),
        default=None,
        help=(
            "Curated multi-phase run list (ignores --phase). "
            f"{SKILL_RESHAPE_SUITE}: ego Seaquest train + mod eval "
            "+ Pong/BeamRider transfer (default --eval-max-frames "
            f"{SUITE_EVAL_MAX_FRAMES}). "
            f"{DESTROY_ABLATION_SUITE}: 3 DESTROY reward arms "
            "(y-align / standoff / contact-fail), lean 5-mod Seaquest + Pong base; "
            "use --gpus 0,1,2 so each GPU owns one arm. "
            f"{DESTROY_URGENCY_SUITE}: contact-fail + living tax (no shot ego); "
            "lean 5-mod Seaquest + Pong base. "
            f"{COLLECT_SHOT_PROBES_SUITE}: COLLECT living tax (NAVIGATE still "
            "shaped) and DESTROY shot-between bit; 2 GPUs, one arm each. "
            f"{EGO_POSE_PROBES_SUITE}: pose-only 12-dim vs DESTROY-only extras "
            "(COLLECT/AVOID/NAVIGATE pose, DESTROY pose+LoS+weapon); 2 GPUs. "
            f"{COLLECT_COMBAT_PROBES_SUITE}: urg-v2 clones — COLLECT death "
            "penalty off vs DESTROY-first (COLLECT never fights); 2 GPUs. "
            f"{NO_SURFACE_DEATH_SUITE}: urg-v2 clone trained with "
            "no_surfacing_death (0-diver surface is not a life loss). "
            f"{EGO_CLIP_BINS_SUITE}: urg-v2 + new heuristic; 2×2 of "
            "distance clip 50px and 8 overlapping world-θ bins "
            "(replace sin/cos(ψ)); 4 GPUs, one arm each."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "With --suite: skip train configs (requires existing ckpts). "
            "Valid for skill_reshape, destroy_ablation, destroy_urgency, "
            "collect_shot_probes, ego_pose_probes, collect_combat_probes, "
            "no_surface_death, and ego_clip_bins."
        ),
    )
    parser.add_argument("--shard", metavar="INDEX/TOTAL")
    parser.add_argument(
        "--gpus",
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated GPU ids for a one-shot multi-GPU launch "
            "(e.g. 0,1). Spawns one worker per GPU with --shard i/N --wait-deps "
            "and blocks until all finish."
        ),
    )
    parser.add_argument(
        "--wait-deps",
        action="store_true",
        help=(
            "If RQ1 ckpts required by a run are missing, poll until they appear "
            "instead of failing the RQ gate (for multi-GPU overnight)."
        ),
    )
    parser.add_argument(
        "--wait-deps-poll",
        type=float,
        default=60.0,
        help="Seconds between dependency polls when --wait-deps is set (default: 60).",
    )
    parser.add_argument(
        "--wait-deps-timeout-h",
        type=float,
        default=36.0,
        help="Hours to wait for deps before failing (default: 36).",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            f"Abort immediately if free space on the project volume drops below "
            f"this many GiB (default: {DEFAULT_MIN_FREE_GB}). Checked at suite "
            "start, before each run, and during --wait-deps polls. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--batch-tag",
        default=None,
        help=(
            "W&B batch tag (default: paper_eval_YYYYMMDD_HHMM). "
            "Also set via PAPER_EVAL_BATCH_TAG. Added as batch:<tag> alongside suite:paper_eval."
        ),
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-mods", nargs="?", const="*", metavar="ENV")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--trial-steps", type=int, default=TRIAL_STEPS_DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="Run curated wiring smoke matrix")
    parser.add_argument(
        "--post-eval",
        action="store_true",
        help=(
            "After train/finetune, also run frozen mod eval on the new checkpoint. "
            "Default off (except --smoke). Prefer --phase eval for the paper protocol."
        ),
    )
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record end-of-train videos (default: off; videos burn disk).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Deprecated alias: videos are off by default.",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        help=(
            "Minimize local disk writes: no videos, no training_metrics.csv, "
            "wandb local dirs under $TMPDIR/$USER/cbl_paper_eval."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip experiments that already have their checkpoint OR a complete "
            "mod-eval JSON on disk (all allowlisted mods present, including base). "
            "Missing/partial evals from disk-full failures are re-queued. "
            "Filter runs before sharding so GPUs only share remaining work."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="STEM",
        help=(
            "Force-skip these config stems and any run that requires them as an RQ1 dep "
            "(e.g. --exclude seaquest_skills_pixel also drops pixel transfer/finetune)."
        ),
    )
    parser.add_argument("--abort-on-failure", action="store_true")
    parser.add_argument("--skip-rq-gate", action="store_true")
    parser.add_argument("--jaxatari-root", default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-max-frames", type=int, default=None)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Regenerate RQ2/RQ3 YAMLs then exit",
    )
    args = parser.parse_args()

    if args.materialize:
        import importlib.util

        mat_path = PROJECT_ROOT / "scripts" / "materialize_paper_eval_configs.py"
        spec = importlib.util.spec_from_file_location("materialize_paper_eval_configs", mat_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.main()
        return

    if args.gpus:
        gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
        if len(gpu_ids) < 1:
            raise SystemExit("--gpus needs at least one id (e.g. --gpus 0,1)")
        if args.shard:
            raise SystemExit("Use either --gpus or --shard, not both ( --gpus adds --shard )")
        batch_tag = _resolve_batch_tag(args.batch_tag)
        os.environ["PAPER_EVAL_BATCH_TAG"] = batch_tag
        print(f"paper_eval batch tag: batch:{batch_tag}", flush=True)
        # Parent only orchestrates; workers set jaxatari / CUDA / --lean themselves.
        raise SystemExit(_launch_multi_gpu(sys.argv[1:], gpu_ids))

    if not CONFIG_DIR.is_dir():
        raise SystemExit(f"Missing {CONFIG_DIR}")

    if args.list_mods is not None:
        _print_mods(None if args.list_mods == "*" else args.list_mods)
        return

    _prepend_jaxatari_root(_resolve_jaxatari_root(args.jaxatari_root))

    only = list(args.only) if args.only else None
    suite = str(args.suite).strip() if args.suite else None
    phase_filter = args.phase
    if args.smoke:
        only = list(SMOKE_STEMS)
    elif suite:
        if args.only:
            raise SystemExit("Use either --suite or --only, not both")
        if args.eval_only:
            only = []
            for s in SUITE_STEMS[suite]:
                matches = list(CONFIG_DIR.rglob(f"{s}.yaml"))
                if not matches:
                    raise SystemExit(f"Config(s) not found: {[s]}")
                phase = str(_load_yaml(matches[0]).get("PAPER_EVAL_PHASE", "train")).lower()
                if phase != "train":
                    only.append(s)
        else:
            only = list(SUITE_STEMS[suite])
        # Suites span train+eval; do not filter by --phase.
        phase_filter = None
        print(f"paper_eval suite={suite} ({len(only)} run(s))", flush=True)
    elif args.eval_only:
        raise SystemExit("--eval-only requires --suite")

    paths = _discover_configs(rq=args.rq, only=only, phase=phase_filter)
    if not paths:
        raise SystemExit("No matching configs (run --materialize if RQ2/RQ3 missing)")

    # Preserve smoke / suite / --only order (discovery is alphabetical).
    if only:
        by_stem = {p.stem: p for p in paths}
        ordered = [by_stem[s] for s in only if s in by_stem]
        missing = [s for s in only if s not in by_stem]
        if missing:
            raise SystemExit(f"Config(s) not found: {missing}")
        paths = ordered
    else:
        # Overnight / multi-GPU: ungated RQ1 trains before gated eval/RQ2/RQ3.
        paths = sorted(paths, key=_schedule_key)

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    exclude = {str(s).removesuffix(".yaml") for s in (args.exclude or [])}
    if args.resume or exclude:
        paths = _filter_remaining_paths(
            paths,
            seed=seed,
            exclude=exclude,
            skip_done=bool(args.resume),
        )
        if not paths:
            raise SystemExit("Nothing left to run (all done or excluded).")

    if args.shard:
        idx, n = _parse_shard(args.shard)
        paths = _shard_paths(paths, shard_idx=idx, num_shards=n)
        if not paths:
            raise SystemExit(f"Shard {args.shard} empty")

    if args.list:
        _print_matrix(paths)
        if args.resume or exclude:
            print("\n(status with --resume / --exclude applied above)")
        return

    # Smoke / skill-reshape train first in-order; gate per-run instead of upfront.
    # --wait-deps: skip the upfront fail; each run polls instead.
    suite_will_train = bool(suite and not args.eval_only)
    skip_gate = bool(
        args.skip_rq_gate or args.smoke or args.wait_deps or suite_will_train
    )
    if not skip_gate:
        gated = [
            p
            for p in paths
            if _load_yaml(p).get("PAPER_EVAL_REQUIRES_RQ1")
            or (
                int(_load_yaml(p).get("PAPER_EVAL_RQ", 1)) >= 2
                and str(_load_yaml(p).get("PAPER_EVAL_PHASE")) != "train"
            )
            or str(_load_yaml(p).get("PAPER_EVAL_KIND", "")).lower() in {"meta", "tandem"}
        ]
        if gated:
            _check_rq_gate(gated, seed=seed)

    wandb_mode = _resolve_wandb_mode(trial=args.trial or args.smoke, wandb_mode=args.wandb_mode)
    trial_steps = args.trial_steps if (args.trial or args.smoke) else None
    trial = bool(args.trial or args.smoke)
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else (1 if trial else 3)
    if args.eval_max_frames is not None:
        eval_max_frames = int(args.eval_max_frames)
    elif trial:
        eval_max_frames = 400
    elif suite in {
        SKILL_RESHAPE_SUITE,
        DESTROY_ABLATION_SUITE,
        DESTROY_URGENCY_SUITE,
        COLLECT_SHOT_PROBES_SUITE,
        EGO_POSE_PROBES_SUITE,
        COLLECT_COMBAT_PROBES_SUITE,
        NO_SURFACE_DEATH_SUITE,
        EGO_CLIP_BINS_SUITE,
    }:
        eval_max_frames = SUITE_EVAL_MAX_FRAMES
    else:
        eval_max_frames = 800
    batch_tag = _resolve_batch_tag(args.batch_tag)
    os.environ["PAPER_EVAL_BATCH_TAG"] = batch_tag
    print(f"paper_eval batch tag: batch:{batch_tag}")
    if suite:
        print(f"paper_eval eval_max_frames={eval_max_frames}", flush=True)

    lean = bool(args.lean)
    if lean:
        _configure_lean_disk()

    # Videos off by default; --video opts in. --lean always disables videos.
    record_video = bool(args.video) and not lean and not trial and not args.no_video
    save_training_metrics_csv = not lean

    min_free_gb = float(args.min_free_gb)
    _assert_disk_space(min_free_gb=min_free_gb, where="suite start")
    print(
        f"disk: {_free_gb():.2f} GiB free (abort below {min_free_gb:.2f} GiB); "
        f"videos={'on' if record_video else 'off'}; "
        f"training_csv={'on' if save_training_metrics_csv else 'off'}"
    )

    print(f"paper_eval: {len(paths)} run(s)")
    for p in paths:
        print(f"  - {p.relative_to(CONFIG_DIR)}")
    print()

    _init_rtpt_for_paths(
        paths,
        experiment_name=_rtpt_name(args.shard, args.rq, suite=suite),
        trial_steps=trial_steps,
        trial=trial,
    )

    failures: list[tuple[str, str]] = []
    failed_stems: set[str] = set()
    for path in paths:
        try:
            _assert_disk_space(
                min_free_gb=min_free_gb,
                where=f"before {path.stem}",
            )
            if args.wait_deps:
                _wait_for_path_deps(
                    path,
                    seed=seed,
                    poll_s=float(args.wait_deps_poll),
                    timeout_h=float(args.wait_deps_timeout_h),
                    min_free_gb=min_free_gb,
                    failed_local_stems=failed_stems,
                )
            # Reclaim VRAM left by the previous JAX train/eval before compiling again.
            _release(label=f"before:{path.stem}")
            _run_experiment(
                path,
                wandb_mode=wandb_mode,
                trial_steps=trial_steps,
                record_video=record_video,
                seed=args.seed,
                trial=trial,
                eval_episodes=eval_episodes,
                eval_max_frames=eval_max_frames,
                enforce_rq_gate=bool(args.smoke) and not args.skip_rq_gate,
                post_eval=bool(args.post_eval or args.smoke),
                batch_tag=batch_tag,
                save_training_metrics_csv=save_training_metrics_csv,
            )
        except SystemExit:
            raise
        except Exception as exc:
            traceback.print_exc()
            failures.append((path.name, str(exc)))
            failed_stems.add(path.stem)
            if _is_no_space_error(exc):
                print(
                    f"\n  FATAL disk full during {path.name}; aborting suite "
                    f"(not waiting on further deps).\n",
                    flush=True,
                )
                break
            if args.abort_on_failure:
                break
            print(f"\n  FAILED (continuing): {path.name}: {exc}\n")

    if failures:
        print("\nFailed runs:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    print("\nAll paper_eval runs finished successfully.")


if __name__ == "__main__":
    main()

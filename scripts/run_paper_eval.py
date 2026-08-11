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
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
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


def _rtpt_name(shard: str | None, rq: int | None) -> str:
    base = RTPT_BASE if rq is None else f"{RTPT_BASE}_rq{rq}"
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


def _apply_run_defaults(
    cfg: dict[str, Any],
    *,
    wandb_mode: str | None,
    trial_steps: int | None,
    record_video: bool,
    seed: int | None,
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


def _skill_ckpt_from_save(save_path: str | Path, env_name: str, seed: int) -> Path:
    root = Path(save_path)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / env_name / f"hierarchical_pqn_{env_name}_seed{seed}.safetensors"


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


def _updates_for_config_path(path: Path, *, trial_steps: int | None) -> int:
    raw = _load_yaml(path)
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    if phase == "eval" or kind == "heuristic_random":
        return 0
    if phase == "eval":
        return 0
    cfg = _flatten_experiment_cfg(copy.deepcopy(raw))
    if trial_steps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(trial_steps)
        _align_trial_timesteps(cfg)
    from concept_based_learning.training.rtpt_progress import num_updates_from_timesteps

    return num_updates_from_timesteps(
        int(cfg.get("TOTAL_TIMESTEPS", 0)),
        num_steps=int(cfg.get("NUM_STEPS", 128)),
        num_envs=int(cfg.get("NUM_ENVS", 128)),
    )


def _init_rtpt_for_paths(paths: list[Path], *, experiment_name: str, trial_steps: int | None) -> None:
    from concept_based_learning.training.rtpt_progress import start_rtpt

    total = max(1, sum(_updates_for_config_path(p, trial_steps=trial_steps) for p in paths))
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
        print(f"RTPT initialized: {total:,} updates across {len(paths)} run(s)\n")


def _clear_jax_memory() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def _release() -> None:
    gc.collect()
    _clear_jax_memory()


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


def _architecture_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Load arch YAML referenced by PAPER_EVAL_ARCHITECTURE_OF if present."""
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
    return _flatten_experiment_cfg(copy.deepcopy(_load_yaml(ref_path)))


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
        _release()


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
        _release()


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
            },
        )
    finally:
        _release()


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
) -> None:
    raw = _load_yaml(path)
    kind = str(raw.get("PAPER_EVAL_KIND", "skills")).lower()
    phase = str(raw.get("PAPER_EVAL_PHASE", "train")).lower()
    status = str(raw.get("PAPER_EVAL_STATUS", "ready")).lower()
    if status in {"stub", "planned"}:
        print(f"\n  SKIP (status={status}): {path.stem}")
        return

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
                    _release()
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
    parser.add_argument("--shard", metavar="INDEX/TOTAL")
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
    parser.add_argument("--no-video", action="store_true")
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

    if not CONFIG_DIR.is_dir():
        raise SystemExit(f"Missing {CONFIG_DIR}")

    if args.list_mods is not None:
        _print_mods(None if args.list_mods == "*" else args.list_mods)
        return

    _prepend_jaxatari_root(_resolve_jaxatari_root(args.jaxatari_root))

    only = list(args.only) if args.only else None
    if args.smoke:
        only = list(SMOKE_STEMS)

    paths = _discover_configs(rq=args.rq, only=only, phase=args.phase)
    if not paths:
        raise SystemExit("No matching configs (run --materialize if RQ2/RQ3 missing)")

    # Preserve smoke / --only order (discovery is alphabetical).
    if only:
        by_stem = {p.stem: p for p in paths}
        ordered = [by_stem[s] for s in only if s in by_stem]
        missing = [s for s in only if s not in by_stem]
        if missing:
            raise SystemExit(f"Config(s) not found: {missing}")
        paths = ordered

    if args.shard:
        idx, n = _parse_shard(args.shard)
        paths = _shard_paths(paths, shard_idx=idx, num_shards=n)
        if not paths:
            raise SystemExit(f"Shard {args.shard} empty")

    if args.list:
        _print_matrix(paths)
        return

    seed = args.seed if args.seed is not None else DEFAULT_SEED
    # Smoke trains RQ1 first in-order; gate per-run instead of upfront.
    skip_gate = bool(args.skip_rq_gate or args.smoke)
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
    eval_max_frames = (
        args.eval_max_frames if args.eval_max_frames is not None else (400 if trial else 800)
    )

    print(f"paper_eval: {len(paths)} run(s)")
    for p in paths:
        print(f"  - {p.relative_to(CONFIG_DIR)}")
    print()

    _init_rtpt_for_paths(
        paths, experiment_name=_rtpt_name(args.shard, args.rq), trial_steps=trial_steps
    )

    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            _run_experiment(
                path,
                wandb_mode=wandb_mode,
                trial_steps=trial_steps,
                record_video=not args.no_video and not trial,
                seed=args.seed,
                trial=trial,
                eval_episodes=eval_episodes,
                eval_max_frames=eval_max_frames,
                enforce_rq_gate=bool(args.smoke) and not args.skip_rq_gate,
                post_eval=bool(args.post_eval or args.smoke),
            )
        except Exception as exc:
            traceback.print_exc()
            failures.append((path.name, str(exc)))
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

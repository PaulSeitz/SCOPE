"""Greedy mod evaluation for hierarchical / flat PQN / heuristic_random agents."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from concept_based_learning.baselines.flat_pqn import (
    run_flat_pqn_greedy_eval,
    run_random_action_eval,
)
from concept_based_learning.paper_eval.mods import ModSpec, mod_specs_for_env
from concept_based_learning.training.rtpt_progress import (
    RTPT_EVAL_UNITS_PER_MOD,
    rtpt_add_units,
)
from concept_based_learning.pqn_visualizer import (
    align_config_with_skill_checkpoint,
    load_params_from_file,
    run_greedy_eval_episodes,
)


def merge_train_and_eval_game_mods(
    config: dict[str, Any], eval_mods: list[str] | tuple[str, ...] | None
) -> list[str] | None:
    """Keep training GAME_MODS on eval so persistent env mods are not wiped.

    Eval specs replace GAME_MODS; if the agent was trained with e.g.
    ``no_surfacing_death``, that mod must still apply on ``base`` / OOD keys.

    Does **not** enable CBL label remaps — those are driven only by
    ``ModSpec.relabel`` via ``_apply_eval_mod_label_overrides``.
    """
    train = [str(m) for m in (config.get("GAME_MODS") or []) if m]
    extra = [str(m) for m in (eval_mods or []) if m]
    merged = list(dict.fromkeys(train + extra))
    return merged or None


def _apply_eval_mod_label_overrides(
    eval_cfg: dict[str, Any], mod: ModSpec
) -> dict[str, Any]:
    """Eval-only semantic label overrides for architecture adaptability (E2).

    Driven solely by ``mod.relabel`` from the allowlist YAML — never by whether
    ``swap_diver_enemy_labels`` / ``swap_diver_shark_roles`` appear in merged
    GAME_MODS. Sets ``_PAPER_EVAL_MOD_RELABEL`` so env_manager will not apply
    remaps on train-time ``test_env`` builds that lack this sentinel.
    """
    eval_cfg["_PAPER_EVAL_MOD_RELABEL"] = True
    # Explicit False on every condition so a prior True cannot stick via shared dicts.
    eval_cfg["SEAQUEST_SWAP_DIVER_ENEMY_LABELS"] = False
    eval_cfg["SEAQUEST_SWAP_DIVER_SHARK_ROLES"] = False
    if mod.relabel:
        if mod.key == "swap_diver_shark_roles":
            eval_cfg["SEAQUEST_SWAP_DIVER_SHARK_ROLES"] = True
        else:
            # OC/sprite swap path (swap_diver_enemy_labels and any future relabel keys).
            eval_cfg["SEAQUEST_SWAP_DIVER_ENEMY_LABELS"] = True
    return eval_cfg


def _rtpt_tick_eval_mod() -> None:
    """Count one completed greedy-mod eval toward the suite RTPT bar."""
    rtpt_add_units(RTPT_EVAL_UNITS_PER_MOD)


def _promote_unclipped_score(stats: dict[str, Any]) -> None:
    """Keep the unclipped reward sum as ``mean``. Engine score stays diagnostic."""
    if "mean" in stats and "mean_reward_sum" not in stats:
        stats["mean_reward_sum"] = stats["mean"]


def _attach_drop_from_base(results: dict[str, dict[str, Any]]) -> None:
    for stats in results.values():
        _promote_unclipped_score(stats)

    base = results.get("base")
    if not base or "mean" not in base:
        return
    base_mean = float(base["mean"])
    for key, stats in results.items():
        if key == "base" or "mean" not in stats:
            continue
        mean = float(stats["mean"])
        stats["drop_from_base"] = mean - base_mean
        stats["drop_frac_from_base"] = (
            (mean - base_mean) / abs(base_mean) if abs(base_mean) > 1e-8 else float("nan")
        )


def _parallel_mods_enabled(config: dict[str, Any], n_mods: int) -> bool:
    if os.environ.get("CBL_EVAL_MOD_WORKER") == "1":
        return False
    if n_mods <= 1:
        return False
    return bool(config.get("EVAL_PARALLEL_MODS", True))


def _max_parallel_mods() -> int:
    """How many modification workers may compile at once.

    Nine at once exhausted CUDA-graph memory on a full 48 GB card.
    """
    raw = os.environ.get("CBL_EVAL_MAX_PARALLEL_MODS", "2")
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, n)


def _mod_mem_fraction(n_mods: int) -> str:
    """Per-process cap so N workers cannot each grab the whole GPU."""
    return f"{min(0.22, 0.70 / max(int(n_mods), 1)):.3f}"


def _run_skill_mods_in_subprocesses(
    config: dict[str, Any],
    *,
    skills_path: str | Path,
    meta_path: str | Path | None,
    mod_specs: list[ModSpec],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
) -> dict[str, Any]:
    """One process per mod. Same ``run_hierarchical_mod_eval`` as the serial path."""
    import pickle
    import subprocess
    import sys
    import tempfile

    # Optional CBL_EVAL_SKILL_BATCH runs mods in smaller groups; unset keeps all at once.
    width = max(1, int(os.environ.get("CBL_EVAL_SKILL_BATCH") or len(mod_specs)))
    width = min(width, len(mod_specs))
    frac = _mod_mem_fraction(width)
    print(
        f"  Parallel mods: {len(mod_specs)} total, {width} at a time, "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={frac} (preallocate off)",
        flush=True,
    )
    tmp = Path(tempfile.mkdtemp(prefix="cbl_eval_mods_"))
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    failed: list[tuple[str, int]] = []
    for index, mod in enumerate(mod_specs):
        job = {
            "config": config,
            "skills_path": str(skills_path),
            "meta_path": str(meta_path) if meta_path else None,
            "mod": {
                "key": mod.key,
                "mods": list(mod.mods),
                "category": mod.category,
                "relabel": bool(mod.relabel),
            },
            "num_episodes": int(num_episodes),
            "num_seeds": int(num_seeds),
            "base_seed": int(base_seed),
            "max_frames": int(max_frames),
        }
        job_path = tmp / f"{mod.key}.pkl"
        out_path = tmp / f"{mod.key}.json"
        job["out"] = str(out_path)
        job_path.write_bytes(pickle.dumps(job))
        env = os.environ.copy()
        env["CBL_EVAL_MOD_WORKER"] = "1"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = frac
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env.setdefault("WANDB_MODE", "disabled")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "concept_based_learning.paper_eval.eval_mod_worker",
                str(job_path),
            ],
            env=env,
        )
        procs.append((mod.key, proc, out_path))
        if (index + 1) % width == 0 or index + 1 == len(mod_specs):
            for key, running, path in procs[-(index % width + 1) :]:
                rc = running.wait()
                if rc != 0 or not path.is_file():
                    failed.append((key, int(rc)))
    if failed:
        raise RuntimeError(f"parallel mod workers failed: {failed} (dir {tmp})")

    results: dict[str, Any] = {}
    for _key, _proc, out_path in procs:
        piece = json.loads(out_path.read_text(encoding="utf-8"))
        results.update(piece)
    _attach_drop_from_base(results)
    return results


def _run_flat_pqn_mods_in_subprocesses(
    *,
    flat_pqn_path: str | Path,
    config: dict[str, Any],
    env_name: str,
    object_centric: bool,
    mod_specs: list[ModSpec],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
) -> dict[str, Any]:
    """One process per mod. Same ``run_flat_pqn_greedy_eval`` as the serial path."""
    import pickle
    import subprocess
    import sys
    import tempfile

    width = min(_max_parallel_mods(), len(mod_specs))
    frac = _mod_mem_fraction(width)
    print(
        f"  Parallel mods: {len(mod_specs)} total, {width} at a time, "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={frac} (preallocate off)",
        flush=True,
    )
    tmp = Path(tempfile.mkdtemp(prefix="cbl_eval_pqn_mods_"))
    procs: list[tuple[ModSpec, subprocess.Popen, Path]] = []
    failed: list[tuple[str, int]] = []
    for start in range(0, len(mod_specs), width):
        batch = mod_specs[start : start + width]
        running: list[tuple[ModSpec, subprocess.Popen, Path]] = []
        for mod in batch:
            merged = merge_train_and_eval_game_mods(config, mod.mods)
            out_path = tmp / f"{mod.key}.json"
            job = {
                "model_path": str(flat_pqn_path),
                "game_mods": list(merged or ()),
                "num_episodes": int(num_episodes),
                "num_seeds": int(num_seeds),
                "base_seed": int(base_seed),
                "max_frames": int(max_frames),
                "object_centric": bool(object_centric),
                "env_name": str(env_name),
                "out": str(out_path),
            }
            job_path = tmp / f"{mod.key}.pkl"
            job_path.write_bytes(pickle.dumps(job))
            env = os.environ.copy()
            env["CBL_EVAL_MOD_WORKER"] = "1"
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = frac
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("WANDB_MODE", "disabled")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "concept_based_learning.paper_eval.eval_flat_pqn_worker",
                    str(job_path),
                ],
                env=env,
            )
            running.append((mod, proc, out_path))
        for mod, proc, out_path in running:
            rc = proc.wait()
            procs.append((mod, proc, out_path))
            if rc != 0 or not out_path.is_file():
                failed.append((mod.key, int(rc)))
    if failed:
        raise RuntimeError(f"parallel flat PQN mod workers failed: {failed} (dir {tmp})")

    results: dict[str, Any] = {}
    for mod, _proc, out_path in procs:
        stats = json.loads(out_path.read_text(encoding="utf-8"))
        stats["mods"] = list(merge_train_and_eval_game_mods(config, mod.mods) or ())
        stats["category"] = mod.category
        results[mod.key] = stats
        _rtpt_tick_eval_mod()
    _attach_drop_from_base(results)
    return results


def run_hierarchical_mod_eval(
    config: dict[str, Any],
    *,
    params_tree=None,
    meta_params=None,
    mod_specs: list[ModSpec],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for mod in mod_specs:
        eval_cfg = copy.deepcopy(config)
        merged = merge_train_and_eval_game_mods(config, mod.mods)
        eval_cfg["GAME_MODS"] = merged
        eval_cfg.pop("EVAL_GAME_MODS", None)
        _apply_eval_mod_label_overrides(eval_cfg, mod)
        relabel_s = " relabel=on" if mod.relabel else ""
        print(f"\n--- hierarchical / {mod.key} (mods={merged or 'none'}){relabel_s} ---")
        stats = run_greedy_eval_episodes(
            eval_cfg,
            params_tree=params_tree,
            meta_params=meta_params,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            progress=True,
        )
        stats["mods"] = list(merged or ())
        stats["category"] = mod.category
        stats["relabel"] = bool(mod.relabel)
        results[mod.key] = stats
        _rtpt_tick_eval_mod()
    _attach_drop_from_base(results)
    return results


def _overlay_ego_flags_from_skill_config(
    cfg: dict[str, Any], skills_path: str | Path | None
) -> dict[str, Any]:
    """
    Match ego observation channels to the skill checkpoint training config.

    Cross-env transfer must keep the *source* ego feature flags (weapon/blocking/…)
    even when the target env would not enable them natively.
    """
    if skills_path is None:
        return cfg
    ckpt = Path(str(skills_path))
    # main.py: {stem}_config.yaml next to the safetensors
    candidates = [
        ckpt.with_name(ckpt.stem + "_config.yaml"),
        ckpt.with_name(ckpt.name.replace(".safetensors", "_config.yaml")),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            continue
        alg = raw.pop("alg", {}) or {}
        src = {**raw, **alg} if isinstance(alg, dict) else raw
        for key in (
            "USE_LEGACY_EGO_OBS",
            "USE_EGO_SIZE_FEATURES",
            "USE_EGO_VELOCITY_FEATURES",
            "USE_EGO_BLOCKING_FEATURES",
            "USE_EGO_WEAPON_FEATURES",
            "USE_EGO_FACING_ENEMY_FEATURES",
            "USE_EGO_SHOT_FEATURES",
            "USE_EGO_SHOT_BETWEEN",
            "USE_DESTROY_ONLY_EGO_EXTRAS",
            "EGO_DISTANCE_CLIP_MAX_PX",
            "EGO_DISTANCE_CLIP_CHOICES_PX",
            "EGO_DISTANCE_MODE",
            "EGO_AABB_GEOM",
            "EGO_DISTANCE_DELTA_SIGN_EPS_PX",
            "EGO_DISTANCE_BINS",
            "EGO_DISTANCE_BINS_SCALAR",
            "EGO_DISTANCE_BIN_CENTERS_PX",
            "EGO_DISTANCE_BIN_MAX_PX",
            "EGO_DISTANCE_FLAG_THRESHOLDS_PX",
            "EGO_DISTANCE_HARD_EDGES_PX",
            "EGO_ANGLE_BINS",
            "EGO_ANGLE_BINS_SCALAR",
            "EGO_ANGLE_SINCOS_SNAP",
            "EGO_PSI_ANGLE_BINS",
            "EGO_PSI_ANGLE_SCALAR",
            "EGO_POSE_XY",
            "USE_POV_RELATIVE_MAPPER",
            "EGO_OBSERVATION_FRAME",
            "EGO_ACTION_FRAME",
            "USE_POV_FIXED_BILATERAL",
            "USE_POV_FACING_FRAME",
            "FRAME_STACK_SIZE",
            "SKILL_INPUT_MODE",
        ):
            if key not in src:
                continue
            val = src[key]
            # Sidecar null must not wipe explicit eval-yaml knobs (e.g. clip 50).
            if val is None and cfg.get(key) is not None:
                continue
            cfg[key] = val
        print(f"  Overlayed ego flags from {path.name}")
        break
    return cfg


def run_mod_eval(
    *,
    agent_kind: str,
    config: dict[str, Any],
    env_name: str,
    skills_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    flat_pqn_path: str | Path | None = None,
    object_centric: bool = True,
    paper_core_only: bool = False,
    trial: bool = False,
    mod_keys: list[str] | None = None,
    mods_allowlist: str | None = None,
    num_episodes: int = 3,
    num_seeds: int = 1,
    base_seed: int = 44,
    max_frames: int = 800,
    output_json: str | Path | None = None,
    output_csv_stem: str | Path | None = None,
    master_csv_dir: str | Path | None = None,
    extra_meta: dict[str, Any] | None = None,
    wandb_mode: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate one agent across the env's mod allowlist.

    ``agent_kind``:
      - ``skills`` / ``hierarchical``: load skills (+ optional meta)
      - ``heuristic_random``: heuristic meta + randomly initialized skills
      - ``fixed_skills`` / ``fixed_ego_rules`` / ``constant_ego``: same heuristic
        scaffolding with hand-coded ego-action rules (no skill checkpoint)
      - ``flat_pqn``: flat PQN checkpoint
      - ``random_actions``: uniform random over the minimal action set (no ckpt)

    ``mods_allowlist``: optional paper_eval mods YAML stem (e.g. ``seaquest_visual_ood``)
    when it differs from ``env_name``. Defaults to ``env_name``.

    Always writes CSV when ``output_csv_stem`` or ``output_json`` is set:
      ``{stem}_summary.csv`` and ``{stem}_episodes.csv``, plus appends into
      ``master_csv_dir/all_mod_eval_{summary,episodes}.csv`` when provided.

    After eval, writes ``mod_eval_*_summary.txt`` next to the evaluated
    checkpoint and best-effort resumes that train's wandb run (via
    ``wandb_run.json``) to attach the same scores.
    """
    from concept_based_learning.paper_eval.csv_export import write_mod_eval_csvs

    allowlist = str(mods_allowlist or env_name)
    mod_specs = mod_specs_for_env(
        allowlist,
        paper_core_only=paper_core_only,
        trial=trial,
        keys=mod_keys,
    )
    if not mod_specs:
        raise ValueError(f"No mods selected for allowlist={allowlist} env={env_name}")

    kind = agent_kind.lower().strip()
    payload: dict[str, Any] = {
        "agent_kind": kind,
        "env": env_name,
        "skills_path": str(skills_path) if skills_path else "",
        "meta_path": str(meta_path) if meta_path else "",
        "flat_pqn_path": str(flat_pqn_path) if flat_pqn_path else "",
        "object_centric": bool(object_centric),
        "base_seed": int(base_seed),
        "num_episodes": int(num_episodes),
        "num_seeds": int(num_seeds),
        "max_frames": int(max_frames),
        "skill_input_mode": str(config.get("SKILL_INPUT_MODE", "")),
        "meta_input_mode": str(config.get("META_INPUT_MODE", "")),
        "mods": [
            {
                "key": m.key,
                "mods": list(m.mods),
                "category": m.category,
                "relabel": bool(m.relabel),
            }
            for m in mod_specs
        ],
    }
    if extra_meta:
        payload.update(extra_meta)

    if kind in {
        "skills",
        "hierarchical",
        "transfer",
        "heuristic_random",
        "fixed_skills",
        "fixed_ego_rules",
        "constant_ego",
    }:
        cfg = copy.deepcopy(config)
        params_tree = None
        meta_params = None
        if kind == "constant_ego":
            cfg["SKILL_POLICY_MODE"] = "constant_ego"
        elif kind == "fixed_ego_rules":
            cfg["SKILL_POLICY_MODE"] = "fixed_ego_rules"
        elif kind == "fixed_skills":
            # Kind is shared by both arms; honor YAML SKILL_POLICY_MODE.
            mode = str(cfg.get("SKILL_POLICY_MODE", "fixed_ego_rules")).lower()
            if mode not in {"fixed_ego_rules", "constant_ego"}:
                mode = "fixed_ego_rules"
            cfg["SKILL_POLICY_MODE"] = mode
        if kind == "tandem" and (
            meta_path is None or not Path(str(meta_path)).is_file()
        ):
            raise FileNotFoundError(
                "tandem eval requires the learned selector checkpoint, "
                f"missing {meta_path}"
            )
        if kind not in {
            "heuristic_random",
            "fixed_skills",
            "fixed_ego_rules",
            "constant_ego",
        }:
            if skills_path is None:
                raise ValueError(f"{kind} eval requires skills_path")
            params_tree = load_params_from_file(str(skills_path))
            cfg = _overlay_ego_flags_from_skill_config(cfg, skills_path)
            cfg = align_config_with_skill_checkpoint(cfg, params_tree)
            if meta_path is not None and Path(str(meta_path)).is_file():
                meta_params = load_params_from_file(str(meta_path))
        # Learned skills reload from disk in each worker. Heuristic / fixed
        # arms stay in-process (no checkpoint to reopen).
        if (
            skills_path is not None
            and _parallel_mods_enabled(cfg, len(mod_specs))
        ):
            results = _run_skill_mods_in_subprocesses(
                cfg,
                skills_path=skills_path,
                meta_path=meta_path,
                mod_specs=mod_specs,
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_frames=max_frames,
            )
        else:
            results = run_hierarchical_mod_eval(
                cfg,
                params_tree=params_tree,
                meta_params=meta_params,
                mod_specs=mod_specs,
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_frames=max_frames,
            )
        payload["results"] = results
        payload["skill_policy_mode"] = str(cfg.get("SKILL_POLICY_MODE", "learned"))

    elif kind == "flat_pqn":
        if flat_pqn_path is None:
            raise ValueError("flat_pqn eval requires flat_pqn_path")
        if _parallel_mods_enabled(config, len(mod_specs)):
            results = _run_flat_pqn_mods_in_subprocesses(
                flat_pqn_path=flat_pqn_path,
                config=config,
                env_name=str(env_name),
                object_centric=object_centric,
                mod_specs=mod_specs,
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_frames=max_frames,
            )
        else:
            results = {}
            for mod in mod_specs:
                merged = merge_train_and_eval_game_mods(config, mod.mods)
                print(f"\n--- flat_pqn / {mod.key} (mods={merged or 'none'}) ---")
                stats = run_flat_pqn_greedy_eval(
                    model_path=str(flat_pqn_path),
                    game_mods=merged or [],
                    num_episodes=num_episodes,
                    num_seeds=num_seeds,
                    base_seed=base_seed,
                    max_frames=max_frames,
                    object_centric=object_centric,
                    env_name=str(env_name),
                    progress=True,
                )
                stats["mods"] = list(merged or ())
                stats["category"] = mod.category
                results[mod.key] = stats
                _rtpt_tick_eval_mod()
            _attach_drop_from_base(results)
        payload["results"] = results
    elif kind == "random_actions":
        results = {}
        for mod in mod_specs:
            merged = merge_train_and_eval_game_mods(config, mod.mods)
            print(f"\n--- random_actions / {mod.key} (mods={merged or 'none'}) ---")
            stats = run_random_action_eval(
                config=config,
                game_mods=merged or [],
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_frames=max_frames,
                env_name=str(env_name),
                progress=True,
            )
            stats["mods"] = list(merged or ())
            stats["category"] = mod.category
            results[mod.key] = stats
            _rtpt_tick_eval_mod()
        _attach_drop_from_base(results)
        payload["results"] = results
    else:
        raise ValueError(f"Unknown agent_kind={agent_kind!r}")

    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote mod eval JSON → {out}")

    csv_stem = output_csv_stem
    if csv_stem is None and output_json is not None:
        csv_stem = Path(output_json).with_suffix("")
    if csv_stem is not None:
        master = master_csv_dir
        if master is None:
            master = Path(csv_stem).resolve().parent
        write_mod_eval_csvs(
            payload,
            output_stem=csv_stem,
            extra_meta=extra_meta,
            master_dir=master,
        )

    # Compact summary
    print("\nMod eval summary:")
    for key, stats in payload["results"].items():
        drop = stats.get("drop_from_base")
        drop_s = f"  drop={drop:+.1f}" if drop is not None else ""
        print(f"  {key:28s} mean={stats['mean']:.2f}±{stats['std']:.2f}{drop_s}")

    # Text summary next to the evaluated checkpoint (+ optional train-run wandb sync).
    try:
        from concept_based_learning.paper_eval.mod_eval_report import (
            publish_mod_eval_report,
        )

        meta = dict(extra_meta or {})
        meta.setdefault("agent_kind", kind)
        meta.setdefault("env_name", env_name)
        publish_mod_eval_report(
            payload,
            skills_path=skills_path,
            meta_path=meta_path,
            flat_pqn_path=flat_pqn_path,
            save_path=config.get("SAVE_PATH"),
            env_name=env_name,
            config_stem=str(meta.get("config_stem") or "unknown"),
            extra_meta=meta,
            wandb_mode=wandb_mode
            or os.environ.get("PAPER_EVAL_WANDB_MODE")
            or os.environ.get("WANDB_MODE"),
        )
    except Exception as exc:  # noqa: BLE001 — never fail the eval on reporting
        print(f"  WARN: mod_eval report publish failed: {exc}")

    return payload

#!/usr/bin/env python3
"""
Compile smoke for tandem JIT split regions on dual-pixel ablation configs.

Verifies that split-region compilation no longer OOMs on heavy configs where the
monolithic ``jit(vmap(train))`` graph exceeded ~48GB HBM at compile time.

Usage (from project root):
  python scripts/compile_tandem_regions_smoke.py --num-envs 256
  python scripts/compile_tandem_regions_smoke.py --num-envs 512
  python scripts/compile_tandem_regions_smoke.py --num-envs 256 --run-one-update
  python scripts/compile_tandem_regions_smoke.py --config seaquest_mi_pixel_embed_si_pixel_embed --num-envs 4
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any

# Reduce cudnn autotune memory spikes for pixel_embed tandem runs.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import outer_make_tandem_train  # noqa: E402

INPUT_ABLATION_DIR = PROJECT_ROOT / "configs" / "experiment" / "input_ablation"
DEFAULT_CONFIG_STEM = "seaquest_mi_pixel_embed_si_pixel_embed"

SPLIT_REGION_KEYS = ("collect_jit", "meta_jit", "skills_jit", "init_fn", "compile_regions")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_experiment_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    alg = cfg.pop("alg", {})
    return {**cfg, **alg}


def _resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.is_file():
        return path
    candidate = INPUT_ABLATION_DIR / f"{config_arg}.yaml"
    if candidate.is_file():
        return candidate
    candidate = INPUT_ABLATION_DIR / config_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Config not found: {config_arg!r} (tried {candidate} and {path})"
    )


def _smoke_overrides(*, num_envs: int, num_steps: int, run_one_update: bool) -> dict[str, Any]:
    total_timesteps = int(num_envs) * int(num_steps)
    return {
        "WANDB_MODE": "disabled",
        "RTPT_ENABLED": False,
        "RECORD_VIDEO": False,
        "NUM_SEEDS": 1,
        "SEED": 42,
        "NUM_ENVS": int(num_envs),
        "NUM_STEPS": int(num_steps),
        "NUM_EPOCHS": 1,
        "META_NUM_EPOCHS": 1,
        "TOTAL_TIMESTEPS": total_timesteps if run_one_update else int(num_envs) * int(num_steps),
        "CURRICULUM_ENABLED": False,
    }


def _has_accelerator() -> bool:
    try:
        return any(
            d.platform in ("gpu", "cuda", "rocm", "tpu") for d in jax.devices()
        )
    except Exception:
        return False


def _unpack_runner_state(runner_state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Mirror ``train_tandem._unpack_runner_state`` for compile shape warmup."""
    (
        meta_state,
        skill_states,
        expl_state,
        running_returns,
        returned_episode_returns,
        returned_episode_lengths,
        episode_lengths,
        full_game_running,
        returned_game_returns,
        returned_game_lengths,
        game_lengths,
        rng,
    ) = runner_state
    collect_carry = (
        expl_state,
        running_returns,
        returned_episode_returns,
        returned_episode_lengths,
        episode_lengths,
        full_game_running,
        returned_game_returns,
        returned_game_lengths,
        game_lengths,
    )
    return meta_state, skill_states, collect_carry, rng


def _compile_regions_timed(
    bundle: dict[str, Any],
    rng: Any,
    *,
    init_meta_params: Any = None,
    init_skill_params: Any = None,
) -> dict[str, float]:
    """Compile each region once; time first lower().compile() or first invocation."""
    init_fn = bundle["init_fn"]
    collect_jit = bundle["collect_jit"]
    meta_jit = bundle["meta_jit"]
    skills_jit = bundle["skills_jit"]
    metrics_jit = bundle["metrics_jit"]

    timings: dict[str, float] = {}

    t0 = time.time()
    init_fn.lower(rng, init_meta_params, init_skill_params).compile()
    timings["init_fn"] = time.time() - t0

    t0 = time.time()
    runner_state = init_fn(rng, init_meta_params, init_skill_params)
    timings["init_fn_exec"] = time.time() - t0

    meta_state, skill_states, collect_carry, rng_carry = _unpack_runner_state(runner_state)

    t0 = time.time()
    collect_jit.lower(meta_state, skill_states, collect_carry, rng_carry).compile()
    timings["collect_jit"] = time.time() - t0

    t0 = time.time()
    new_collect, transitions, persist_carry, car_hits_sum = collect_jit(
        meta_state, skill_states, collect_carry, rng_carry
    )
    timings["collect_jit_exec"] = time.time() - t0

    t0 = time.time()
    meta_jit.lower(meta_state, transitions, rng_carry, persist_carry).compile()
    timings["meta_jit"] = time.time() - t0

    t0 = time.time()
    meta_state, rng_carry, meta_loss, meta_qvals, skip_meta = meta_jit(
        meta_state, transitions, rng_carry, persist_carry
    )
    timings["meta_jit_exec"] = time.time() - t0

    meta_n_updates = meta_state.n_updates

    t0 = time.time()
    skills_jit.lower(skill_states, transitions, meta_n_updates, rng_carry).compile()
    timings["skills_jit"] = time.time() - t0

    t0 = time.time()
    skill_states, skill_metrics, skills_frozen = skills_jit(
        skill_states, transitions, meta_n_updates, rng_carry
    )
    timings["skills_jit_exec"] = time.time() - t0

    metrics_args = (
        transitions,
        meta_state,
        meta_loss,
        meta_qvals,
        skip_meta,
        skill_metrics,
        skills_frozen,
        car_hits_sum,
        new_collect[2],
        new_collect[3],
        new_collect[6],
        new_collect[7],
    )
    t0 = time.time()
    metrics_jit.lower(*metrics_args).compile()
    timings["metrics_jit"] = time.time() - t0

    return timings


def _print_memory_guidance(num_envs: int) -> None:
    print("\n--- Memory / env-count guidance ---")
    print(
        "Dual-pixel tandem previously OOM'd at monolith compile time above ~128 envs "
        "even when runtime fit. Split regions compile independently; target 256 or 512 "
        "envs on ~48GB GPUs."
    )
    if num_envs >= 512:
        print(
            "512 envs: highest throughput if compile succeeds. Watch nvidia-smi during "
            "skills_jit compile (often the largest region)."
        )
    elif num_envs >= 256:
        print(
            "256 envs: good default after split. If skills_jit OOMs, try 256 before "
            "512 or enable Phase 4 per-skill R3."
        )
    else:
        print(
            f"{num_envs} envs: sanity-check tier only; re-run with --num-envs 256 or "
            "512 on the target GPU."
        )
    print("\nSuggested commands:")
    print("  python scripts/compile_tandem_regions_smoke.py --num-envs 256")
    print("  python scripts/compile_tandem_regions_smoke.py --num-envs 512")
    print(
        "  python scripts/compile_tandem_regions_smoke.py --num-envs 256 --run-one-update"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile tandem JIT split regions (dual-pixel ablation smoke)."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_STEM,
        help=(
            "Experiment YAML stem or path "
            f"(default: {DEFAULT_CONFIG_STEM} under input_ablation/)"
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Parallel envs for compile smoke (default: 256)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Rollout length per env (default: keep YAML value, else 128)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile regions; do not run training (default unless --run-one-update)",
    )
    parser.add_argument(
        "--run-one-update",
        action="store_true",
        help="After compile, run one Python update via run_training",
    )
    args = parser.parse_args()

    run_training_after = args.run_one_update and not args.compile_only

    config_path = _resolve_config_path(args.config)
    raw = _load_yaml(config_path)
    config = _flatten_experiment_cfg(raw)

    num_steps = int(args.num_steps if args.num_steps is not None else config.get("NUM_STEPS", 128))
    overrides = _smoke_overrides(
        num_envs=args.num_envs,
        num_steps=num_steps,
        run_one_update=run_training_after,
    )
    config.update(overrides)

    num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
    print("=== Tandem split-region compile smoke ===")
    print(f"Config: {config_path}")
    print(f"NUM_ENVS={config['NUM_ENVS']}  NUM_STEPS={config['NUM_STEPS']}  "
          f"NUM_UPDATES={num_updates}  TOTAL_TIMESTEPS={config['TOTAL_TIMESTEPS']}")
    print(f"JAX devices: {jax.devices()}")
    if not _has_accelerator():
        print("WARNING: no GPU/TPU detected; compile may be slow or differ from target machine.")

    t_build = time.time()
    bundle, _renderer = outer_make_tandem_train(config)
    print(f"Built tandem region bundle in {time.time() - t_build:.2f}s")

    missing = [k for k in SPLIT_REGION_KEYS if k not in bundle]
    if missing:
        print(f"ERROR: bundle missing keys: {missing}", file=sys.stderr)
        return 1

    rng = jax.random.split(jax.random.PRNGKey(int(config["SEED"])), config["NUM_SEEDS"])

    try:
        t_compile = time.time()
        timings = _compile_regions_timed(bundle, rng)
        total_compile = time.time() - t_compile

        print("\n--- Compile results (all regions OK) ---")
        region_names = ("init_fn", "collect_jit", "meta_jit", "skills_jit", "metrics_jit")
        for name in region_names:
            compile_s = timings.get(name, 0.0)
            exec_s = timings.get(f"{name}_exec")
            if exec_s is not None:
                print(f"  {name}: compile {compile_s:.2f}s, first exec {exec_s:.2f}s")
            else:
                print(f"  {name}: compile {compile_s:.2f}s")
        print(f"  total: {total_compile:.2f}s")

        # Also exercise bundle.compile_regions for API parity with main.py.
        bundle["compile_regions"](rng, None, None)
        print("bundle['compile_regions'] OK (main.py entry point)")

    except Exception as exc:
        print(f"\nCOMPILE FAILED: {exc}", file=sys.stderr)
        _print_memory_guidance(args.num_envs)
        return 1

    if run_training_after:
        run_training = bundle.get("run_training")
        if run_training is None:
            print("ERROR: bundle missing run_training", file=sys.stderr)
            return 1
        print("\n--- Running one update via run_training ---")
        t_run = time.time()
        out = jax.block_until_ready(run_training(rng, None, None))
        elapsed = time.time() - t_run
        meta_state, *_ = out["runner_state"]
        n_updates = int(jax.device_get(jax.tree_util.tree_map(lambda x: x[0], meta_state).n_updates))
        print(f"One update completed in {elapsed:.2f}s (meta n_updates={n_updates})")

    _print_memory_guidance(args.num_envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

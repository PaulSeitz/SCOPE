"""
Seaquest robustness evaluation across environment mods.

Evaluates three agents trained on the base environment:
  1. Hierarchical tandem (skills + meta)
  2. Flat PQN object-centric baseline
  3. Flat PQN pixel baseline

Usage (from project root):
  python scripts/run_robustness_eval.py \\
    --hierarchical-skills ./outputs/models/seaquest_pqn_comparison/.../skills.safetensors \\
    --hierarchical-meta ./outputs/models/seaquest_pqn_comparison/.../meta.safetensors \\
    --flat-oc ./path/to/pqn_seaquest_oc_seed0_vmap0.safetensors \\
    --flat-pixel ./path/to/pqn_seaquest_pixel_seed0_vmap0.safetensors

  python scripts/run_robustness_eval.py --trial   # 1 seed × 2 episodes, subset of mods
  python scripts/run_robustness_eval.py --list-mods
  python scripts/run_robustness_eval.py --validate-mods   # reset+step every mod (preflight)

RTPT is initialized once with the total episode count (agents × mods × seeds × episodes)
so nvidia-smi shows a single ETA for the full run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from concept_based_learning.baselines.flat_pqn import (  # noqa: E402
    load_flat_pqn_training_config,
    make_flat_pqn_eval_env,
    run_flat_pqn_greedy_eval,
)
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    align_config_with_skill_checkpoint,
    create_eval_env,
    load_base_config,
    load_config_for_checkpoint,
    load_params_from_file,
    resolve_config_path,
    run_greedy_eval_episodes,
)
from concept_based_learning.training.rtpt_progress import (  # noqa: E402
    rtpt_host_step,
    start_rtpt_with_total_units,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "models" / "robustness_eval" / "robustness_scores.json"
DEFAULT_SEED = 44


@dataclass(frozen=True)
class ModSpec:
    key: str
    mods: tuple[str, ...]
    category: str


ROBUSTNESS_MODS: tuple[ModSpec, ...] = (
    ModSpec("base", (), "control"),
    ModSpec("disable_enemies", ("disable_enemies",), "semantic"),
    ModSpec("no_enemies", ("no_enemies",), "semantic"),
    ModSpec("no_divers", ("no_divers",), "semantic"),
    ModSpec("gravity", ("gravity",), "dynamics"),
    ModSpec("peaceful_enemies", ("peaceful_enemies",), "semantic"),
    ModSpec("no_enemy_torpedoes", ("no_enemy_torpedoes",), "semantic"),
    ModSpec("unlimited_oxygen", ("unlimited_oxygen",), "semantic"),
    ModSpec("faster_enemies", ("faster_enemies",), "dynamics"),
    ModSpec("slower_enemies", ("slower_enemies",), "dynamics"),
    ModSpec("shift_lanes", ("shift_lanes",), "dynamics"),
    ModSpec("only_submarines", ("only_submarines",), "semantic"),
    ModSpec("only_sharks", ("only_sharks",), "semantic"),
    ModSpec("vertical_oscillation", ("vertical_oscillation",), "dynamics"),
    ModSpec("dense_spawns", ("dense_spawns",), "dynamics"),
    ModSpec("fast_oxygen_drain", ("fast_oxygen_drain",), "dynamics"),
    ModSpec("slow_oxygen_drain", ("slow_oxygen_drain",), "dynamics"),
    ModSpec("surface_sub_always", ("surface_sub_always",), "semantic"),
    ModSpec("inverted_lanes", ("inverted_lanes",), "dynamics"),
    ModSpec("lethal_divers", ("lethal_divers",), "semantic"),
    ModSpec("gravity_faster_enemies", ("gravity", "faster_enemies"), "dynamics"),
    ModSpec("peaceful_no_divers", ("peaceful_enemies", "no_divers"), "semantic"),
)

TRIAL_MOD_KEYS = (
    "base",
    "disable_enemies",
    "no_divers",
    "gravity",
    "peaceful_enemies",
    "faster_enemies",
)


def _prepare_hierarchical_config(
    *,
    experiment: str,
    skills_path: str,
    meta_path: str | None,
) -> tuple[dict[str, Any], Any, Any | None]:
    base_config = load_base_config(experiment=experiment)
    base_config, _ = load_config_for_checkpoint(skills_path, base_config)
    params_tree = load_params_from_file(skills_path)
    base_config = align_config_with_skill_checkpoint(base_config, params_tree)

    meta_params = None
    if meta_path is not None:
        meta_resolved = resolve_config_path(meta_path)
        if Path(meta_resolved).is_file():
            base_config, _ = load_config_for_checkpoint(meta_path, base_config)
            meta_params = load_params_from_file(meta_path)
    return base_config, params_tree, meta_params


def _eval_mod_for_agent(
    *,
    agent_key: str,
    mod: ModSpec,
    hierarchical_cfg: dict[str, Any] | None,
    hierarchical_params,
    hierarchical_meta,
    flat_oc_path: str | None,
    flat_pixel_path: str | None,
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
    on_episode_done=None,
) -> dict[str, Any]:
    mods_list = list(mod.mods)
    print(f"\n--- {agent_key} / {mod.key} (mods={mods_list or 'none'}) ---")

    if agent_key == "hierarchical":
        if hierarchical_cfg is None or hierarchical_params is None:
            raise ValueError("Hierarchical agent requires --hierarchical-skills")
        eval_cfg = copy.deepcopy(hierarchical_cfg)
        eval_cfg["GAME_MODS"] = mods_list if mods_list else None
        eval_cfg.pop("EVAL_GAME_MODS", None)
        return run_greedy_eval_episodes(
            eval_cfg,
            params_tree=hierarchical_params,
            meta_params=hierarchical_meta,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            progress=True,
            on_episode_done=on_episode_done,
        )

    if agent_key == "flat_oc":
        if flat_oc_path is None:
            raise ValueError("Flat OC agent requires --flat-oc")
        return run_flat_pqn_greedy_eval(
            model_path=flat_oc_path,
            game_mods=mods_list,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            object_centric=True,
            progress=True,
            on_episode_done=on_episode_done,
        )

    if agent_key == "flat_pixel":
        if flat_pixel_path is None:
            raise ValueError("Flat pixel agent requires --flat-pixel")
        return run_flat_pqn_greedy_eval(
            model_path=flat_pixel_path,
            game_mods=mods_list,
            num_episodes=num_episodes,
            num_seeds=num_seeds,
            base_seed=base_seed,
            max_frames=max_frames,
            object_centric=False,
            progress=True,
            on_episode_done=on_episode_done,
        )

    raise ValueError(f"Unknown agent key: {agent_key}")


def _smoke_reset_and_step(env, seed: int) -> None:
    """One reset + one NOOP step; forces JAX tracing of the full eval wrapper stack."""
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    jax.block_until_ready(obs)
    step_out = env.step(state, jnp.int32(0))
    if len(step_out) not in (5, 6):
        raise ValueError(f"Unexpected step() return length: {len(step_out)}")
    jax.block_until_ready(step_out[0])


def validate_robustness_mods(
    *,
    experiment: str,
    mod_specs: tuple[ModSpec, ...],
    flat_oc: str | None = None,
    flat_pixel: str | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Preflight: reset + one step for every mod on each eval env stack.

    Catches dtype / tracing bugs before a multi-hour robustness run.
    """
    stacks: list[tuple[str, Callable[[ModSpec], Any]]] = []

    base_config = load_base_config(experiment=experiment)

    def _hierarchical_env(mod: ModSpec):
        eval_cfg = copy.deepcopy(base_config)
        eval_cfg["GAME_MODS"] = list(mod.mods) if mod.mods else None
        eval_cfg.pop("EVAL_GAME_MODS", None)
        env, _renderer = create_eval_env(eval_cfg)
        return env

    stacks.append(("hierarchical", _hierarchical_env))

    if flat_oc is not None:
        oc_config = load_flat_pqn_training_config(flat_oc)

        def _flat_oc_env(mod: ModSpec):
            return make_flat_pqn_eval_env(
                object_centric=True,
                game_mods=list(mod.mods),
                config=oc_config,
            )

        stacks.append(("flat_oc", _flat_oc_env))

    if flat_pixel is not None:
        pixel_config = load_flat_pqn_training_config(flat_pixel)

        def _flat_pixel_env(mod: ModSpec):
            return make_flat_pqn_eval_env(
                object_centric=False,
                game_mods=list(mod.mods),
                config=pixel_config,
            )

        stacks.append(("flat_pixel", _flat_pixel_env))

    passed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    print(f"Validating {len(mod_specs)} mod(s) × {len(stacks)} stack(s)...\n")
    for mod in mod_specs:
        mods_s = ", ".join(mod.mods) if mod.mods else "(base)"
        for stack_name, make_env in stacks:
            label = f"{mod.key} / {stack_name}"
            print(f"  {label} (mods={mods_s})...", flush=True)
            try:
                env = make_env(mod)
                _smoke_reset_and_step(env, seed)
                passed.append({"mod": mod.key, "stack": stack_name})
                print("    ok", flush=True)
            except Exception as exc:
                failures.append(
                    {
                        "mod": mod.key,
                        "stack": stack_name,
                        "mods": mods_s,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"    FAIL: {exc}", flush=True)

    payload = {
        "experiment": experiment,
        "mod_suite": [
            {"key": m.key, "mods": list(m.mods), "category": m.category}
            for m in mod_specs
        ],
        "stacks": [name for name, _ in stacks],
        "passed": passed,
        "failures": failures,
        "all_ok": len(failures) == 0,
    }
    print()
    if failures:
        print(f"FAILED: {len(failures)} / {len(mod_specs) * len(stacks)} checks")
        for item in failures:
            print(f"  - {item['mod']} [{item['stack']}]: {item['error']}")
    else:
        print(f"All {len(passed)} checks passed.")
    return payload


def _attach_drop_from_base(results: dict[str, dict[str, Any]]) -> None:
    for agent_key, per_mod in results.items():
        base_mean = per_mod.get("base", {}).get("mean")
        if base_mean is None or not np.isfinite(base_mean) or base_mean == 0:
            continue
        for mod_key, stats in per_mod.items():
            mean = stats.get("mean")
            if mean is None or not np.isfinite(mean):
                stats["drop_pct"] = None
            else:
                stats["drop_pct"] = float((base_mean - mean) / abs(base_mean) * 100.0)


def run_robustness_eval(
    *,
    hierarchical_skills: str | None,
    hierarchical_meta: str | None,
    flat_oc: str | None,
    flat_pixel: str | None,
    experiment: str,
    mod_specs: tuple[ModSpec, ...],
    agent_keys: tuple[str, ...],
    num_episodes: int,
    num_seeds: int,
    base_seed: int,
    max_frames: int,
    output_path: Path,
    rtpt_enabled: bool = True,
    rtpt_name_initials: str = "PS",
    rtpt_experiment_name: str = "HRL_robustness_eval",
    rtpt_step_every: int = 1,
) -> dict[str, Any]:
    total_episodes = len(agent_keys) * len(mod_specs) * num_seeds * num_episodes
    on_episode_done = None
    if rtpt_enabled:
        started = start_rtpt_with_total_units(
            total_episodes,
            {
                "RTPT_ENABLED": True,
                "RTPT_NAME_INITIALS": rtpt_name_initials,
                "RTPT_EXPERIMENT_NAME": rtpt_experiment_name,
                "RTPT_STEP_EVERY": max(1, rtpt_step_every),
            },
        )
        if started:
            rtpt_steps = (total_episodes + max(1, rtpt_step_every) - 1) // max(
                1, rtpt_step_every
            )
            print(
                f"RTPT initialized: {total_episodes:,} eval episodes "
                f"({rtpt_steps:,} RTPT steps, step_every={rtpt_step_every}).\n"
            )
            on_episode_done = rtpt_host_step
        else:
            print("RTPT not started (disabled or rtpt package unavailable).\n")

    hierarchical_cfg = None
    hierarchical_params = None
    hierarchical_meta_params = None
    if hierarchical_skills is not None:
        hierarchical_cfg, hierarchical_params, hierarchical_meta_params = (
            _prepare_hierarchical_config(
                experiment=experiment,
                skills_path=hierarchical_skills,
                meta_path=hierarchical_meta,
            )
        )

    results: dict[str, dict[str, Any]] = {agent: {} for agent in agent_keys}
    for mod in mod_specs:
        for agent_key in agent_keys:
            stats = _eval_mod_for_agent(
                agent_key=agent_key,
                mod=mod,
                hierarchical_cfg=hierarchical_cfg,
                hierarchical_params=hierarchical_params,
                hierarchical_meta=hierarchical_meta_params,
                flat_oc_path=flat_oc,
                flat_pixel_path=flat_pixel,
                num_episodes=num_episodes,
                num_seeds=num_seeds,
                base_seed=base_seed,
                max_frames=max_frames,
                on_episode_done=on_episode_done,
            )
            stats["category"] = mod.category
            stats["mods"] = list(mod.mods)
            results[agent_key][mod.key] = stats

    _attach_drop_from_base(results)

    payload = {
        "experiment": experiment,
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
        "base_seed": base_seed,
        "max_frames_per_episode": max_frames,
        "agents": {
            "hierarchical": {
                "skills": resolve_config_path(hierarchical_skills) if hierarchical_skills else None,
                "meta": resolve_config_path(hierarchical_meta) if hierarchical_meta else None,
            },
            "flat_oc": resolve_config_path(flat_oc) if flat_oc else None,
            "flat_pixel": resolve_config_path(flat_pixel) if flat_pixel else None,
        },
        "mod_suite": [
            {"key": m.key, "mods": list(m.mods), "category": m.category}
            for m in mod_specs
        ],
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote robustness results to {output_path}")
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("Robustness summary (mean score; drop% vs base)")
    print("=" * 72)
    mod_keys = [m["key"] for m in payload["mod_suite"]]
    for agent_key, per_mod in payload["results"].items():
        print(f"\n[{agent_key}]")
        for mod_key in mod_keys:
            stats = per_mod.get(mod_key, {})
            mean = stats.get("mean")
            drop = stats.get("drop_pct")
            if mean is None:
                continue
            drop_s = f"{drop:+.1f}%" if drop is not None else "n/a"
            print(f"  {mod_key:28s}  mean={mean:8.2f}  drop={drop_s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seaquest robustness eval across env mods.")
    parser.add_argument(
        "--hierarchical-skills",
        type=str,
        default=None,
        help="Path to hierarchical skill checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--hierarchical-meta",
        type=str,
        default=None,
        help="Path to hierarchical meta checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--flat-oc",
        type=str,
        default=None,
        help="Path to flat object-centric PQN checkpoint.",
    )
    parser.add_argument(
        "--flat-pixel",
        type=str,
        default=None,
        help="Path to flat pixel PQN checkpoint.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="seaquest_pqn_comparison",
        help="Experiment yaml for hierarchical config defaults.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="JSON output path.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Greedy episodes per seed (default: 20).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of eval RNG seeds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base eval seed (default: 44).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=2000,
        help="Max env steps per episode (default: 2000).",
    )
    parser.add_argument(
        "--mods",
        nargs="+",
        default=None,
        help="Optional subset of mod keys to run (default: full suite).",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["hierarchical", "flat_oc", "flat_pixel"],
        default=["hierarchical", "flat_oc", "flat_pixel"],
        help="Which agents to evaluate.",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Quick smoke run: 1 seed, 2 episodes, reduced mod set.",
    )
    parser.add_argument(
        "--list-mods",
        action="store_true",
        help="Print mod suite and exit.",
    )
    parser.add_argument(
        "--validate-mods",
        action="store_true",
        help="Preflight: reset+one step for every mod on each eval env stack; exit 1 on failure.",
    )
    parser.add_argument(
        "--no-rtpt",
        action="store_true",
        help="Disable RTPT cluster progress (nvidia-smi ETA).",
    )
    parser.add_argument(
        "--rtpt-step-every",
        type=int,
        default=1,
        help="Call RTPT.step() every N completed eval episodes (default: 1).",
    )
    parser.add_argument(
        "--rtpt-name",
        type=str,
        default="HRL_robustness_eval",
        help="RTPT experiment name shown on the cluster (default: HRL_robustness_eval).",
    )
    parser.add_argument(
        "--rtpt-initials",
        type=str,
        default="PS",
        help="RTPT name initials (default: PS).",
    )
    args = parser.parse_args()

    if args.list_mods:
        print("Robustness mod suite:")
        for mod in ROBUSTNESS_MODS:
            mods_s = ", ".join(mod.mods) if mod.mods else "(base env)"
            print(f"  {mod.key:28s} [{mod.category:8s}]  {mods_s}")
        return

    mod_specs = ROBUSTNESS_MODS
    num_episodes = args.num_episodes
    num_seeds = args.num_seeds
    if args.trial:
        mod_specs = tuple(m for m in ROBUSTNESS_MODS if m.key in TRIAL_MOD_KEYS)
        num_episodes = min(num_episodes, 2)
        num_seeds = min(num_seeds, 1)

    if args.mods:
        wanted = set(args.mods)
        mod_specs = tuple(m for m in ROBUSTNESS_MODS if m.key in wanted)
        missing = wanted - {m.key for m in mod_specs}
        if missing:
            raise ValueError(f"Unknown mod keys: {sorted(missing)}")

    if args.validate_mods:
        payload = validate_robustness_mods(
            experiment=args.experiment,
            mod_specs=mod_specs,
            flat_oc=args.flat_oc,
            flat_pixel=args.flat_pixel,
            seed=args.seed,
        )
        if not payload["all_ok"]:
            sys.exit(1)
        return

    agent_keys = tuple(args.agents)
    if not any(
        [
            args.hierarchical_skills and "hierarchical" in agent_keys,
            args.flat_oc and "flat_oc" in agent_keys,
            args.flat_pixel and "flat_pixel" in agent_keys,
        ]
    ):
        parser.error(
            "Provide checkpoints for each selected agent "
            "(--hierarchical-skills, --flat-oc, --flat-pixel)."
        )

    payload = run_robustness_eval(
        hierarchical_skills=args.hierarchical_skills,
        hierarchical_meta=args.hierarchical_meta,
        flat_oc=args.flat_oc,
        flat_pixel=args.flat_pixel,
        experiment=args.experiment,
        mod_specs=mod_specs,
        agent_keys=agent_keys,
        num_episodes=num_episodes,
        num_seeds=num_seeds,
        base_seed=args.seed,
        max_frames=args.max_frames,
        output_path=Path(args.output),
        rtpt_enabled=not args.no_rtpt,
        rtpt_name_initials=args.rtpt_initials,
        rtpt_experiment_name=args.rtpt_name,
        rtpt_step_every=args.rtpt_step_every,
    )
    _print_summary(payload)


if __name__ == "__main__":
    main()

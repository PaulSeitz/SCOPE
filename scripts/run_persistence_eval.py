"""
Persistence + skill-reward-blend evaluation runner.

Trains shared Seaquest skills, then runs a matrix of tandem conditions:
  - No persistence + SKILL_REWARD_LAMBDA in {0.5, 1.0}
  - Each persistence mode + SKILL_REWARD_LAMBDA 0.0 (concept-only skills)
  - Each persistence mode + SKILL_REWARD_LAMBDA 1.0 (env reward bleeds into skills)

Each condition trains Seaquest tandem, then transfers skills to Phoenix tandem
(meta trains from scratch on each env; Seaquest meta is not portable to Phoenix).

Phases:
  1 — Seaquest skill pretrain (scratch, shared)
  2 — Variant matrix (Seaquest tandem → Phoenix tandem per variant)

Usage (from project root):
  python scripts/run_persistence_eval.py
  python scripts/run_persistence_eval.py --variants no_persist_srl0.5 until_done_srl1.0
  python scripts/run_persistence_eval.py --phases 1 --trial
  # --trial defaults to WANDB_MODE=disabled (override with --wandb-mode online)
  python scripts/run_persistence_eval.py --skip-eval
  python scripts/run_persistence_eval.py --list-matrix   # print wandb names + tags
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import main as train_main  # noqa: E402
from concept_based_learning.pqn_visualizer import (  # noqa: E402
    load_params_from_file,
    run_greedy_eval_episodes,
)
from concept_based_learning.training.rtpt_progress import (  # noqa: E402
    num_updates_from_timesteps,
    start_rtpt,
)

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "persistence_eval"
SEED = 44
WANDB_ENTITY = "paul-seitz-tu-darmstadt"
TRIAL_STEPS_DEFAULT = 32768
PERSISTENCE_MODES = ("until_done", "mission_avoid", "fixed_steps")

EVAL_SCORES_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "persistence_eval" / "eval_scores.json"
)


@dataclass(frozen=True)
class EvalVariant:
    """One tandem training condition in the phase-2 matrix."""

    key: str
    persist_mode: str | None
    skill_reward_lambda: float

    @property
    def save_dir(self) -> str:
        return f"./outputs/models/persistence_eval/{self.key}"


def _default_variants() -> list[EvalVariant]:
    variants: list[EvalVariant] = [
        EvalVariant("no_persist_srl0.5", None, 0.5),
        EvalVariant("no_persist_srl1.0", None, 1.0),
    ]
    for mode in PERSISTENCE_MODES:
        variants.append(EvalVariant(mode, mode, 0.0))
        variants.append(EvalVariant(f"{mode}_srl1.0", mode, 1.0))
    return variants


DEFAULT_VARIANTS = _default_variants()
DEFAULT_VARIANT_KEYS = [v.key for v in DEFAULT_VARIANTS]

# wandb tags appended after main.py defaults: HIERARCHICAL_PQN, ENV_NAME, jax_*.
WANDB_SUITE_TAG = "persistence_eval"


def _srl_tag(skill_reward_lambda: float) -> str:
    if skill_reward_lambda == 0.0:
        return "srl:0"
    if skill_reward_lambda == 0.5:
        return "srl:0.5"
    if skill_reward_lambda == 1.0:
        return "srl:1"
    return f"srl:{skill_reward_lambda}"


def _persist_tag(persist_mode: str | None) -> str:
    return f"persist:{persist_mode}" if persist_mode else "persist:none"


def wandb_tags_phase1(*, trial: bool = False) -> list[str]:
    tags = [
        WANDB_SUITE_TAG,
        "phase:1",
        "env:seaquest",
        "train_mode:skills_only",
        "persist:none",
        "srl:0",
        "variant:phase1_skills",
    ]
    if trial:
        tags.append("trial")
    return tags


def wandb_tags_phase2(
    variant: EvalVariant,
    env: str,
    *,
    transfer: bool = False,
    meta_init: str = "scratch",
    trial: bool = False,
) -> list[str]:
    tags = [
        WANDB_SUITE_TAG,
        "phase:2",
        f"env:{env}",
        "train_mode:tandem",
        _persist_tag(variant.persist_mode),
        _srl_tag(variant.skill_reward_lambda),
        f"variant:{variant.key}",
        "skills_init:seaquest",
        f"meta_init:{meta_init}",
    ]
    if transfer:
        tags.append("transfer")
    if trial:
        tags.append("trial")
    return tags


def _wandb_auto_tags(env_name: str) -> list[str]:
    """Tags main.single_run adds automatically (for matrix documentation)."""
    return ["HIERARCHICAL_PQN", env_name.upper()]


def print_run_matrix(
    *,
    phases: list[int],
    variants: list[EvalVariant],
    trial: bool = False,
) -> None:
    """Print wandb run names and tags for verification before launching."""
    rows: list[tuple[str, str, str, str]] = []

    if 1 in phases:
        custom = wandb_tags_phase1(trial=trial)
        auto = _wandb_auto_tags("seaquest")
        rows.append(
            (
                "1",
                "seaquest",
                "persistence_eval_p1_seaquest_skills",
                ", ".join(auto + custom),
            )
        )

    if 2 in phases:
        for variant in variants:
            for env, transfer in (("seaquest", False), ("phoenix", True)):
                custom = wandb_tags_phase2(
                    variant, env, transfer=transfer, meta_init="scratch", trial=trial
                )
                auto = _wandb_auto_tags(env)
                run_name = f"persistence_eval_p2_{env}_{variant.key}"
                rows.append(
                    (
                        "2",
                        env,
                        run_name,
                        ", ".join(auto + custom),
                    )
                )

    print("Persistence eval — full run matrix")
    print("=" * 100)
    print(
        f"{'Phase':<6} {'Env':<10} {'WANDB_RUN_NAME':<45} "
        f"Tags (auto + custom)"
    )
    print("-" * 100)
    for phase, env, run_name, tags in rows:
        print(f"{phase:<6} {env:<10} {run_name:<45} {tags}")
    print("-" * 100)
    print(f"Total training runs: {len(rows)}")
    print()
    print("Filter examples in wandb:")
    print("  Suite:        tag:persistence_eval")
    print("  Phase:        tag:phase:1  |  tag:phase:2")
    print("  Env:          tag:env:seaquest  |  tag:env:phoenix")
    print("  Persistence:  tag:persist:none  |  tag:persist:until_done  | ...")
    print("  Skill blend:  tag:srl:0  |  tag:srl:0.5  |  tag:srl:1")
    print("  Transfer:     tag:transfer")
    print("  Variant:      tag:variant:until_done_srl1.0")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    alg = raw.pop("alg", {})
    raw.update(alg)
    raw["SEED"] = SEED
    return raw


def _skills_ckpt(save_path: str, env: str, seed: int = SEED) -> Path:
    return Path(save_path) / env / f"hierarchical_pqn_{env}_seed{seed}.safetensors"


def _tandem_skills_ckpt(save_path: str, env: str, seed: int = SEED) -> Path:
    return (
        Path(save_path)
        / env
        / f"hierarchical_pqn_{env}_seed{seed}_skills.safetensors"
    )


def _meta_ckpt(save_path: str, env: str, seed: int = SEED) -> Path:
    return (
        Path(save_path)
        / env
        / f"hierarchical_pqn_{env}_seed{seed}_meta.safetensors"
    )


def _apply_variant(cfg: dict[str, Any], variant: EvalVariant) -> dict[str, Any]:
    """Inject persistence + skill-reward-blend settings for one variant."""
    out = copy.deepcopy(cfg)
    out["SAVE_PATH"] = variant.save_dir
    out["SKILL_REWARD_LAMBDA"] = variant.skill_reward_lambda

    for key in (
        "META_TARGET_PERSIST_MODE",
        "META_TARGET_PERSIST_STEPS",
        "META_AVOID_INTERRUPT_MARGIN",
    ):
        out.pop(key, None)

    if variant.persist_mode is not None:
        out["META_TARGET_PERSIST_MODE"] = variant.persist_mode
        if variant.persist_mode == "fixed_steps":
            out["META_TARGET_PERSIST_STEPS"] = int(
                out.get("META_TARGET_PERSIST_STEPS", 300)
            )
        if variant.persist_mode == "mission_avoid":
            out["META_AVOID_INTERRUPT_MARGIN"] = float(
                out.get("META_AVOID_INTERRUPT_MARGIN", 0.0)
            )
    return out


def _eval_config_from_training(cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = dict(cfg)
    for key in (
        "TOTAL_TIMESTEPS",
        "INIT_MODEL_PATH",
        "INIT_META_MODEL_PATH",
        "WANDB_MODE",
        "WANDB_TAGS",
        "WANDB_RUN_NAME",
        "RECORD_VIDEO",
        "VIDEO_AT_END",
    ):
        eval_cfg.pop(key, None)
    eval_cfg["WANDB_MODE"] = "disabled"
    return eval_cfg


def _load_eval_scores() -> dict[str, Any]:
    if not EVAL_SCORES_PATH.exists():
        return {}
    with open(EVAL_SCORES_PATH) as f:
        return json.load(f)


def _save_eval_scores(scores: dict[str, Any]) -> None:
    EVAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)


def _run_inter_phase_eval(
    *,
    label: str,
    cfg: dict[str, Any],
    skills_path: Path,
    meta_path: Path | None,
    num_episodes: int,
    num_seeds: int,
) -> dict[str, Any]:
    if not skills_path.exists():
        raise FileNotFoundError(f"Eval skills checkpoint not found: {skills_path}")

    eval_cfg = _eval_config_from_training(cfg)
    skill_params = load_params_from_file(str(skills_path))
    meta_params = (
        load_params_from_file(str(meta_path))
        if meta_path is not None and meta_path.exists()
        else None
    )
    meta_label = "learned_meta" if meta_params is not None else "heuristic_meta"
    print(
        f"\n  Greedy eval — {label} ({meta_label}, "
        f"{num_seeds} seeds × {num_episodes} episodes)..."
    )
    stats = run_greedy_eval_episodes(
        eval_cfg,
        params_tree=skill_params,
        meta_params=meta_params,
        num_episodes=num_episodes,
        num_seeds=num_seeds,
        base_seed=int(eval_cfg.get("EVAL_SEED", SEED + 1000)),
    )
    print(
        f"  Eval score: mean={stats['mean']:.2f} ± {stats['std']:.2f} "
        f"(n={stats['num_episodes']})"
    )
    return stats


def _compute_total_updates(
    phases: list[int],
    variants: list[EvalVariant],
    trial_steps: int | None,
) -> int:
    total = 0
    p1_cfg = _load_yaml(CONFIG_DIR / "seaquest_phase1_skills.yaml")
    if 1 in phases:
        ts = trial_steps or int(p1_cfg.get("TOTAL_TIMESTEPS", 0))
        total += num_updates_from_timesteps(
            ts,
            num_steps=int(p1_cfg.get("NUM_STEPS", 32)),
            num_envs=int(p1_cfg.get("NUM_ENVS", 512)),
        )
    if 2 in phases:
        sq_base = _load_yaml(CONFIG_DIR / "seaquest_tandem_base.yaml")
        for _variant in variants:
            for env in ("seaquest", "phoenix"):
                cfg = (
                    sq_base
                    if env == "seaquest"
                    else _load_yaml(CONFIG_DIR / "phoenix_tandem_base.yaml")
                )
                ts = trial_steps or int(cfg.get("TOTAL_TIMESTEPS", 0))
                total += num_updates_from_timesteps(
                    ts,
                    num_steps=int(cfg.get("NUM_STEPS", 64)),
                    num_envs=int(cfg.get("NUM_ENVS", 512)),
                )
    return max(1, total)


def _run(cfg: dict[str, Any], label: str, abort_on_failure: bool) -> bool:
    print(f"\n{'='*70}")
    print(f"  START: {label}")
    print(
        f"  ENV={cfg.get('ENV_NAME')}  MODE={cfg.get('TRAIN_MODE')}  "
        f"PERSIST={cfg.get('META_TARGET_PERSIST_MODE', 'none')}  "
        f"SKILL_REWARD_LAMBDA={cfg.get('SKILL_REWARD_LAMBDA', 0.0)}  "
        f"STEPS={cfg.get('TOTAL_TIMESTEPS', 0):,}"
    )
    print(f"{'='*70}\n")
    try:
        train_main.single_run(cfg)
        print(f"\n  DONE: {label}\n")
        return True
    except Exception:
        print(f"\n{'!'*70}")
        print(f"  FAILED: {label}")
        traceback.print_exc()
        print(f"{'!'*70}\n")
        if abort_on_failure:
            raise
        return False


def run_persistence_eval(
    phases: list[int],
    variants: list[EvalVariant],
    abort_on_failure: bool,
    trial_steps: int | None,
    wandb_mode: str,
    skip_eval: bool,
    eval_episodes: int,
    eval_seeds: int,
) -> None:
    results: list[tuple[str, bool]] = []
    eval_scores = _load_eval_scores()

    total_updates = _compute_total_updates(phases, variants, trial_steps)
    start_rtpt({
        "RTPT_ENABLED": True,
        "RTPT_NAME_INITIALS": "PS",
        "RTPT_EXPERIMENT_NAME": "persistence",
        "RTPT_STEP_EVERY": 10,
        "TOTAL_TIMESTEPS": total_updates,
        "NUM_STEPS": 1,
        "NUM_ENVS": 1,
    })
    print(f"RTPT initialized: {total_updates:,} total updates.\n")
    print(
        "Phase-2 variants: "
        + ", ".join(
            f"{v.key}(persist={v.persist_mode or 'none'}, srl={v.skill_reward_lambda})"
            for v in variants
        )
    )

    p1_cfg = _load_yaml(CONFIG_DIR / "seaquest_phase1_skills.yaml")
    phase1_save = str(p1_cfg["SAVE_PATH"])
    sq_base = _load_yaml(CONFIG_DIR / "seaquest_tandem_base.yaml")
    px_base = _load_yaml(CONFIG_DIR / "phoenix_tandem_base.yaml")

    if trial_steps:
        eval_episodes = min(eval_episodes, 2)
        eval_seeds = min(eval_seeds, 1)

    def _apply_common(cfg: dict, run_name: str, tags: list[str]) -> dict:
        if trial_steps:
            cfg["TOTAL_TIMESTEPS"] = trial_steps
        cfg["WANDB_MODE"] = wandb_mode
        cfg["ENTITY"] = WANDB_ENTITY
        cfg["PROJECT"] = "concept_based_learning"
        cfg["WANDB_TAGS"] = tags
        cfg["WANDB_RUN_NAME"] = run_name
        cfg["RECORD_VIDEO"] = False
        cfg["VIDEO_AT_END"] = False
        return cfg

    def _record_eval(key: str, variant: EvalVariant, env: str, stats: dict[str, Any]) -> None:
        eval_scores[key] = {
            "persist_mode": variant.persist_mode,
            "skill_reward_lambda": variant.skill_reward_lambda,
            "env": env,
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "num_episodes": stats["num_episodes"],
            "returns": stats["returns"],
        }
        _save_eval_scores(eval_scores)

    if 1 in phases:
        cfg = _apply_common(
            dict(p1_cfg),
            "persistence_eval_p1_seaquest_skills",
            wandb_tags_phase1(trial=bool(trial_steps)),
        )
        ok = _run(cfg, "Phase 1 — Seaquest skill pretrain", abort_on_failure)
        results.append(("Phase1/seaquest/skills", ok))
        if ok and not skip_eval:
            stats = _run_inter_phase_eval(
                label="Phase1/seaquest/skills",
                cfg=cfg,
                skills_path=_skills_ckpt(phase1_save, "seaquest"),
                meta_path=None,
                num_episodes=eval_episodes,
                num_seeds=eval_seeds,
            )
            _record_eval(
                "phase1/seaquest",
                EvalVariant("phase1", None, 0.0),
                "seaquest",
                stats,
            )

    phase1_skills = _skills_ckpt(phase1_save, "seaquest")
    if 2 in phases and not phase1_skills.exists():
        raise FileNotFoundError(
            f"Phase-1 Seaquest skills required for phase 2: {phase1_skills}"
        )

    if 2 in phases:
        for variant in variants:
            sq_cfg = _apply_variant(sq_base, variant)
            sq_cfg["INIT_MODEL_PATH"] = str(phase1_skills)
            sq_cfg = _apply_common(
                sq_cfg,
                f"persistence_eval_p2_seaquest_{variant.key}",
                wandb_tags_phase2(
                    variant, "seaquest", meta_init="scratch", trial=bool(trial_steps)
                ),
            )
            sq_label = (
                f"Phase 2 — Seaquest tandem [{variant.key}] "
                f"(persist={variant.persist_mode or 'none'}, "
                f"srl={variant.skill_reward_lambda})"
            )
            ok = _run(sq_cfg, sq_label, abort_on_failure)
            results.append((f"Phase2/{variant.key}/seaquest", ok))

            variant_save = variant.save_dir
            if ok and not skip_eval:
                stats = _run_inter_phase_eval(
                    label=f"Phase2/{variant.key}/seaquest",
                    cfg=sq_cfg,
                    skills_path=_tandem_skills_ckpt(variant_save, "seaquest"),
                    meta_path=_meta_ckpt(variant_save, "seaquest"),
                    num_episodes=eval_episodes,
                    num_seeds=eval_seeds,
                )
                _record_eval(f"phase2/{variant.key}/seaquest", variant, "seaquest", stats)

            if not ok:
                results.append((f"Phase2/{variant.key}/phoenix", False))
                continue

            px_cfg = _apply_variant(px_base, variant)
            px_cfg["INIT_MODEL_PATH"] = str(phase1_skills)
            px_cfg.pop("INIT_META_MODEL_PATH", None)
            px_cfg = _apply_common(
                px_cfg,
                f"persistence_eval_p2_phoenix_{variant.key}",
                wandb_tags_phase2(
                    variant,
                    "phoenix",
                    transfer=True,
                    meta_init="scratch",
                    trial=bool(trial_steps),
                ),
            )
            px_label = (
                f"Phase 2 — Phoenix tandem [{variant.key}] "
                f"(persist={variant.persist_mode or 'none'}, "
                f"srl={variant.skill_reward_lambda})"
            )
            ok_px = _run(px_cfg, px_label, abort_on_failure)
            results.append((f"Phase2/{variant.key}/phoenix", ok_px))
            if ok_px and not skip_eval:
                stats = _run_inter_phase_eval(
                    label=f"Phase2/{variant.key}/phoenix",
                    cfg=px_cfg,
                    skills_path=_tandem_skills_ckpt(variant_save, "phoenix"),
                    meta_path=_meta_ckpt(variant_save, "phoenix"),
                    num_episodes=eval_episodes,
                    num_seeds=eval_seeds,
                )
                _record_eval(f"phase2/{variant.key}/phoenix", variant, "phoenix", stats)

    print("\n" + "=" * 70)
    print("  PERSISTENCE EVAL COMPLETE")
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    print(f"  {passed}/{len(results)} runs succeeded\n")
    for label, ok in results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    if eval_scores and not skip_eval:
        print(f"\n  Greedy eval scores saved to {EVAL_SCORES_PATH}")
    print()


def _resolve_variants(variant_keys: list[str] | None) -> list[EvalVariant]:
    if not variant_keys:
        return list(DEFAULT_VARIANTS)
    by_key = {v.key: v for v in DEFAULT_VARIANTS}
    missing = [k for k in variant_keys if k not in by_key]
    if missing:
        raise ValueError(
            f"Unknown variant key(s): {missing}. "
            f"Available: {DEFAULT_VARIANT_KEYS}"
        )
    return [by_key[k] for k in variant_keys]


def _resolve_wandb_mode(*, trial: bool, wandb_mode: str | None) -> str:
    """Smoke tests skip wandb unless the user sets --wandb-mode explicitly."""
    if wandb_mode is not None:
        return wandb_mode
    return "disabled" if trial else "online"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1, 2],
        choices=[1, 2],
        help="Phase 1 = skills; Phase 2 = variant matrix (default: 1 2)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Subset of variant keys to run (default: full matrix). "
            f"Keys: {', '.join(DEFAULT_VARIANT_KEYS)}"
        ),
    )
    parser.add_argument("--abort-on-failure", action="store_true")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument(
        "--trial-steps",
        type=int,
        default=TRIAL_STEPS_DEFAULT,
        dest="trial_steps",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        dest="wandb_mode",
        help="wandb logging mode (default: disabled with --trial, online otherwise)",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="Print the full run matrix with wandb run names and tags, then exit",
    )
    parser.add_argument("--eval-episodes", type=int, default=5, dest="eval_episodes")
    parser.add_argument("--eval-seeds", type=int, default=3, dest="eval_seeds")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    variants = _resolve_variants(args.variants)
    if args.list_matrix:
        print_run_matrix(
            phases=sorted(set(args.phases)),
            variants=variants,
            trial=args.trial,
        )
        raise SystemExit(0)
    run_persistence_eval(
        phases=sorted(set(args.phases)),
        variants=variants,
        abort_on_failure=args.abort_on_failure,
        trial_steps=args.trial_steps if args.trial else None,
        wandb_mode=_resolve_wandb_mode(trial=args.trial, wandb_mode=args.wandb_mode),
        skip_eval=args.skip_eval,
        eval_episodes=args.eval_episodes,
        eval_seeds=args.eval_seeds,
    )

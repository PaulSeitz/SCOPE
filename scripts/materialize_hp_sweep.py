#!/usr/bin/env python3
"""Materialize hyperparameter-sweep suites on the noclip + sparse C/D/A spine.

Stage ``eps`` (exploration sanity) — Seaquest skills-only ~5M, constant
``SKILL_LR=5e-4`` (no LR decay), then home mod gate + Pong/Beamrider base ZS.

Cells (B = current smoke ε with decay removed)::

  A  EPS 0.1 → 0.01 / 0.2     (recent tandem-ish)
  B  EPS 0.2 → 0.01 / 0.2     (current skills baseline, constant LR)
  C  EPS 0.5 → 0.01 / 0.2
  D  EPS 1.0 → 0.001 / 0.1    (flat PQN / NEXUS Seaquest)
  E  EPS 1.0 → 0.01 / 0.2

Stage ``skills`` — later skills HP grid (scaffold; fill CELLS_SKILLS when ready).
Stage ``tandem`` — later meta-side grid (scaffold).

    python scripts/materialize_hp_sweep.py              # eps only
    python scripts/materialize_hp_sweep.py --stage all
    python scripts/run_paper_eval.py --suite eps_sanity \\
        --gpus 0,1,2,3 --wait-deps --skip-rq-gate --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE = "noclip_facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SPARSE_TAG = "sparse_cda"
# 305 × 128 × 128 — batch-aligned ~5M probe budget.
TIMESTEPS_5M = 4_997_120
BUDGET_TAG = "5m"
SKILL_LR = 0.0005  # constant; no SKILL_LR_FINISH

SUITE_EPS = "eps_sanity"
SUITE_SKILLS = "hp_sweep_skills"
SUITE_TANDEM = "hp_sweep_tandem"

# Home competence / robustness gate (plus base). User ranks the table.
MOD_KEYS: tuple[str, ...] = (
    "base",
    "shift_lanes",
    "lane_scramble",
    "dynamic_lane_drift",
    "change_background_color",
    "no_enemies",
    "gravity",
    "faster_enemies",
)

TRANSFER_ENVS: dict[str, dict] = {
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po"},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br"},
}

# tag → (EPS_START, EPS_FINISH, EPS_DECAY)
CELLS_EPS: dict[str, tuple[float, float, float]] = {
    "a_e01": (0.1, 0.01, 0.2),
    "b_e02": (0.2, 0.01, 0.2),
    "c_e05": (0.5, 0.01, 0.2),
    "d_e10_pqn": (1.0, 0.001, 0.1),
    "e_e10": (1.0, 0.01, 0.2),
}

# Later skills HP grid — same helpers / mod gate / transfers. Fill when Stage 1 runs.
# Example shape: tag → dict of alg overrides (SKILL_LR, EPS_*, …).
CELLS_SKILLS: dict[str, dict] = {}

# Later tandem meta grid (SKILL_EPS matched to winning skills cell).
CELLS_TANDEM: dict[str, dict] = {}


def _lines(*parts: str | list[str]) -> str:
    out: list[str] = []
    for p in parts:
        if isinstance(p, list):
            out.extend(p)
        elif p:
            out.extend(p.rstrip("\n").split("\n"))
    return "\n".join(out) + "\n"


def _write(path: Path, body: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = body if body.endswith("\n") else body + "\n"
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text)
    print(f"wrote {path.relative_to(_ROOT)}")


def _ego_lines() -> list[str]:
    return [
        "USE_POV_RELATIVE_MAPPER: true",
        "USE_POV_FIXED_BILATERAL: false",
        "USE_POV_FACING_FRAME: true",
        "EGO_POSE_XY: false",
        "USE_LEGACY_EGO_OBS: false",
        "USE_EGO_SIZE_FEATURES: false",
        "USE_EGO_VELOCITY_FEATURES: false",
        "USE_EGO_BLOCKING_FEATURES: false",
        "USE_EGO_WEAPON_FEATURES: false",
        "USE_EGO_SHOT_FEATURES: false",
        "AVOID_INTERRUPT_RANGE: 55",
        "AVOID_INTERRUPT_RANGE_Y: 55",
        "SURFACE_SUB_AVOID_RANGE_X: 5",
        "SURFACE_SUB_AVOID_RANGE_Y: 5",
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_ANGLE_BINS: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "EGO_LEGAL_ACTION_MASK: false",
        "SURFACE_SUB_NEAR_RANGE: 5",
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER: false",
        "USE_LEGACY_SEAQUEST_ASCENT_BLOCK: false",
        "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD: false",
    ]


def _sparse_top() -> list[str]:
    return [
        "DESTROY_CONTACT_FAIL: true",
        "DESTROY_LIVING_PENALTY: false",
        "COLLECT_SPARSE_ONLY: true",
        "AVOID_SPARSE_ONLY: true",
    ]


def _sparse_alg() -> list[str]:
    return [
        "  DESTROY_CONTACT_FAIL: true",
        "  DESTROY_LIVING_PENALTY: false",
        "  COLLECT_SPARSE_ONLY: true",
        "  AVOID_SPARSE_ONLY: true",
    ]


def _mod_keys_yaml() -> list[str]:
    return ["PAPER_EVAL_MOD_KEYS:"] + [f"- {k}" for k in MOD_KEYS]


def train_stem(suite: str, tag: str) -> str:
    return (
        f"seaquest_skills_ego_pov_{SPINE}_{SPARSE_TAG}_{suite}_{tag}_{BUDGET_TAG}"
    )


def eval_stem(suite: str, tag: str) -> str:
    return f"{train_stem(suite, tag)}_eval"


def transfer_stem(env: str, suite: str, tag: str) -> str:
    return f"{env}_transfer_ego_pov_{SPINE}_{SPARSE_TAG}_{suite}_{tag}_{BUDGET_TAG}"


def _skills_train_yaml(
    *,
    suite: str,
    tag: str,
    eps_start: float,
    eps_finish: float,
    eps_decay: float,
    skill_lr: float = SKILL_LR,
    extra_alg: dict | None = None,
) -> str:
    stem = train_stem(suite, tag)
    lines: list[str] = [
        f"# Seaquest ego skills — {suite} cell {tag} (~5M, constant SKILL_LR).",
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: skills",
        "PAPER_EVAL_MODS: seaquest",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        "ENV_NAME: seaquest",
    ]
    lines += _ego_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
        f"WANDB_RUN_NAME: sq_paper_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:1",
        "- env:seaquest",
        "- role:base",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- budget:{BUDGET_TAG}",
        "- phase:skills_only",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- lr:constant",
        f"- cell:{tag}",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    lines += _sparse_top()
    lines += [
        "alg:",
        "  NUM_SKILLS: 4",
        "  SKILL_NAMES:",
        "  - DESTROY",
        "  - COLLECT",
        "  - AVOID",
        "  - NAVIGATE",
        "  MAX_OBJECTS: 35",
        "  EGO_ACTION_DIM: 18",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  SKILL_EMBED_DIM: 256",
        f"  SKILL_LR: {skill_lr}",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {TIMESTEPS_5M}",
        "  NUM_ENVS: 128",
        "  NUM_STEPS: 128",
        "  NUM_EPOCHS: 4",
        "  NUM_MINIBATCHES: 64",
        "  GAMMA: 0.99",
        "  SKILL_GAMMA: 0.99",
        "  LAMBDA: 0.65",
        "  SKILL_LAMBDA: 0.65",
        "  LR: 0.0001",
        "  MAX_GRAD_NORM: 0.8",
        "  OFF_POLICY: true",
        "  RELABEL_ALL_OBJECTS: false",
        "  RELABEL_K_OBJECTS: null",
        "  SKILL_BOOTSTRAP: q_lambda",
        f"  EPS_START: {eps_start}",
        f"  EPS_FINISH: {eps_finish}",
        f"  EPS_DECAY: {eps_decay}",
        "  HEURISTIC_MODE: env_specific",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    if extra_alg:
        for k, v in extra_alg.items():
            if k.startswith("EPS_") or k == "SKILL_LR":
                continue  # already set
            lines.append(f"  {k}: {v}")
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
    ]
    return _lines(lines)


def _skills_eval_yaml(*, suite: str, tag: str) -> str:
    stem = train_stem(suite, tag)
    estem = eval_stem(suite, tag)
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44.safetensors"
    )
    lines: list[str] = [
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += _mod_keys_yaml()
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        "ENV_NAME: seaquest",
    ]
    lines += _ego_lines()
    lines += [
        f"SAVE_PATH: ./outputs/models/paper_eval/rq1/{stem}",
        f"WANDB_RUN_NAME: sq_paper_eval_{suite}_{tag}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:1",
        "- env:seaquest",
        "- phase:mod_eval",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        f"- cell:{tag}",
        "SKILL_INPUT_MODE: ego",
        "HEURISTIC_MODE: env_specific",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_top()
    lines += [
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def _transfer_yaml(*, env: str, suite: str, tag: str) -> str:
    meta = TRANSFER_ENVS[env]
    stem = train_stem(suite, tag)
    tstem = transfer_stem(env, suite, tag)
    ckpt = (
        f"./outputs/models/paper_eval/rq1/{stem}/seaquest/"
        f"hierarchical_pqn_seaquest_seed44.safetensors"
    )
    lines: list[str] = [
        "PAPER_EVAL_RQ: 2",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: transfer",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_MOD_KEYS:",
        "- base",
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
        "USE_POV_RELATIVE_MAPPER: true",
        "USE_POV_FIXED_BILATERAL: false",
        "USE_POV_FACING_FRAME: true",
        "EGO_POSE_XY: false",
        "USE_LEGACY_EGO_OBS: false",
        "USE_EGO_SIZE_FEATURES: false",
        "USE_EGO_VELOCITY_FEATURES: false",
        "USE_EGO_BLOCKING_FEATURES: false",
        "USE_EGO_WEAPON_FEATURES: false",
        "USE_EGO_SHOT_FEATURES: false",
        "AVOID_INTERRUPT_RANGE: 55",
        "AVOID_INTERRUPT_RANGE_Y: 55",
        "SURFACE_SUB_AVOID_RANGE_X: 5",
        "SURFACE_SUB_AVOID_RANGE_Y: 5",
        "EGO_DISTANCE_CLIP_MAX_PX: null",
        "EGO_PSI_ANGLE_BINS: 8",
        "EGO_LEGAL_ACTION_MASK: false",
        "SURFACE_SUB_NEAR_RANGE: 5",
        f"SAVE_PATH: ./outputs/models/paper_eval/rq2/{tstem}",
        f"WANDB_RUN_NAME: {meta['prefix']}_paper_{suite}_{tag}_zs_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{suite}",
        "- rq:2",
        f"- env:{env}",
        "- phase:zero_shot",
        "- skill_input:ego",
        "- mapper:pov_facing",
        "- ego:phi_bins8",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        f"- ablation:{SPARSE_TAG}",
        f"- cell:{tag}",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    lines += _sparse_top()
    lines += [
        "alg:",
        "  NUM_SKILLS: 4",
        "  SKILL_NAMES:",
        "  - DESTROY",
        "  - COLLECT",
        "  - AVOID",
        "  - NAVIGATE",
        f"  MAX_OBJECTS: {meta['max_objects']}",
        "  EGO_ACTION_DIM: 18",
        "  HIDDEN_SIZE: 128",
        "  NUM_LAYERS: 3",
        "  SKILL_EMBED_DIM: 256",
        "  FRAME_SKIP: 4",
        "  FRAME_STACK_SIZE: 4",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += _sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return _lines(lines)


def cell_pipeline_stems(suite: str, tag: str) -> tuple[str, ...]:
    return (
        train_stem(suite, tag),
        eval_stem(suite, tag),
        transfer_stem("pong", suite, tag),
        transfer_stem("beamrider", suite, tag),
    )


def eps_train_stems() -> tuple[str, ...]:
    return tuple(train_stem(SUITE_EPS, t) for t in CELLS_EPS)


def eps_eval_stems() -> tuple[str, ...]:
    out: list[str] = []
    for tag in CELLS_EPS:
        out.extend(cell_pipeline_stems(SUITE_EPS, tag)[1:])
    return tuple(out)


def eps_sanity_stems() -> tuple[str, ...]:
    """Train-first so ``--gpus`` parallelizes the five ~5M trains."""
    return eps_train_stems() + eps_eval_stems()


def skills_sweep_stems() -> tuple[str, ...]:
    trains = tuple(train_stem(SUITE_SKILLS, t) for t in CELLS_SKILLS)
    evals: list[str] = []
    for tag in CELLS_SKILLS:
        evals.extend(cell_pipeline_stems(SUITE_SKILLS, tag)[1:])
    return trains + tuple(evals)


def tandem_sweep_stems() -> tuple[str, ...]:
    # Placeholder: tandem materialization not wired yet.
    return tuple()


def _materialize_eps(*, dry: bool) -> None:
    for tag, (eps_s, eps_f, eps_d) in CELLS_EPS.items():
        _write(
            _RQ1 / f"{train_stem(SUITE_EPS, tag)}.yaml",
            _skills_train_yaml(
                suite=SUITE_EPS,
                tag=tag,
                eps_start=eps_s,
                eps_finish=eps_f,
                eps_decay=eps_d,
            ),
            dry=dry,
        )
        _write(
            _RQ1 / f"{eval_stem(SUITE_EPS, tag)}.yaml",
            _skills_eval_yaml(suite=SUITE_EPS, tag=tag),
            dry=dry,
        )
        for env in TRANSFER_ENVS:
            _write(
                _RQ2 / f"{transfer_stem(env, SUITE_EPS, tag)}.yaml",
                _transfer_yaml(env=env, suite=SUITE_EPS, tag=tag),
                dry=dry,
            )


def _materialize_skills(*, dry: bool) -> None:
    if not CELLS_SKILLS:
        print(
            f"(skip {SUITE_SKILLS}: CELLS_SKILLS empty — fill in "
            "materialize_hp_sweep.py when Stage 1 runs)"
        )
        return
    for tag, overrides in CELLS_SKILLS.items():
        eps_s = float(overrides.get("EPS_START", 1.0))
        eps_f = float(overrides.get("EPS_FINISH", 0.001))
        eps_d = float(overrides.get("EPS_DECAY", 0.1))
        lr = float(overrides.get("SKILL_LR", SKILL_LR))
        _write(
            _RQ1 / f"{train_stem(SUITE_SKILLS, tag)}.yaml",
            _skills_train_yaml(
                suite=SUITE_SKILLS,
                tag=tag,
                eps_start=eps_s,
                eps_finish=eps_f,
                eps_decay=eps_d,
                skill_lr=lr,
                extra_alg=overrides,
            ),
            dry=dry,
        )
        _write(
            _RQ1 / f"{eval_stem(SUITE_SKILLS, tag)}.yaml",
            _skills_eval_yaml(suite=SUITE_SKILLS, tag=tag),
            dry=dry,
        )
        for env in TRANSFER_ENVS:
            _write(
                _RQ2 / f"{transfer_stem(env, SUITE_SKILLS, tag)}.yaml",
                _transfer_yaml(env=env, suite=SUITE_SKILLS, tag=tag),
                dry=dry,
            )


def _materialize_tandem(*, dry: bool) -> None:
    if not CELLS_TANDEM:
        print(
            f"(skip {SUITE_TANDEM}: CELLS_TANDEM empty — fill after skills lock)"
        )
        return
    raise NotImplementedError(
        "Tandem HP cells: wire train_tandem YAMLs here once Stage 3 starts."
    )


def materialize(*, stage: str, dry: bool) -> None:
    if stage in ("eps", "all"):
        _materialize_eps(dry=dry)
        stems = eps_sanity_stems()
        print(
            f"\n{SUITE_EPS}: {len(CELLS_EPS)} cells, {len(stems)} stems "
            f"(trains-first). MOD_KEYS={list(MOD_KEYS)}"
        )
        for tag, sched in CELLS_EPS.items():
            print(f"  {tag}: EPS {sched[0]} → {sched[1]} / {sched[2]}")
    if stage in ("skills", "all"):
        _materialize_skills(dry=dry)
        print(f"{SUITE_SKILLS}: {len(CELLS_SKILLS)} cells")
    if stage in ("tandem", "all"):
        _materialize_tandem(dry=dry)
        print(f"{SUITE_TANDEM}: {len(CELLS_TANDEM)} cells")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("eps", "skills", "tandem", "all"),
        default="eps",
        help="Which stage to materialize (default: eps sanity).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(stage=args.stage, dry=args.dry_run)


if __name__ == "__main__":
    main()

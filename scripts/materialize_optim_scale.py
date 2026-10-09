#!/usr/bin/env python3
"""Materialize Seaquest ego-skills + tandem-pixel at PQN optimal-scaling 1:1 HPs.

Reference (flat PQN pixel)::

    /home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml
    → TOTAL_TIMESTEPS=5e7, TOTAL_TIMESTEPS_DECAY=5e7,
      NUM_ENVS=1024, NUM_STEPS=32, NUM_EPOCHS=2, NUM_MINIBATCHES=256,
      EPS 1→0.001 over EPS_DECAY=0.1 of decay updates,
      LR=2.5e-4 + LR_LINEAR_DECAY, MAX_GRAD_NORM=10, GAMMA=0.99, LAMBDA=0.65.

CBL mapping (true 1:1 on shared HPs)
------------------------------------
* Outer: 1024 envs × NUM_STEPS=32 → NUM_UPDATES = floor(5e7/(1024×32)) = 1525
  → TOTAL_TIMESTEPS kept at 50_000_000 (same as YAML; trainers floor to 1525 updates).
* Skills: SKILL_LR=2.5e-4 + LR_LINEAR_DECAY (→1e-20 over TOTAL_TIMESTEPS_DECAY).
* Tandem: same for SKILL_LR and META_LR; same ε schedule on skill + meta heads.
* MAX_GRAD_NORM=10 (YAML), not the hierarchical template default 0.8.
* 3 train seeds via separate stems (44/45/46).
* Probe suite ``optim_scale_probe``: skills-only seed=1 arms that keep the
  YAML outer (32/1024/50M/grad10) but ablate LR/ε — run these before/alongside
  the full tandem-first suite to see what breaks.
* Skills final (post-NOOP), locked recipe = ε B + LR 2.5e-4+linear:
  - ``optim_scale_skills_bases``: phoenix / namethisgame / asteroids (done)
  - ``optim_scale_skills_transfers``: remaining EVAL_PLAN transfer scratch envs
    (spaceinvaders, beamrider, timepilot, pong, breakout, freeway)
  - ``optim_scale_pqn_pixel_envs``: flat PQN pixel @ 1:1 HPs × seeds 44/45/46
    on bases (phoenix/namethisgame/asteroids) + transfers (same paper_core mods)
  - ``optim_scale_skills_clip_foils``: locked εB × clip10/clip50 on Seaquest +
    robustness-fail envs (phoenix/asteroids/SI/BR/TP/Pong) — distance regime probe
  - ``optim_scale_skills_living_urgency``: same envs, noclip εB, but COLLECT +
    DESTROY living tax on (discourage idle/spam-until-kill)
  - ``optim_scale_pqn_oc_builds``: flat PQN on tandem OC feature builds
    (oc_simple_rel_role / oc_simple_role / oc_embed) × seed 44 + home eval
  - ``optim_scale_skills_lr`` / ``optim_scale_skills_old_eps``: Seaquest HP probes

    python scripts/materialize_optim_scale.py
    python scripts/run_paper_eval.py \\
      --suite optim_scale_skills_living_urgency --gpus 0,1,2 \\
      --wait-deps --lean --skip-rq-gate
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import materialize_final_10m as f10

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "configs/experiment/paper_eval/optim_scale"
_REF_SRC = Path("/home/paul/Downloads/pqn_jaxatari_pixel_optimal_scaling.yaml")

SUITE = "optim_scale_sq"
PROBE_SUITE = "optim_scale_probe"
BASES_SUITE = "optim_scale_skills_bases"
TRANSFERS_SUITE = "optim_scale_skills_transfers"
LR_SUITE = "optim_scale_skills_lr"
OLD_EPS_SUITE = "optim_scale_skills_old_eps"
CLIP_FOILS_SUITE = "optim_scale_skills_clip_foils"
LIVING_URGENCY_SUITE = "optim_scale_skills_living_urgency"
PQN_OC_SUITE = "optim_scale_pqn_oc"
PQN_PIXEL_SUITE = "optim_scale_pqn_pixel"
# Flat PQN pixel on non-Seaquest bases + transfer envs (matched vs skills).
PQN_PIXEL_ENVS_SUITE = "optim_scale_pqn_pixel_envs"
# Flat PQN on tandem-candidate OC feature builds (overnight format probe).
PQN_OC_BUILDS_SUITE = "optim_scale_pqn_oc_builds"
BUDGET_TAG = "50m_optim"
# Match YAML literally; NUM_UPDATES = TOTAL // (NUM_ENVS * NUM_STEPS).
TIMESTEPS = 50_000_000
TIMESTEPS_DECAY = 50_000_000
NUM_ENVS = 1024
NUM_STEPS = 32
NUM_UPDATES = TIMESTEPS // (NUM_ENVS * NUM_STEPS)  # 1525
NUM_EPOCHS = 2
NUM_MINIBATCHES = 256
LR = 0.00025
SKILL_LR_SMOKE = 0.0005  # old healthy constant LR
SKILL_LR_LOW = 0.0001
MAX_GRAD_NORM = 10.0
EPS_START = 1.0
EPS_FINISH = 0.001
EPS_DECAY = 0.1
# Skills lock ε B (eps_sanity).
EPS_B_START = 0.2
EPS_B_FINISH = 0.01
EPS_B_DECAY = 0.2
OUT_MODELS = "./outputs/models/paper_eval/optim_scale"

SEEDS = (44, 45, 46)
PROBE_SEED = 1
OLD_EPS_SEED = 44  # single YAML-ε comparison arm (Seaquest)
BASE_SEED = 44
# Paper bases excluding Seaquest (already covered by post-NOOP probes).
BASE_OTHER_ENVS: tuple[str, ...] = ("phoenix", "namethisgame", "asteroids")
# EVAL_PLAN transfer scratch envs not yet run at optim_scale lock.
TRANSFER_ENV_META: dict[str, dict] = {
    "spaceinvaders": {"max_objects": 41, "heu": "sweep", "prefix": "si", "rq": 2},
    "beamrider": {"max_objects": 31, "heu": "env_specific", "prefix": "br", "rq": 2},
    "timepilot": {"max_objects": 7, "heu": "env_specific", "prefix": "tp", "rq": 2},
    "pong": {"max_objects": 2, "heu": "nearest", "prefix": "po", "rq": 2},
    "breakout": {"max_objects": 3, "heu": "nearest", "prefix": "bo", "rq": 2},
    "freeway": {"max_objects": 12, "heu": "env_specific", "prefix": "fw", "rq": 2},
}
TRANSFER_ENVS: tuple[str, ...] = tuple(TRANSFER_ENV_META)
# Non-Seaquest envs that need matched flat PQN pixel baselines (3 seeds).
PQN_PIXEL_ENVS: tuple[str, ...] = BASE_OTHER_ENVS + TRANSFER_ENVS
# Distance-clip probe on Seaquest + envs where noclip skills flopped on foils.
CLIP_FOIL_CLIPS: tuple[int, ...] = (10, 50)
CLIP_FOIL_ENVS: tuple[str, ...] = (
    "seaquest",
    "phoenix",
    "asteroids",
    "spaceinvaders",
    "beamrider",
    "timepilot",
    "pong",
)
# Same env set as clip foils — urgency clock on COLLECT + DESTROY (noclip εB).
LIVING_URGENCY_ENVS: tuple[str, ...] = CLIP_FOIL_ENVS

# Tandem-candidate OC feature builds for flat PQN (same META_INPUT_MODE builders).
OC_BUILD_MODES: tuple[str, ...] = (
    "oc_simple_rel_role",
    "oc_simple_role",
    "oc_embed",
)
OC_BUILD_SEED = BASE_SEED
SEAQUEST_MAX_OBJECTS = 35

MOD_KEYS = f10.MOD_KEYS


def _oc_build_token(mode: str) -> str:
    return {
        "oc_simple_rel_role": "rel_role",
        "oc_simple_role": "role",
        "oc_embed": "embed",
    }[mode]

# Skills-only seed-1 probes: same outer as 1:1, ablate LR / ε.
# Order: control (full YAML) → isolate ε → isolate ε+old LR.
PROBE_ARMS: tuple[tuple[str, dict], ...] = (
    (
        "1to1",
        {
            "skill_lr": LR,
            "lr_linear_decay": True,
            "eps_start": EPS_START,
            "eps_finish": EPS_FINISH,
            "eps_decay": EPS_DECAY,
            "tag_lr": f"lr:{LR}",
            "tag_lr_decay": "lr_decay:linear",
            "tag_eps": "eps:1to0.001",
        },
    ),
    (
        "epsB",
        {
            "skill_lr": LR,
            "lr_linear_decay": True,
            "eps_start": EPS_B_START,
            "eps_finish": EPS_B_FINISH,
            "eps_decay": EPS_B_DECAY,
            "tag_lr": f"lr:{LR}",
            "tag_lr_decay": "lr_decay:linear",
            "tag_eps": "eps:B",
        },
    ),
    (
        "lr5e4_epsB",
        {
            "skill_lr": SKILL_LR_SMOKE,
            "lr_linear_decay": False,
            "eps_start": EPS_B_START,
            "eps_finish": EPS_B_FINISH,
            "eps_decay": EPS_B_DECAY,
            "tag_lr": f"lr:{SKILL_LR_SMOKE}",
            "tag_lr_decay": "lr_decay:none",
            "tag_eps": "eps:B",
        },
    ),
)


def _skill_hp(
    *,
    skill_lr: float,
    lr_linear_decay: bool,
    eps_start: float,
    eps_finish: float,
    eps_decay: float,
    tag_eps: str,
) -> dict:
    return {
        "skill_lr": skill_lr,
        "lr_linear_decay": lr_linear_decay,
        "eps_start": eps_start,
        "eps_finish": eps_finish,
        "eps_decay": eps_decay,
        "tag_lr": f"lr:{skill_lr}",
        "tag_lr_decay": "lr_decay:linear" if lr_linear_decay else "lr_decay:none",
        "tag_eps": tag_eps,
    }


# Short LR probe (seed=1, ε B fixed). Skips lr2p5e4_lin (→ epsB multi-seed)
# and lr5e4_const (→ already probe lr5e4_epsB).
LR_ARMS: tuple[tuple[str, dict], ...] = (
    (
        "lr1e4_lin",
        _skill_hp(
            skill_lr=SKILL_LR_LOW,
            lr_linear_decay=True,
            eps_start=EPS_B_START,
            eps_finish=EPS_B_FINISH,
            eps_decay=EPS_B_DECAY,
            tag_eps="eps:B",
        ),
    ),
    (
        "lr2p5e4_const",
        _skill_hp(
            skill_lr=LR,
            lr_linear_decay=False,
            eps_start=EPS_B_START,
            eps_finish=EPS_B_FINISH,
            eps_decay=EPS_B_DECAY,
            tag_eps="eps:B",
        ),
    ),
    (
        "lr5e4_lin",
        _skill_hp(
            skill_lr=SKILL_LR_SMOKE,
            lr_linear_decay=True,
            eps_start=EPS_B_START,
            eps_finish=EPS_B_FINISH,
            eps_decay=EPS_B_DECAY,
            tag_eps="eps:B",
        ),
    ),
)

EPSB_HP = _skill_hp(
    skill_lr=LR,
    lr_linear_decay=True,
    eps_start=EPS_B_START,
    eps_finish=EPS_B_FINISH,
    eps_decay=EPS_B_DECAY,
    tag_eps="eps:B",
)
OLD_EPS_HP = _skill_hp(
    skill_lr=LR,
    lr_linear_decay=True,
    eps_start=EPS_START,
    eps_finish=EPS_FINISH,
    eps_decay=EPS_DECAY,
    tag_eps="eps:1to0.001",
)


def skills_stem(seed: int) -> str:
    return f"optim_scale_skills_ego_seed{seed}"


def tandem_stem(seed: int) -> str:
    return f"optim_scale_tandem_pixel_ego_seed{seed}"


def probe_skills_stem(arm: str, seed: int = PROBE_SEED) -> str:
    return f"optim_scale_probe_skills_{arm}_seed{seed}"


def skills_lr_stem(arm: str) -> str:
    return f"optim_scale_skills_lr_{arm}_seed{PROBE_SEED}"


def skills_old_eps_stem() -> str:
    return f"optim_scale_skills_old_eps_seed{OLD_EPS_SEED}"


def skills_base_stem(env: str) -> str:
    return f"optim_scale_skills_bases_{env}"


def skills_transfer_stem(env: str) -> str:
    return f"optim_scale_skills_transfers_{env}"


def skills_clip_foil_stem(env: str, clip_px: int) -> str:
    return f"optim_scale_skills_clip{int(clip_px)}_{env}"


def skills_living_urgency_stem(env: str) -> str:
    return f"optim_scale_skills_living_urgency_{env}"


def pqn_stem(variant: str, seed: int) -> str:
    """variant in {pixel, oc}."""
    return f"optim_scale_pqn_{variant}_seed{seed}"


def pqn_pixel_env_stem(env: str, seed: int) -> str:
    """Flat PQN pixel on a non-Seaquest env (bases or transfers)."""
    return f"optim_scale_pqn_pixel_{env}_seed{seed}"


def pqn_oc_build_stem(mode: str, seed: int = OC_BUILD_SEED) -> str:
    return f"optim_scale_pqn_oc_build_{_oc_build_token(mode)}_seed{seed}"


def optim_scale_sq_stems() -> tuple[str, ...]:
    """Tandem trains first (OOM-fast), then skills, then home evals."""
    tandem_trains = tuple(tandem_stem(s) for s in SEEDS)
    skills_trains = tuple(skills_stem(s) for s in SEEDS)
    tandem_evals = tuple(f"{s}_eval" for s in tandem_trains)
    skills_evals = tuple(f"{s}_eval" for s in skills_trains)
    return tandem_trains + skills_trains + tandem_evals + skills_evals


def optim_scale_probe_stems() -> tuple[str, ...]:
    """Skills-only seed-1 HP probes (all trains, then all evals)."""
    trains = tuple(probe_skills_stem(arm) for arm, _ in PROBE_ARMS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_rerun_stems() -> tuple[str, ...]:
    """εB skills re-run × seeds 44/45/46 (nearest-combat heuristic); trains then evals."""
    trains = tuple(probe_skills_stem("epsB", seed) for seed in SEEDS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_bases_stems() -> tuple[str, ...]:
    """Locked recipe on other paper bases — trains first for --gpus sharding."""
    trains = tuple(skills_base_stem(env) for env in BASE_OTHER_ENVS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_transfers_stems() -> tuple[str, ...]:
    """Locked recipe on remaining transfer scratch envs — trains first for --gpus."""
    trains = tuple(skills_transfer_stem(env) for env in TRANSFER_ENVS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_clip_foils_stems() -> tuple[str, ...]:
    """clip10/clip50 × foil envs — trains first (clip major) for --gpus sharding."""
    trains = tuple(
        skills_clip_foil_stem(env, clip)
        for clip in CLIP_FOIL_CLIPS
        for env in CLIP_FOIL_ENVS
    )
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_living_urgency_stems() -> tuple[str, ...]:
    """COLLECT+DESTROY living tax on foil envs — trains first for --gpus."""
    trains = tuple(skills_living_urgency_stem(env) for env in LIVING_URGENCY_ENVS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_lr_stems() -> tuple[str, ...]:
    """Short LR probe seed=1 (ε B fixed) — trains first for --gpus sharding."""
    trains = tuple(skills_lr_stem(arm) for arm, _ in LR_ARMS)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_skills_old_eps_stems() -> tuple[str, ...]:
    """Single YAML-ε comparison arm (seed=44)."""
    stem = skills_old_eps_stem()
    return (stem, f"{stem}_eval")


def optim_scale_pqn_oc_stems() -> tuple[str, ...]:
    """GPU0: OC flat PQN — seed train→eval sequential (3 seeds)."""
    out: list[str] = []
    for seed in SEEDS:
        stem = pqn_stem("oc", seed)
        out.append(stem)
        out.append(f"{stem}_eval")
    return tuple(out)


def optim_scale_pqn_pixel_stems() -> tuple[str, ...]:
    """GPU1: pixel flat PQN — seed train→eval sequential (3 seeds)."""
    out: list[str] = []
    for seed in SEEDS:
        stem = pqn_stem("pixel", seed)
        out.append(stem)
        out.append(f"{stem}_eval")
    return tuple(out)


def optim_scale_pqn_pixel_envs_stems() -> tuple[str, ...]:
    """Bases+transfers flat PQN pixel — trains first for --gpus sharding."""
    trains = tuple(
        pqn_pixel_env_stem(env, seed) for env in PQN_PIXEL_ENVS for seed in SEEDS
    )
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def optim_scale_pqn_oc_builds_stems() -> tuple[str, ...]:
    """Overnight: OC builds — trains first for --gpus sharding, then evals."""
    trains = tuple(pqn_oc_build_stem(mode) for mode in OC_BUILD_MODES)
    evals = tuple(f"{s}_eval" for s in trains)
    return trains + evals


def _sub_alg_key(body: str, key: str, value: str) -> str:
    """Replace ``  KEY: ...`` under alg (2-space indent) if present; else no-op."""
    pat = re.compile(rf"(?m)^  {re.escape(key)}:.*$")
    repl = f"  {key}: {value}"
    if pat.search(body):
        return pat.sub(repl, body, count=1)
    return body


def _sub_top_key(body: str, key: str, value: str) -> str:
    """Replace top-level ``KEY: ...`` (no indent) if present; else no-op."""
    pat = re.compile(rf"(?m)^{re.escape(key)}:.*$")
    repl = f"{key}: {value}"
    if pat.search(body):
        return pat.sub(repl, body, count=1)
    return body


def _ensure_alg_key_after(body: str, after_key: str, key: str, value: str) -> str:
    """Insert ``  KEY: value`` after ``after_key`` if ``key`` is missing."""
    if re.search(rf"(?m)^  {re.escape(key)}:", body):
        return _sub_alg_key(body, key, value)
    pat = re.compile(rf"(?m)^(  {re.escape(after_key)}:.*)$")
    m = pat.search(body)
    if not m:
        return body
    insert = f"{m.group(1)}\n  {key}: {value}"
    return pat.sub(insert, body, count=1)


def _patch_1to1(
    body: str,
    *,
    kind: str,
    suite: str = SUITE,
    skill_lr: float = LR,
    lr_linear_decay: bool = True,
    eps_start: float = EPS_START,
    eps_finish: float = EPS_FINISH,
    eps_decay: float = EPS_DECAY,
    tag_lr: str | None = None,
    tag_lr_decay: str | None = None,
    tag_eps: str | None = None,
    scale_tag: str = "pqn_optim_1to1",
) -> str:
    """Retarget final_10m body → YAML outer + chosen LR/ε/grad clip."""
    if tag_lr is None:
        tag_lr = f"lr:{skill_lr}"
    if tag_lr_decay is None:
        tag_lr_decay = "lr_decay:linear" if lr_linear_decay else "lr_decay:none"
    if tag_eps is None:
        tag_eps = f"eps:{eps_start}to{eps_finish}"

    reps = [
        ("  TOTAL_TIMESTEPS: 9994240", f"  TOTAL_TIMESTEPS: {TIMESTEPS}"),
        ("  NUM_ENVS: 128", f"  NUM_ENVS: {NUM_ENVS}"),
        ("  NUM_STEPS: 128", f"  NUM_STEPS: {NUM_STEPS}"),
        ("  NUM_EPOCHS: 4", f"  NUM_EPOCHS: {NUM_EPOCHS}"),
        ("  NUM_MINIBATCHES: 64", f"  NUM_MINIBATCHES: {NUM_MINIBATCHES}"),
        ("- budget:10m", f"- budget:{BUDGET_TAG}"),
        ("- suite:final_10m", f"- suite:{suite}"),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_SKILLS}",
            "SAVE_PATH: PLACEHOLDER",
        ),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_TANDEM_PIXEL}",
            "SAVE_PATH: PLACEHOLDER",
        ),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_PQN_OC}",
            "SAVE_PATH: PLACEHOLDER",
        ),
        (
            f"SAVE_PATH: {f10.OUT_MODELS}/{f10.STEM_PQN_PIXEL}",
            "SAVE_PATH: PLACEHOLDER",
        ),
    ]
    for old, new in reps:
        body = body.replace(old, new)

    # Shared outer HPs. Skills/tandem nest under ``alg:`` (2-space); flat PQN is top-level.
    sub = _sub_top_key if kind == "flat_pqn" else _sub_alg_key
    body = sub(body, "NUM_STEPS", str(NUM_STEPS))
    body = sub(body, "NUM_ENVS", str(NUM_ENVS))
    body = sub(body, "NUM_EPOCHS", str(NUM_EPOCHS))
    body = sub(body, "NUM_MINIBATCHES", str(NUM_MINIBATCHES))
    body = sub(body, "TOTAL_TIMESTEPS", str(TIMESTEPS))
    body = sub(body, "TOTAL_TIMESTEPS_DECAY", str(TIMESTEPS_DECAY))
    if kind != "flat_pqn":
        body = _ensure_alg_key_after(
            body, "TOTAL_TIMESTEPS", "TOTAL_TIMESTEPS_DECAY", str(TIMESTEPS_DECAY)
        )
    body = sub(body, "LR", str(skill_lr))
    if kind != "flat_pqn":
        body = _sub_alg_key(body, "SKILL_LR", str(skill_lr))
    body = sub(body, "MAX_GRAD_NORM", str(MAX_GRAD_NORM))
    body = sub(body, "LR_LINEAR_DECAY", "true" if lr_linear_decay else "false")
    if kind != "flat_pqn":
        body = _ensure_alg_key_after(
            body,
            "MAX_GRAD_NORM",
            "LR_LINEAR_DECAY",
            "true" if lr_linear_decay else "false",
        )

    body = sub(body, "EPS_START", str(eps_start))
    body = sub(body, "EPS_FINISH", str(eps_finish))
    body = sub(body, "EPS_DECAY", str(eps_decay))

    if kind == "tandem":
        body = _sub_alg_key(body, "META_LR", str(skill_lr))
        body = _sub_alg_key(body, "SKILL_EPS_START", str(eps_start))
        body = _sub_alg_key(body, "SKILL_EPS_FINISH", str(eps_finish))
        body = _sub_alg_key(body, "SKILL_EPS_DECAY", str(eps_decay))
        body = _sub_alg_key(body, "META_EPS_START", str(eps_start))
        body = _sub_alg_key(body, "META_EPS_FINISH", str(eps_finish))
        body = _sub_alg_key(body, "META_EPS_DECAY", str(eps_decay))

    for drop in (
        "- lock:skills_lr1e3\n",
        "- lock:skills_lr5e4\n",
        "- lock:tandem_e01\n",
        "- lr:constant\n",
    ):
        body = body.replace(drop, "")
    # Drop any prior scale tag block so re-materialize is idempotent.
    body = re.sub(
        r"(?m)^- scale:pqn_optim[^\n]*\n"
        r"(?:- num_envs:\d+\n)?"
        r"(?:- num_steps:\d+\n)?"
        r"(?:- num_updates:\d+\n)?"
        r"(?:- num_epochs:\d+\n)?"
        r"(?:- num_minibatches:\d+\n)?"
        r"(?:- lr:[^\n]+\n)?"
        r"(?:- lr_decay:[^\n]+\n)?"
        r"(?:- max_grad_norm:[^\n]+\n)?"
        r"(?:- eps:[^\n]+\n)?"
        r"(?:- probe:[^\n]+\n)?",
        "",
        body,
    )
    needle = f"- budget:{BUDGET_TAG}\n"
    insert = (
        f"- budget:{BUDGET_TAG}\n"
        f"- scale:{scale_tag}\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_steps:{NUM_STEPS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
        f"- num_epochs:{NUM_EPOCHS}\n"
        f"- num_minibatches:{NUM_MINIBATCHES}\n"
        f"- {tag_lr}\n"
        f"- {tag_lr_decay}\n"
        f"- max_grad_norm:{int(MAX_GRAD_NORM)}\n"
        f"- {tag_eps}\n"
    )
    if needle in body and f"- scale:{scale_tag}\n" not in body:
        body = body.replace(needle, insert, 1)
    return body


def _retarget(
    body: str,
    *,
    stem: str,
    seed: int,
    header: str,
    run_name: str,
    kind: str,
    suite: str = SUITE,
    env: str = "seaquest",
    patch_kwargs: dict | None = None,
) -> str:
    body = _patch_1to1(body, kind=kind, suite=suite, **(patch_kwargs or {}))
    lines = body.splitlines()
    if lines and lines[0].startswith("#"):
        lines[0] = header
    out: list[str] = []
    for line in lines:
        if line.startswith("WANDB_RUN_NAME:"):
            out.append(f"WANDB_RUN_NAME: {run_name}")
        elif line.startswith("SAVE_PATH:"):
            out.append(f"SAVE_PATH: {OUT_MODELS}/{stem}")
        elif line.startswith("SEED:"):
            out.append(f"SEED: {seed}")
        elif line.startswith("EVAL_SEED:"):
            out.append(f"EVAL_SEED: {seed + 10_000}")
        elif "PAPER_EVAL_REQUIRES_RQ1:" in line:
            out.append(line)
        elif line.strip() in ("- PLACEHOLDER_STEM", "- PLACEHOLDER"):
            out.append(f"- {stem}")
        elif line.strip().startswith("- final_10m_") or line.strip().startswith(
            "- optim_scale_"
        ):
            out.append(f"- {stem}")
        elif line.startswith("PAPER_EVAL_ARCHITECTURE_OF:"):
            out.append(f"PAPER_EVAL_ARCHITECTURE_OF: {stem}")
        elif line.startswith("PAPER_EVAL_SKILLS_CKPT:"):
            if kind == "skills":
                ckpt = (
                    f"{OUT_MODELS}/{stem}/{env}/"
                    f"hierarchical_pqn_{env}_seed{seed}.safetensors"
                )
            else:
                ckpt = (
                    f"{OUT_MODELS}/{stem}/{env}/"
                    f"hierarchical_pqn_{env}_seed{seed}_skills.safetensors"
                )
            out.append(f"PAPER_EVAL_SKILLS_CKPT: {ckpt}")
        elif line.startswith("PAPER_EVAL_META_CKPT:"):
            ckpt = (
                f"{OUT_MODELS}/{stem}/{env}/"
                f"hierarchical_pqn_{env}_seed{seed}_meta.safetensors"
            )
            out.append(f"PAPER_EVAL_META_CKPT: {ckpt}")
        elif line.startswith("PAPER_EVAL_FLAT_PQN_CKPT:"):
            variant = "oc" if "pqn_oc" in stem else "pixel"
            ckpt = (
                f"{OUT_MODELS}/{stem}/{env}/"
                f"flat_pqn_{env}_{variant}_seed{seed}_vmap0.safetensors"
            )
            out.append(f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}")
        else:
            out.append(line)
    body = "\n".join(out) + "\n"
    body = body.replace(f"- {f10.STEM_SKILLS}\n", f"- {stem}\n")
    body = body.replace(f"- {f10.STEM_TANDEM_PIXEL}\n", f"- {stem}\n")
    body = body.replace(f"- {f10.STEM_PQN_OC}\n", f"- {stem}\n")
    body = body.replace(f"- {f10.STEM_PQN_PIXEL}\n", f"- {stem}\n")
    for e in f10.BASE_SKILL_ENVS:
        body = body.replace(f"- {f10.skills_stem(e)}\n", f"- {stem}\n")
        body = body.replace(f"- {f10.pqn_pixel_stem(e)}\n", f"- {stem}\n")
    body = body.replace(f"{f10.OUT_MODELS}/", f"{OUT_MODELS}/")
    # Path-only placeholder (never PLACEHOLDER_STEM — that is handled above).
    body = body.replace("/PLACEHOLDER/", f"/{stem}/")
    body = body.replace(": PLACEHOLDER\n", f": {OUT_MODELS}/{stem}\n")
    return body


def _pqn_train(*, variant: str, seed: int) -> str:
    object_centric = variant == "oc"
    stem = pqn_stem(variant, seed)
    suite = PQN_OC_SUITE if object_centric else PQN_PIXEL_SUITE
    base = f10._pqn_train(env="seaquest", object_centric=object_centric)
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=suite,
        header=(
            f"# Flat PQN {variant} — Seaquest optim_scale 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay; "
            f"MAX_GRAD_NORM={MAX_GRAD_NORM:g}; ε {EPS_START}→{EPS_FINISH}); "
            f"seed={seed}."
        ),
        run_name=f"sq_{suite}_seed{seed}_{BUDGET_TAG}_1to1",
        kind="flat_pqn",
    )


def _pqn_eval(*, variant: str, seed: int) -> str:
    object_centric = variant == "oc"
    stem = pqn_stem(variant, seed)
    suite = PQN_OC_SUITE if object_centric else PQN_PIXEL_SUITE
    base = f10._pqn_eval(env="seaquest", object_centric=object_centric)
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=suite,
        header=(
            f"# Home mod eval — optim_scale 1:1 flat PQN {variant} seed{seed}."
        ),
        run_name=f"sq_{suite}_eval_seed{seed}_{BUDGET_TAG}_1to1",
        kind="flat_pqn",
    )


def _pqn_pixel_env_train(env: str, seed: int) -> str:
    """Flat PQN pixel train for a non-Seaquest base/transfer env."""
    meta = _env_meta(env)
    stem = pqn_pixel_env_stem(env, seed)
    role = "base" if env in BASE_OTHER_ENVS else "transfer"
    rq = int(meta["rq"])
    prefix = meta["prefix"]
    return f10._lines(
        [
            f"# Flat PQN pixel — {env} optim_scale 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay; "
            f"MAX_GRAD_NORM={MAX_GRAD_NORM:g}; ε {EPS_START}→{EPS_FINISH}); "
            f"seed={seed}.",
            f"PAPER_EVAL_RQ: {rq}",
            "PAPER_EVAL_PHASE: train",
            "PAPER_EVAL_KIND: flat_pqn",
            f"PAPER_EVAL_MODS: {env}",
            "PAPER_EVAL_REQUIRES_RQ1: []",
            "NUM_SEEDS: 1",
            "ALG_NAME: flat_pqn",
            f"ENV_NAME: {env}",
            "TRAIN_MODE: flat_pqn",
            "OBJECT_CENTRIC: false",
            "PIXEL_RESIZE: true",
            "PIXEL_RESIZE_SHAPE:",
            "- 84",
            "- 84",
            "PIXEL_GRAYSCALE: true",
            "USE_NATIVE_DOWNSCALING: true",
            "NORM_TYPE: layer_norm",
            "NORM_INPUT: false",
            "LR_LINEAR_DECAY: true",
            f"SAVE_PATH: {OUT_MODELS}/{stem}",
            f"WANDB_RUN_NAME: {prefix}_{PQN_PIXEL_ENVS_SUITE}_seed{seed}_{BUDGET_TAG}",
            "WANDB_TAGS:",
            f"- suite:{PQN_PIXEL_ENVS_SUITE}",
            f"- rq:{rq}",
            f"- env:{env}",
            f"- role:{role}",
            "- train_mode:flat_pqn",
            "- variant:pqn_pixel_paper_hps",
            f"- budget:{BUDGET_TAG}",
            "- scale:pqn_optim_1to1",
            f"- num_envs:{NUM_ENVS}",
            f"- num_steps:{NUM_STEPS}",
            f"- num_updates:{NUM_UPDATES}",
            f"- num_epochs:{NUM_EPOCHS}",
            f"- num_minibatches:{NUM_MINIBATCHES}",
            f"- lr:{LR}",
            "- lr_decay:linear",
            f"- max_grad_norm:{int(MAX_GRAD_NORM)}",
            f"- eps:{EPS_START}to{EPS_FINISH}",
            f"TOTAL_TIMESTEPS: {TIMESTEPS}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_DECAY}",
            f"NUM_ENVS: {NUM_ENVS}",
            f"NUM_STEPS: {NUM_STEPS}",
            f"NUM_EPOCHS: {NUM_EPOCHS}",
            f"NUM_MINIBATCHES: {NUM_MINIBATCHES}",
            "FRAME_SKIP: 4",
            "FRAME_STACK_SIZE: 4",
            "FIRST_FIRE: true",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "CLIP_REWARD: true",
            "GAMMA: 0.99",
            "LAMBDA: 0.65",
            f"LR: {LR}",
            f"MAX_GRAD_NORM: {MAX_GRAD_NORM}",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 2",
            f"EPS_START: {EPS_START}",
            f"EPS_FINISH: {EPS_FINISH}",
            f"EPS_DECAY: {EPS_DECAY}",
            "GAME_MODS: null",
            "EVAL_GAME_MODS: null",
            f"SEED: {seed}",
        ]
    )


def _pqn_pixel_env_eval(env: str, seed: int) -> str:
    """Home paper_core mod eval for non-Seaquest flat PQN pixel."""
    meta = _env_meta(env)
    stem = pqn_pixel_env_stem(env, seed)
    role = "base" if env in BASE_OTHER_ENVS else "transfer"
    rq = int(meta["rq"])
    prefix = meta["prefix"]
    mod_keys = f10._paper_core_keys(env)
    ckpt = (
        f"{OUT_MODELS}/{stem}/{env}/"
        f"flat_pqn_{env}_pixel_seed{seed}_vmap0.safetensors"
    )
    lines = [
        f"# Home mod eval — optim_scale 1:1 flat PQN pixel {env} seed{seed}.",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: flat_pqn",
        f"PAPER_EVAL_MODS: {env}",
    ]
    lines += f10._mod_keys_yaml(mod_keys)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}",
        "PAPER_EVAL_OBJECT_CENTRIC: false",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: flat_pqn",
        f"ENV_NAME: {env}",
        "OBJECT_CENTRIC: false",
        "PIXEL_RESIZE: true",
        "PIXEL_RESIZE_SHAPE:",
        "- 84",
        "- 84",
        "PIXEL_GRAYSCALE: true",
        "USE_NATIVE_DOWNSCALING: true",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: {prefix}_{PQN_PIXEL_ENVS_SUITE}_eval_seed{seed}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{PQN_PIXEL_ENVS_SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        f"- role:{role}",
        "- phase:mod_eval",
        "- train_mode:flat_pqn",
        "- variant:pqn_pixel",
        f"- budget:{BUDGET_TAG}",
        "- scale:pqn_optim_1to1",
        f"SEED: {seed}",
        f"EVAL_SEED: {seed + 10_000}",
        "EVAL_NUM_SEEDS: 3",
    ]
    return f10._lines(lines)


def _pqn_oc_build_train(mode: str, *, seed: int = OC_BUILD_SEED) -> str:
    tok = _oc_build_token(mode)
    stem = pqn_oc_build_stem(mode, seed)
    tag_block = (
        f"- suite:{PQN_OC_BUILDS_SUITE}\n"
        f"- rq:1\n"
        f"- env:seaquest\n"
        f"- role:base\n"
        f"- train_mode:flat_pqn\n"
        f"- variant:pqn_oc_build\n"
        f"- oc_input:{mode}\n"
        f"- budget:{BUDGET_TAG}\n"
        f"- scale:pqn_optim_1to1\n"
        f"- num_envs:{NUM_ENVS}\n"
        f"- num_steps:{NUM_STEPS}\n"
        f"- num_updates:{NUM_UPDATES}\n"
        f"- num_epochs:{NUM_EPOCHS}\n"
        f"- num_minibatches:{NUM_MINIBATCHES}\n"
        f"- lr:{LR}\n"
        f"- lr_decay:linear\n"
        f"- max_grad_norm:{int(MAX_GRAD_NORM)}\n"
        f"- eps:{EPS_START}to{EPS_FINISH}\n"
    )
    return f10._lines(
        [
            f"# Flat PQN OC build={mode} — Seaquest optim_scale 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear; "
            f"MAX_GRAD_NORM={MAX_GRAD_NORM:g}; ε {EPS_START}→{EPS_FINISH}); "
            f"seed={seed}. Same feature builders as tandem META_INPUT_MODE.",
            "PAPER_EVAL_RQ: 1",
            "PAPER_EVAL_PHASE: train",
            "PAPER_EVAL_KIND: flat_pqn",
            "PAPER_EVAL_MODS: seaquest",
            "PAPER_EVAL_REQUIRES_RQ1: []",
            "NUM_SEEDS: 1",
            "ALG_NAME: flat_pqn",
            "ENV_NAME: seaquest",
            "TRAIN_MODE: flat_pqn",
            "OBJECT_CENTRIC: true",
            f"OC_INPUT_MODE: {mode}",
            f"MAX_OBJECTS: {SEAQUEST_MAX_OBJECTS}",
            "NORM_TYPE: layer_norm",
            "NORM_INPUT: false",
            "LR_LINEAR_DECAY: true",
            f"SAVE_PATH: {OUT_MODELS}/{stem}",
            f"WANDB_RUN_NAME: sq_{PQN_OC_BUILDS_SUITE}_{tok}_seed{seed}_{BUDGET_TAG}",
            "WANDB_TAGS:",
            *[line for line in tag_block.strip().splitlines()],
            f"TOTAL_TIMESTEPS: {TIMESTEPS}",
            f"TOTAL_TIMESTEPS_DECAY: {TIMESTEPS_DECAY}",
            f"NUM_ENVS: {NUM_ENVS}",
            f"NUM_STEPS: {NUM_STEPS}",
            f"NUM_EPOCHS: {NUM_EPOCHS}",
            f"NUM_MINIBATCHES: {NUM_MINIBATCHES}",
            "FRAME_SKIP: 4",
            "FRAME_STACK_SIZE: 4",
            "FIRST_FIRE: true",
            "STICKY_ACTIONS: false",
            "EPISODIC_LIFE: true",
            "CLIP_REWARD: true",
            "GAMMA: 0.99",
            "LAMBDA: 0.65",
            f"LR: {LR}",
            f"MAX_GRAD_NORM: {MAX_GRAD_NORM}",
            "HIDDEN_SIZE: 128",
            "NUM_LAYERS: 2",
            f"EPS_START: {EPS_START}",
            f"EPS_FINISH: {EPS_FINISH}",
            f"EPS_DECAY: {EPS_DECAY}",
            "GAME_MODS: null",
            "EVAL_GAME_MODS: null",
            f"SEED: {seed}",
        ]
    )


def _pqn_oc_build_eval(mode: str, *, seed: int = OC_BUILD_SEED) -> str:
    tok = _oc_build_token(mode)
    stem = pqn_oc_build_stem(mode, seed)
    ckpt = (
        f"{OUT_MODELS}/{stem}/seaquest/"
        f"flat_pqn_seaquest_oc_{tok}_seed{seed}_vmap0.safetensors"
    )
    lines = [
        f"# Home mod eval — optim_scale flat PQN OC build={mode} seed{seed}.",
        "PAPER_EVAL_RQ: 1",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: flat_pqn",
        "PAPER_EVAL_MODS: seaquest",
    ]
    lines += f10._mod_keys_yaml(MOD_KEYS)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        f"- {stem}",
        f"PAPER_EVAL_FLAT_PQN_CKPT: {ckpt}",
        "PAPER_EVAL_OBJECT_CENTRIC: true",
        f"PAPER_EVAL_ARCHITECTURE_OF: {stem}",
        "NUM_SEEDS: 1",
        "ALG_NAME: flat_pqn",
        "ENV_NAME: seaquest",
        "OBJECT_CENTRIC: true",
        f"OC_INPUT_MODE: {mode}",
        f"MAX_OBJECTS: {SEAQUEST_MAX_OBJECTS}",
        f"SAVE_PATH: {OUT_MODELS}/{stem}",
        f"WANDB_RUN_NAME: sq_{PQN_OC_BUILDS_SUITE}_eval_{tok}_seed{seed}_{BUDGET_TAG}",
        "WANDB_TAGS:",
        f"- suite:{PQN_OC_BUILDS_SUITE}",
        "- rq:1",
        "- env:seaquest",
        "- phase:mod_eval",
        "- train_mode:flat_pqn",
        "- variant:pqn_oc_build",
        f"- oc_input:{mode}",
        f"- budget:{BUDGET_TAG}",
        f"SEED: {seed}",
        f"EVAL_SEED: {seed + 10_000}",
        "EVAL_NUM_SEEDS: 3",
    ]
    return f10._lines(lines)


def _skills_train(seed: int) -> str:
    stem = skills_stem(seed)
    base = f10._skills_train("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=(
            f"# Seaquest ego skills — PQN optimal-scaling 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay; "
            f"MAX_GRAD_NORM={MAX_GRAD_NORM:g}; ε {EPS_START}→{EPS_FINISH}); "
            f"seed={seed}."
        ),
        run_name=f"sq_{SUITE}_skills_ego_seed{seed}_{BUDGET_TAG}_1to1",
        kind="skills",
    )


def _skills_eval(seed: int) -> str:
    stem = skills_stem(seed)
    base = f10._skills_eval("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=f"# Home mod eval — optim_scale 1:1 skills seed{seed}.",
        run_name=f"sq_{SUITE}_eval_skills_ego_seed{seed}_{BUDGET_TAG}_1to1",
        kind="skills",
    )


def _tandem_train(seed: int) -> str:
    stem = tandem_stem(seed)
    base = f10._tandem_train(
        stem=f10.STEM_TANDEM_PIXEL,
        meta_mode="pixel_embed",
        meta_tag="pixel",
    )
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=(
            f"# Seaquest tandem pixel — PQN optimal-scaling 1:1 "
            f"({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}; LR={LR}+linear decay "
            f"skill+meta; MAX_GRAD_NORM={MAX_GRAD_NORM:g}); seed={seed}. "
            f"Soft risk: pixel meta @ {NUM_ENVS} envs may OOM."
        ),
        run_name=f"sq_{SUITE}_tandem_pixel_seed{seed}_{BUDGET_TAG}_1to1",
        kind="tandem",
    )


def _tandem_eval(seed: int) -> str:
    stem = tandem_stem(seed)
    base = f10._tandem_eval(
        stem=f10.STEM_TANDEM_PIXEL,
        meta_mode="pixel_embed",
        meta_tag="pixel",
    )
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        header=f"# Home mod eval — optim_scale 1:1 tandem pixel seed{seed}.",
        run_name=f"sq_{SUITE}_eval_tandem_pixel_seed{seed}_{BUDGET_TAG}_1to1",
        kind="tandem",
    )


def _probe_skills_train(arm: str, hp: dict, *, seed: int = PROBE_SEED) -> str:
    stem = probe_skills_stem(arm, seed)
    base = f10._skills_train("seaquest")
    lr = hp["skill_lr"]
    decay = "linear" if hp["lr_linear_decay"] else "const"
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=PROBE_SUITE,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": f"pqn_optim_probe_{arm}",
        },
        header=(
            f"# Seaquest ego skills PROBE seed={seed} arm={arm}: "
            f"outer 1:1 ({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}, grad={MAX_GRAD_NORM:g}); "
            f"SKILL_LR={lr} ({decay}); "
            f"ε {hp['eps_start']}→{hp['eps_finish']}/decay={hp['eps_decay']}."
        ),
        run_name=f"sq_{PROBE_SUITE}_skills_{arm}_seed{seed}_{BUDGET_TAG}",
        kind="skills",
    )


def _probe_skills_eval(arm: str, hp: dict, *, seed: int = PROBE_SEED) -> str:
    stem = probe_skills_stem(arm, seed)
    base = f10._skills_eval("seaquest")
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=PROBE_SUITE,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": f"pqn_optim_probe_{arm}",
        },
        header=f"# Home mod eval — optim_scale probe skills {arm} seed{seed}.",
        run_name=f"sq_{PROBE_SUITE}_eval_skills_{arm}_seed{seed}_{BUDGET_TAG}",
        kind="skills",
    )


def _skills_hp_train(
    *,
    stem: str,
    seed: int,
    suite: str,
    hp: dict,
    header: str,
    run_name: str,
    scale_tag: str,
    env: str = "seaquest",
) -> str:
    base = _skills_train_body(env)
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=suite,
        env=env,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": scale_tag,
        },
        header=header,
        run_name=run_name,
        kind="skills",
    )


def _skills_hp_eval(
    *,
    stem: str,
    seed: int,
    suite: str,
    hp: dict,
    header: str,
    run_name: str,
    scale_tag: str,
    env: str = "seaquest",
) -> str:
    base = _skills_eval_body(env)
    return _retarget(
        base,
        stem=stem,
        seed=seed,
        suite=suite,
        env=env,
        patch_kwargs={
            "skill_lr": hp["skill_lr"],
            "lr_linear_decay": hp["lr_linear_decay"],
            "eps_start": hp["eps_start"],
            "eps_finish": hp["eps_finish"],
            "eps_decay": hp["eps_decay"],
            "tag_lr": hp["tag_lr"],
            "tag_lr_decay": hp["tag_lr_decay"],
            "tag_eps": hp["tag_eps"],
            "scale_tag": scale_tag,
        },
        header=header,
        run_name=run_name,
        kind="skills",
    )


def _apply_distance_clip(body: str, clip_px: int) -> str:
    """Set EGO_DISTANCE_CLIP_MAX_PX and swap ego:noclip tag → ego:clipN."""
    clip = int(clip_px)
    body = re.sub(
        r"(?m)^EGO_DISTANCE_CLIP_MAX_PX:.*$",
        f"EGO_DISTANCE_CLIP_MAX_PX: {clip}",
        body,
    )
    body = body.replace("- ego:noclip\n", f"- ego:clip{clip}\n")
    if f"- ego:clip{clip}\n" not in body:
        # Insert after budget/scale block if noclip tag was absent.
        needle = f"- scale:pqn_optim_epsB_clip{clip}\n"
        if needle in body and f"- ego:clip{clip}\n" not in body:
            body = body.replace(needle, f"{needle}- ego:clip{clip}\n", 1)
    return body


def _clip_foil_train(env: str, clip_px: int) -> str:
    stem = skills_clip_foil_stem(env, clip_px)
    prefix = _env_meta(env)["prefix"]
    clip = int(clip_px)
    body = _skills_hp_train(
        stem=stem,
        seed=BASE_SEED,
        suite=CLIP_FOILS_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# {env} ego skills — optim_scale εB + clip{clip} "
            f"(LR={LR}+linear; {NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}); "
            f"seed={BASE_SEED}. Foil / distance-regime probe."
        ),
        run_name=f"{prefix}_{CLIP_FOILS_SUITE}_clip{clip}_{BUDGET_TAG}",
        scale_tag=f"pqn_optim_epsB_clip{clip}",
    )
    return _apply_distance_clip(body, clip)


def _clip_foil_eval(env: str, clip_px: int) -> str:
    stem = skills_clip_foil_stem(env, clip_px)
    prefix = _env_meta(env)["prefix"]
    clip = int(clip_px)
    body = _skills_hp_eval(
        stem=stem,
        seed=BASE_SEED,
        suite=CLIP_FOILS_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# Home mod eval — optim_scale εB clip{clip} {env} seed{BASE_SEED}."
        ),
        run_name=f"{prefix}_{CLIP_FOILS_SUITE}_eval_clip{clip}_{BUDGET_TAG}",
        scale_tag=f"pqn_optim_epsB_clip{clip}",
    )
    return _apply_distance_clip(body, clip)


def _apply_living_urgency(body: str) -> str:
    """COLLECT + DESTROY living tax; turn off COLLECT_SPARSE_ONLY so tax applies.

    Keeps AVOID_SPARSE_ONLY and DESTROY_CONTACT_FAIL from the sparse lock.
    """
    body = re.sub(
        r"(?m)^DESTROY_LIVING_PENALTY:.*$",
        "DESTROY_LIVING_PENALTY: true",
        body,
    )
    body = re.sub(
        r"(?m)^  DESTROY_LIVING_PENALTY:.*$",
        "  DESTROY_LIVING_PENALTY: true",
        body,
    )
    body = re.sub(
        r"(?m)^COLLECT_SPARSE_ONLY:.*$",
        "COLLECT_SPARSE_ONLY: false",
        body,
    )
    body = re.sub(
        r"(?m)^  COLLECT_SPARSE_ONLY:.*$",
        "  COLLECT_SPARSE_ONLY: false",
        body,
    )
    # Insert COLLECT_LIVING_PENALTY after COLLECT_SPARSE_ONLY if missing.
    if not re.search(r"(?m)^COLLECT_LIVING_PENALTY:", body):
        body = re.sub(
            r"(?m)^(COLLECT_SPARSE_ONLY:.*)$",
            r"\1\nCOLLECT_LIVING_PENALTY: true",
            body,
            count=1,
        )
    else:
        body = re.sub(
            r"(?m)^COLLECT_LIVING_PENALTY:.*$",
            "COLLECT_LIVING_PENALTY: true",
            body,
        )
    if not re.search(r"(?m)^  COLLECT_LIVING_PENALTY:", body):
        body = re.sub(
            r"(?m)^(  COLLECT_SPARSE_ONLY:.*)$",
            r"\1\n  COLLECT_LIVING_PENALTY: true",
            body,
            count=1,
        )
    else:
        body = re.sub(
            r"(?m)^  COLLECT_LIVING_PENALTY:.*$",
            "  COLLECT_LIVING_PENALTY: true",
            body,
        )
    body = body.replace(
        "- ablation:sparse_cda\n",
        "- ablation:living_urgency\n",
    )
    if "- reward:living_urgency\n" not in body:
        body = body.replace(
            "- ablation:living_urgency\n",
            "- ablation:living_urgency\n- reward:living_urgency\n",
        )
    return body


def _living_urgency_train(env: str) -> str:
    stem = skills_living_urgency_stem(env)
    prefix = _env_meta(env)["prefix"]
    body = _skills_hp_train(
        stem=stem,
        seed=BASE_SEED,
        suite=LIVING_URGENCY_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# {env} ego skills — optim_scale εB + COLLECT/DESTROY living tax "
            f"(noclip; {NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}); seed={BASE_SEED}. "
            f"Discourage idle/spam-until-kill."
        ),
        run_name=f"{prefix}_{LIVING_URGENCY_SUITE}_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB_living",
    )
    return _apply_living_urgency(body)


def _living_urgency_eval(env: str) -> str:
    stem = skills_living_urgency_stem(env)
    prefix = _env_meta(env)["prefix"]
    body = _skills_hp_eval(
        stem=stem,
        seed=BASE_SEED,
        suite=LIVING_URGENCY_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# Home mod eval — optim_scale εB living-urgency {env} "
            f"seed{BASE_SEED}."
        ),
        run_name=f"{prefix}_{LIVING_URGENCY_SUITE}_eval_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB_living",
    )
    return _apply_living_urgency(body)


def _env_meta(env: str) -> dict:
    if env in f10.BASE_SKILL_ENVS:
        return f10.BASE_SKILL_ENVS[env]
    if env in TRANSFER_ENV_META:
        return TRANSFER_ENV_META[env]
    raise KeyError(f"unknown env for optim_scale skills: {env}")


def _skills_train_body(env: str) -> str:
    """final_10m-shaped skills train YAML for base or transfer envs."""
    if env in f10.BASE_SKILL_ENVS:
        return f10._skills_train(env)
    meta = TRANSFER_ENV_META[env]
    rq = int(meta["rq"])
    lines = [
        f"# {env} ego skills — final_10m template (retargeted to optim_scale).",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: train",
        "PAPER_EVAL_KIND: skills",
        f"PAPER_EVAL_MODS: {env}",
        "PAPER_EVAL_REQUIRES_RQ1: []",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
    ]
    lines += f10._ego_lines(seaquest_legacy=False)
    lines += [
        f"SAVE_PATH: {f10.OUT_MODELS}/PLACEHOLDER_STEM",
        f"WANDB_RUN_NAME: {meta['prefix']}_placeholder",
        "WANDB_TAGS:",
        f"- suite:{f10.SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- role:transfer",
        "- ego:noblocking",
        "- ego:noweapon",
        "- ego:noclip",
        "- ablation:sparse_cda",
        f"- budget:{f10.BUDGET_TAG}",
        "- phase:from_scratch",
        "- skill_input:ego",
    ]
    lines += f10._mapper_tags(env)
    lines += [
        "- lr:constant",
        "- lock:skills_lr1e3",
        "GAME_MODS: null",
        "EVAL_GAME_MODS: null",
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
        "RECORD_VIDEO: false",
        "VIDEO_AT_END: false",
    ]
    lines += f10._sparse_top()
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
        f"  SKILL_LR: {f10.LOCKED_SKILL_LR}",
        "  SKILL_REWARD_LAMBDA: 0.0",
        "  CLIP_REWARD: true",
        "  RESTRICT_FIRE_FOR_NON_DESTROY: false",
        "  FIRST_FIRE: true",
        "  STICKY_ACTIONS: false",
        "  EPISODIC_LIFE: true",
        "  FRAME_STACK_SIZE: 4",
        "  FRAME_SKIP: 4",
        f"  TOTAL_TIMESTEPS: {f10.TIMESTEPS_10M}",
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
        f"  EPS_START: {f10.LOCKED_EPS_START}",
        f"  EPS_FINISH: {f10.LOCKED_EPS_FINISH}",
        f"  EPS_DECAY: {f10.LOCKED_EPS_DECAY}",
        f"  HEURISTIC_MODE: {meta['heu']}",
        "  SKILL_INPUT_MODE: ego",
        "  SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += f10._sparse_alg()
    lines += [
        "  USE_EGO_SHOT_FEATURES: false",
        "SEED: 44",
    ]
    return f10._lines(lines)


def _skills_eval_body(env: str) -> str:
    if env in f10.BASE_SKILL_ENVS:
        return f10._skills_eval(env)
    meta = TRANSFER_ENV_META[env]
    rq = int(meta["rq"])
    mod_keys = f10._paper_core_keys(env)
    ckpt = (
        f"{f10.OUT_MODELS}/PLACEHOLDER_STEM/{env}/"
        f"hierarchical_pqn_{env}_seed44.safetensors"
    )
    lines = [
        f"# Home mod eval for {env} skills.",
        f"PAPER_EVAL_RQ: {rq}",
        "PAPER_EVAL_PHASE: eval",
        "PAPER_EVAL_KIND: skills",
        f"PAPER_EVAL_MODS: {env}",
    ]
    lines += f10._mod_keys_yaml(mod_keys)
    lines += [
        "PAPER_EVAL_REQUIRES_RQ1:",
        "- PLACEHOLDER_STEM",
        f"PAPER_EVAL_SKILLS_CKPT: {ckpt}",
        "PAPER_EVAL_ARCHITECTURE_OF: PLACEHOLDER_STEM",
        "NUM_SEEDS: 1",
        "ALG_NAME: hierarchical_pqn",
        "TRAIN_MODE: skills_only",
        f"ENV_NAME: {env}",
    ]
    lines += f10._ego_lines(seaquest_legacy=False)
    lines += [
        f"SAVE_PATH: {f10.OUT_MODELS}/PLACEHOLDER_STEM",
        f"WANDB_RUN_NAME: {meta['prefix']}_placeholder_eval",
        "WANDB_TAGS:",
        f"- suite:{f10.SUITE}",
        f"- rq:{rq}",
        f"- env:{env}",
        "- phase:mod_eval",
        "- skill_input:ego",
    ]
    lines += f10._mapper_tags(env)
    lines += [
        f"- budget:{f10.BUDGET_TAG}",
        "- lock:skills_lr1e3",
        "SKILL_INPUT_MODE: ego",
        f"HEURISTIC_MODE: {meta['heu']}",
        "SKILL_DEATH_PENALIZE_COLLECT: true",
    ]
    lines += f10._sparse_top()
    lines += [
        "SEED: 44",
        "EVAL_SEED: 10044",
        "EVAL_NUM_SEEDS: 3",
    ]
    return f10._lines(lines)


def _base_env_train(env: str) -> str:
    stem = skills_base_stem(env)
    prefix = _env_meta(env)["prefix"]
    return _skills_hp_train(
        stem=stem,
        seed=BASE_SEED,
        suite=BASES_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# {env} ego skills — optim_scale LOCKED recipe "
            f"(ε B + LR={LR}+linear; {NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}); "
            f"seed={BASE_SEED}."
        ),
        run_name=f"{prefix}_{BASES_SUITE}_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB",
    )


def _base_env_eval(env: str) -> str:
    stem = skills_base_stem(env)
    prefix = _env_meta(env)["prefix"]
    return _skills_hp_eval(
        stem=stem,
        seed=BASE_SEED,
        suite=BASES_SUITE,
        hp=EPSB_HP,
        env=env,
        header=f"# Home mod eval — optim_scale locked skills {env} seed{BASE_SEED}.",
        run_name=f"{prefix}_{BASES_SUITE}_eval_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB",
    )


def _transfer_env_train(env: str) -> str:
    stem = skills_transfer_stem(env)
    prefix = _env_meta(env)["prefix"]
    return _skills_hp_train(
        stem=stem,
        seed=BASE_SEED,
        suite=TRANSFERS_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# {env} ego skills — optim_scale LOCKED recipe "
            f"(ε B + LR={LR}+linear; {NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES}); "
            f"seed={BASE_SEED}."
        ),
        run_name=f"{prefix}_{TRANSFERS_SUITE}_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB",
    )


def _transfer_env_eval(env: str) -> str:
    stem = skills_transfer_stem(env)
    prefix = _env_meta(env)["prefix"]
    return _skills_hp_eval(
        stem=stem,
        seed=BASE_SEED,
        suite=TRANSFERS_SUITE,
        hp=EPSB_HP,
        env=env,
        header=(
            f"# Home mod eval — optim_scale locked skills transfer "
            f"{env} seed{BASE_SEED}."
        ),
        run_name=f"{prefix}_{TRANSFERS_SUITE}_eval_{BUDGET_TAG}",
        scale_tag="pqn_optim_epsB",
    )


def _lr_train(arm: str, hp: dict) -> str:
    stem = skills_lr_stem(arm)
    decay = "linear" if hp["lr_linear_decay"] else "const"
    return _skills_hp_train(
        stem=stem,
        seed=PROBE_SEED,
        suite=LR_SUITE,
        hp=hp,
        header=(
            f"# Seaquest ego skills LR probe seed={PROBE_SEED} arm={arm}: "
            f"SKILL_LR={hp['skill_lr']} ({decay}); ε B; "
            f"outer 1:1 ({NUM_ENVS}×{NUM_STEPS}×{NUM_UPDATES})."
        ),
        run_name=f"sq_{LR_SUITE}_{arm}_seed{PROBE_SEED}_{BUDGET_TAG}",
        scale_tag=f"pqn_optim_lr_{arm}",
    )


def _lr_eval(arm: str, hp: dict) -> str:
    stem = skills_lr_stem(arm)
    return _skills_hp_eval(
        stem=stem,
        seed=PROBE_SEED,
        suite=LR_SUITE,
        hp=hp,
        header=f"# Home mod eval — optim_scale skills LR probe {arm} seed{PROBE_SEED}.",
        run_name=f"sq_{LR_SUITE}_eval_{arm}_seed{PROBE_SEED}_{BUDGET_TAG}",
        scale_tag=f"pqn_optim_lr_{arm}",
    )


def _old_eps_train() -> str:
    stem = skills_old_eps_stem()
    return _skills_hp_train(
        stem=stem,
        seed=OLD_EPS_SEED,
        suite=OLD_EPS_SUITE,
        hp=OLD_EPS_HP,
        header=(
            f"# Seaquest ego skills — single YAML-ε arm "
            f"(ε {EPS_START}→{EPS_FINISH}/d={EPS_DECAY}; LR={LR}+linear); "
            f"seed={OLD_EPS_SEED}."
        ),
        run_name=f"sq_{OLD_EPS_SUITE}_seed{OLD_EPS_SEED}_{BUDGET_TAG}",
        scale_tag="pqn_optim_old_eps",
    )


def _old_eps_eval() -> str:
    stem = skills_old_eps_stem()
    return _skills_hp_eval(
        stem=stem,
        seed=OLD_EPS_SEED,
        suite=OLD_EPS_SUITE,
        hp=OLD_EPS_HP,
        header=(
            f"# Home mod eval — optim_scale skills old-ε seed{OLD_EPS_SEED}."
        ),
        run_name=f"sq_{OLD_EPS_SUITE}_eval_seed{OLD_EPS_SEED}_{BUDGET_TAG}",
        scale_tag="pqn_optim_old_eps",
    )


def _manifest() -> str:
    return f10._lines(
        [
            "# optim_scale_sq inventory (not runnable).",
            f"suite: {SUITE}",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS}",
            f"total_timesteps_decay: {TIMESTEPS_DECAY}",
            f"num_envs: {NUM_ENVS}",
            f"num_steps: {NUM_STEPS}",
            f"num_updates: {NUM_UPDATES}",
            f"num_epochs: {NUM_EPOCHS}",
            f"num_minibatches: {NUM_MINIBATCHES}",
            f"lr: {LR}",
            "lr_linear_decay: true",
            f"max_grad_norm: {MAX_GRAD_NORM}",
            f"eps: {EPS_START} -> {EPS_FINISH} (decay={EPS_DECAY})",
            f"seeds: {list(SEEDS)}",
            "ref: pqn_jaxatari_pixel_optimal_scaling.yaml (1:1 shared HPs)",
            "skills: SKILL_LR + LR_LINEAR_DECAY + YAML ε",
            "tandem: META_LR=SKILL_LR + same decay + YAML ε on skill+meta",
            "note: suite order is tandem-first (OOM-fast); use optim_scale_probe for skills-first HP checks",
            f"outputs_root: {OUT_MODELS}",
            "stems:",
            *[f"  - {s}" for s in optim_scale_sq_stems()],
        ]
    )


def _probe_manifest() -> str:
    lines = [
        "# optim_scale_probe inventory (not runnable).",
        f"suite: {PROBE_SUITE}",
        f"budget_tag: {BUDGET_TAG}",
        f"seed: {PROBE_SEED}",
        "kind: skills_only",
        f"outer: {NUM_ENVS}x{NUM_STEPS}x{NUM_UPDATES} T={TIMESTEPS} max_grad={MAX_GRAD_NORM}",
        "arms:",
    ]
    for arm, hp in PROBE_ARMS:
        lines.append(
            f"  - {arm}: lr={hp['skill_lr']} decay={hp['lr_linear_decay']} "
            f"eps={hp['eps_start']}->{hp['eps_finish']}/{hp['eps_decay']}"
        )
    lines.append("stems:")
    lines.extend(f"  - {s}" for s in optim_scale_probe_stems())
    return f10._lines(lines)


def _skills_final_manifest() -> str:
    lines = [
        "# optim_scale skills final inventory (not runnable).",
        f"locked_recipe: ε B ({EPS_B_START}->{EPS_B_FINISH}/{EPS_B_DECAY}) "
        f"+ LR={LR}+linear + MAX_GRAD_NORM={MAX_GRAD_NORM}",
        f"bases_suite: {BASES_SUITE}",
        f"  seed: {BASE_SEED}",
        f"  envs: {list(BASE_OTHER_ENVS)}",
        "  stems:",
        *[f"    - {s}" for s in optim_scale_skills_bases_stems()],
        f"transfers_suite: {TRANSFERS_SUITE}",
        f"  seed: {BASE_SEED}",
        f"  envs: {list(TRANSFER_ENVS)}",
        "  stems:",
        *[f"    - {s}" for s in optim_scale_skills_transfers_stems()],
        f"clip_foils_suite: {CLIP_FOILS_SUITE}",
        f"  seed: {BASE_SEED}",
        f"  clips: {list(CLIP_FOIL_CLIPS)}",
        f"  envs: {list(CLIP_FOIL_ENVS)}",
        "  stems:",
        *[f"    - {s}" for s in optim_scale_skills_clip_foils_stems()],
        f"living_urgency_suite: {LIVING_URGENCY_SUITE}",
        f"  seed: {BASE_SEED}",
        "  flags: DESTROY_LIVING_PENALTY + COLLECT_LIVING_PENALTY "
        "(COLLECT_SPARSE_ONLY off; AVOID_SPARSE_ONLY on)",
        f"  envs: {list(LIVING_URGENCY_ENVS)}",
        "  stems:",
        *[f"    - {s}" for s in optim_scale_skills_living_urgency_stems()],
        f"lr_suite: {LR_SUITE}",
        f"  seed: {PROBE_SEED}",
        "  fixed_eps: B",
        "  arms:",
    ]
    for arm, hp in LR_ARMS:
        decay = "linear" if hp["lr_linear_decay"] else "const"
        lines.append(f"    - {arm}: lr={hp['skill_lr']} ({decay})")
    lines.append("  stems:")
    lines.extend(f"    - {s}" for s in optim_scale_skills_lr_stems())
    lines += [
        f"old_eps_suite: {OLD_EPS_SUITE}",
        f"  seed: {OLD_EPS_SEED}",
        f"  eps: {EPS_START}->{EPS_FINISH}/{EPS_DECAY}",
        f"  lr: {LR}+linear",
        "  stems:",
        *[f"    - {s}" for s in optim_scale_skills_old_eps_stems()],
    ]
    return f10._lines(lines)


def _pqn_manifest() -> str:
    return f10._lines(
        [
            "# optim_scale flat PQN inventory (not runnable).",
            f"suites: [{PQN_OC_SUITE}, {PQN_PIXEL_SUITE}, {PQN_PIXEL_ENVS_SUITE}, "
            f"{PQN_OC_BUILDS_SUITE}]",
            f"budget_tag: {BUDGET_TAG}",
            f"total_timesteps: {TIMESTEPS}",
            f"num_envs: {NUM_ENVS}",
            f"num_steps: {NUM_STEPS}",
            f"num_updates: {NUM_UPDATES}",
            f"num_epochs: {NUM_EPOCHS}",
            f"num_minibatches: {NUM_MINIBATCHES}",
            f"lr: {LR}",
            "lr_linear_decay: true",
            f"max_grad_norm: {MAX_GRAD_NORM}",
            f"eps: {EPS_START} -> {EPS_FINISH} (decay={EPS_DECAY})",
            f"seeds: {list(SEEDS)}",
            "order: train→eval per seed (sequential)",
            f"gpu0_oc_stems:",
            *[f"  - {s}" for s in optim_scale_pqn_oc_stems()],
            f"gpu1_pixel_stems:",
            *[f"  - {s}" for s in optim_scale_pqn_pixel_stems()],
            f"pixel_envs_suite: {PQN_PIXEL_ENVS_SUITE}",
            f"  envs: {list(PQN_PIXEL_ENVS)}",
            f"  seeds: {list(SEEDS)}",
            "  stems:",
            *[f"    - {s}" for s in optim_scale_pqn_pixel_envs_stems()],
            f"oc_builds_suite: {PQN_OC_BUILDS_SUITE}",
            f"  seed: {OC_BUILD_SEED}",
            f"  modes: {list(OC_BUILD_MODES)}",
            "  stems:",
            *[f"    - {s}" for s in optim_scale_pqn_oc_builds_stems()],
        ]
    )


def materialize(*, dry: bool) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"optim_scale_sq 1:1 → {_OUT.relative_to(_ROOT)}  "
        f"T={TIMESTEPS} envs={NUM_ENVS} steps={NUM_STEPS} updates={NUM_UPDATES} "
        f"LR={LR} max_grad={MAX_GRAD_NORM} seeds={SEEDS}"
    )
    pairs: list[tuple[Path, str]] = []
    for seed in SEEDS:
        pairs.append((_OUT / f"{skills_stem(seed)}.yaml", _skills_train(seed)))
        pairs.append((_OUT / f"{skills_stem(seed)}_eval.yaml", _skills_eval(seed)))
        pairs.append((_OUT / f"{tandem_stem(seed)}.yaml", _tandem_train(seed)))
        pairs.append((_OUT / f"{tandem_stem(seed)}_eval.yaml", _tandem_eval(seed)))
    pairs.append((_OUT / "manifest.yaml", _manifest()))

    print(
        f"optim_scale_probe → skills-only seed={PROBE_SEED} arms="
        f"{[a for a, _ in PROBE_ARMS]}"
    )
    for arm, hp in PROBE_ARMS:
        stem = probe_skills_stem(arm)
        pairs.append((_OUT / f"{stem}.yaml", _probe_skills_train(arm, hp)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _probe_skills_eval(arm, hp)))
    # εB multi-seed re-run (suite optim_scale_skills_rerun)
    epsb_hp = dict(PROBE_ARMS)["epsB"]
    print(
        f"optim_scale_skills_rerun → εB skills × seeds={list(SEEDS)}"
    )
    for seed in SEEDS:
        stem = probe_skills_stem("epsB", seed)
        pairs.append(
            (_OUT / f"{stem}.yaml", _probe_skills_train("epsB", epsb_hp, seed=seed))
        )
        pairs.append(
            (
                _OUT / f"{stem}_eval.yaml",
                _probe_skills_eval("epsB", epsb_hp, seed=seed),
            )
        )
    pairs.append((_OUT / "probe_manifest.yaml", _probe_manifest()))

    print(
        f"optim_scale_skills_bases → εB recipe × envs={BASE_OTHER_ENVS} "
        f"seed={BASE_SEED}"
    )
    for env in BASE_OTHER_ENVS:
        stem = skills_base_stem(env)
        pairs.append((_OUT / f"{stem}.yaml", _base_env_train(env)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _base_env_eval(env)))

    print(
        f"optim_scale_skills_transfers → εB recipe × envs={TRANSFER_ENVS} "
        f"seed={BASE_SEED}"
    )
    for env in TRANSFER_ENVS:
        stem = skills_transfer_stem(env)
        pairs.append((_OUT / f"{stem}.yaml", _transfer_env_train(env)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _transfer_env_eval(env)))

    print(
        f"optim_scale_skills_clip_foils → clips={CLIP_FOIL_CLIPS} "
        f"× envs={CLIP_FOIL_ENVS} seed={BASE_SEED}"
    )
    for clip in CLIP_FOIL_CLIPS:
        for env in CLIP_FOIL_ENVS:
            stem = skills_clip_foil_stem(env, clip)
            pairs.append((_OUT / f"{stem}.yaml", _clip_foil_train(env, clip)))
            pairs.append((_OUT / f"{stem}_eval.yaml", _clip_foil_eval(env, clip)))

    print(
        f"optim_scale_skills_living_urgency → envs={LIVING_URGENCY_ENVS} "
        f"seed={BASE_SEED} (COLLECT+DESTROY living tax)"
    )
    for env in LIVING_URGENCY_ENVS:
        stem = skills_living_urgency_stem(env)
        pairs.append((_OUT / f"{stem}.yaml", _living_urgency_train(env)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _living_urgency_eval(env)))

    print(
        f"optim_scale_skills_lr → seed={PROBE_SEED} arms="
        f"{[a for a, _ in LR_ARMS]}"
    )
    for arm, hp in LR_ARMS:
        stem = skills_lr_stem(arm)
        pairs.append((_OUT / f"{stem}.yaml", _lr_train(arm, hp)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _lr_eval(arm, hp)))

    print(
        f"optim_scale_skills_old_eps → single YAML-ε seed={OLD_EPS_SEED}"
    )
    old_stem = skills_old_eps_stem()
    pairs.append((_OUT / f"{old_stem}.yaml", _old_eps_train()))
    pairs.append((_OUT / f"{old_stem}_eval.yaml", _old_eps_eval()))
    pairs.append((_OUT / "skills_final_manifest.yaml", _skills_final_manifest()))

    print(
        f"optim_scale_pqn → pixel+OC × seeds={SEEDS} "
        f"(suites {PQN_OC_SUITE} / {PQN_PIXEL_SUITE})"
    )
    for variant in ("oc", "pixel"):
        for seed in SEEDS:
            stem = pqn_stem(variant, seed)
            pairs.append((_OUT / f"{stem}.yaml", _pqn_train(variant=variant, seed=seed)))
            pairs.append(
                (_OUT / f"{stem}_eval.yaml", _pqn_eval(variant=variant, seed=seed))
            )
    pairs.append((_OUT / "pqn_manifest.yaml", _pqn_manifest()))

    print(
        f"optim_scale_pqn_pixel_envs → envs={PQN_PIXEL_ENVS} × seeds={SEEDS} "
        f"(suite {PQN_PIXEL_ENVS_SUITE})"
    )
    for env in PQN_PIXEL_ENVS:
        for seed in SEEDS:
            stem = pqn_pixel_env_stem(env, seed)
            pairs.append((_OUT / f"{stem}.yaml", _pqn_pixel_env_train(env, seed)))
            pairs.append(
                (_OUT / f"{stem}_eval.yaml", _pqn_pixel_env_eval(env, seed))
            )

    print(
        f"optim_scale_pqn_oc_builds → modes={list(OC_BUILD_MODES)} "
        f"seed={OC_BUILD_SEED}"
    )
    for mode in OC_BUILD_MODES:
        stem = pqn_oc_build_stem(mode)
        pairs.append((_OUT / f"{stem}.yaml", _pqn_oc_build_train(mode)))
        pairs.append((_OUT / f"{stem}_eval.yaml", _pqn_oc_build_eval(mode)))

    for path, body in pairs:
        f10._write(path, body, dry=dry)

    ref_dst = _OUT / "pqn_jaxatari_pixel_optimal_scaling.yaml"
    if _REF_SRC.is_file() and not dry:
        shutil.copy2(_REF_SRC, ref_dst)
        print(f"copied ref → {ref_dst.relative_to(_ROOT)}")
    elif dry:
        print(f"would copy ref → {ref_dst.relative_to(_ROOT)}")

    print(
        "\nThesis skills (post-NOOP, locked εB):\n"
        f"  # already done: bases={list(BASE_OTHER_ENVS)} + Seaquest epsB probe\n"
        f"  # remaining transfer scratch envs + home evals\n"
        f"  python scripts/run_paper_eval.py --suite {TRANSFERS_SUITE} "
        f"--gpus 0,1 --wait-deps --skip-rq-gate --lean\n"
        f"  # distance-clip probe on Seaquest + foil envs (clip10/50)\n"
        f"  python scripts/run_paper_eval.py --suite {CLIP_FOILS_SUITE} "
        f"--gpus 0,1 --wait-deps --skip-rq-gate --lean\n"
        f"  # COLLECT+DESTROY living tax (discourage idle/spam) on same envs\n"
        f"  python scripts/run_paper_eval.py --suite {LIVING_URGENCY_SUITE} "
        f"--gpus 0,1,2 --wait-deps --skip-rq-gate --lean\n"
        f"  # matched flat PQN pixel on bases+transfers (3 seeds, same mods)\n"
        f"  python scripts/run_paper_eval.py --suite {PQN_PIXEL_ENVS_SUITE} "
        f"--gpus 0,1,2,3,4,5,6,7 --wait-deps --skip-rq-gate --lean\n"
        f"  # overnight free GPU: tandem OC feature builds as flat PQN\n"
        f"  python scripts/run_paper_eval.py --suite {PQN_OC_BUILDS_SUITE} "
        f"--gpus 0 --wait-deps --skip-rq-gate --lean\n"
        "Flat PQN Seaquest baselines (2 GPUs):\n"
        f"  CUDA_VISIBLE_DEVICES=0 python scripts/run_paper_eval.py "
        f"--suite {PQN_OC_SUITE} --gpus 0 --wait-deps --skip-rq-gate --lean &\n"
        f"  CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py "
        f"--suite {PQN_PIXEL_SUITE} --gpus 0 --wait-deps --skip-rq-gate --lean &\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    materialize(dry=args.dry_run)


if __name__ == "__main__":
    main()

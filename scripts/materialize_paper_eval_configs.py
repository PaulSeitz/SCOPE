#!/usr/bin/env python3
"""Materialize RQ2 / RQ3 YAML configs under configs/experiment/paper_eval."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiment" / "paper_eval"

# Thesis winner (clip50 + 8θ-bins ego on Seaquest).
RQ1_EGO_STEM = "seaquest_skills_ego_clip50_angbins"
RQ1_EGO = (
    "./outputs/models/paper_eval/rq1/seaquest_skills_ego_clip50_angbins/seaquest/"
    "hierarchical_pqn_seaquest_seed44.safetensors"
)
RQ1_PIXEL = "./outputs/models/paper_eval/rq1/seaquest_skills_pixel/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"
RQ1_OC = "./outputs/models/paper_eval/rq1/seaquest_skills_oc/seaquest/hierarchical_pqn_seaquest_seed44.safetensors"

# Matched budgets
STEPS_25M = 24_985_600
STEPS_5M = 4_997_120  # ~5M aligned to 128*128
BATCH = {"NUM_ENVS": 128, "NUM_STEPS": 128, "NUM_EPOCHS": 4, "NUM_MINIBATCHES": 64}

ENVS = {
    "pong": {
        "MAX_OBJECTS": 2,
        "FIRST_FIRE": True,
        "EPISODIC_LIFE": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "HEURISTIC_MODE": "nearest",
    },
    "tennis": {
        "MAX_OBJECTS": 2,
        "FIRST_FIRE": True,
        "EPISODIC_LIFE": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "HEURISTIC_MODE": "nearest",
    },
    "beamrider": {
        "MAX_OBJECTS": 31,
        "FIRST_FIRE": True,
        "EPISODIC_LIFE": True,
        "USE_EGO_WEAPON_FEATURES": True,
        "HEURISTIC_MODE": "env_specific",
        "NO_TARGET_GAME_ACTION": 1,
    },
    "freeway": {
        "MAX_OBJECTS": 12,
        "FIRST_FIRE": False,
        "EPISODIC_LIFE": False,
        "USE_EGO_WEAPON_FEATURES": False,
        "HEURISTIC_MODE": "env_specific",
    },
    "asterix": {
        "MAX_OBJECTS": 17,
        "FIRST_FIRE": False,
        "EPISODIC_LIFE": True,
        "USE_EGO_WEAPON_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": True,
        "HEURISTIC_MODE": "env_specific",
    },
    "phoenix": {
        "MAX_OBJECTS": 21,
        "FIRST_FIRE": True,
        "EPISODIC_LIFE": True,
        "USE_EGO_WEAPON_FEATURES": True,
        "HEURISTIC_MODE": "env_specific",
    },
}

SKILL_INPUTS_TRANSFER = ("ego", "pixel", "oc")
SKILL_INPUTS_FROM_SCRATCH = ("ego", "pixel")  # protocol: ego + pixel for now

# Winner ego observation + destroy shaping (Seaquest-trained transfer path).
WINNER_EGO_FLAGS = {
    "USE_EGO_SHOT_FEATURES": False,
    "EGO_DISTANCE_CLIP_MAX_PX": 50,
    "EGO_ANGLE_BINS": 8,
    "EGO_LEGAL_ACTION_MASK": False,
    "DESTROY_CONTACT_FAIL": True,
    "DESTROY_LIVING_PENALTY": True,
}

# Shared destroy shaping for all skill arms on Seaquest (fair RQ1 comparison).
DESTROY_SHAPING = {
    "DESTROY_CONTACT_FAIL": True,
    "DESTROY_LIVING_PENALTY": True,
}


def _dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path.relative_to(ROOT)}")


def _skill_mode_key(mode: str) -> str:
    return {"ego": "ego", "pixel": "pixel_embed", "oc": "oc_embed"}[mode]


def _rq1_ckpt(mode: str) -> str:
    return {"ego": RQ1_EGO, "pixel": RQ1_PIXEL, "oc": RQ1_OC}[mode]


def _rq1_stem(mode: str) -> str:
    if mode == "ego":
        return RQ1_EGO_STEM
    if mode == "pixel":
        return "seaquest_skills_pixel"
    return "seaquest_skills_oc"


def _base_skill_flags(env: str, skill_mode: str, *, transfer_from_seaquest: bool = False) -> dict:
    meta = ENVS[env]
    if transfer_from_seaquest:
        weapon = True
        blocking = True
    else:
        weapon = bool(meta.get("USE_EGO_WEAPON_FEATURES", False))
        blocking = bool(meta.get("USE_EGO_BLOCKING_FEATURES", True))
    flags = {
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": blocking,
        "USE_EGO_WEAPON_FEATURES": weapon,
        "EGO_DISTANCE_CLIP_MAX_PX": None,
        "GAME_MODS": None,
        "EVAL_GAME_MODS": None,
        "SKILL_INPUT_MODE": _skill_mode_key(skill_mode),
        "HEURISTIC_MODE": meta["HEURISTIC_MODE"],
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "RECORD_VIDEO": False,
        "VIDEO_AT_END": False,
        **DESTROY_SHAPING,
    }
    if skill_mode == "ego" and transfer_from_seaquest:
        flags.update(WINNER_EGO_FLAGS)
    if skill_mode == "pixel":
        flags.update(
            {
                "META_PIXEL_DO_RESIZE": True,
                "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
                "META_PIXEL_RESIZE_SHAPE": [84, 84],
                "META_PIXEL_GRAYSCALE": True,
            }
        )
    if "NO_TARGET_GAME_ACTION" in meta:
        flags["NO_TARGET_GAME_ACTION"] = meta["NO_TARGET_GAME_ACTION"]
    return flags


def _alg_skills(env: str, *, total_steps: int, skill_mode: str = "ego") -> dict:
    meta = ENVS[env]
    alg = {
        "NUM_SKILLS": 3,
        "SKILL_NAMES": ["DESTROY", "COLLECT", "AVOID"],
        "MAX_OBJECTS": meta["MAX_OBJECTS"],
        "EGO_ACTION_DIM": 18,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "SKILL_EMBED_DIM": 256,
        "SKILL_LR": 0.0005,
        "SKILL_LR_FINISH": 0.0001,
        "SKILL_LR_DECAY": 0.5,
        "SKILL_REWARD_LAMBDA": 0.0,
        "CLIP_REWARD": True,
        "RESTRICT_FIRE_FOR_NON_DESTROY": False,
        "FIRST_FIRE": meta["FIRST_FIRE"],
        "STICKY_ACTIONS": False,
        "EPISODIC_LIFE": meta["EPISODIC_LIFE"],
        "FRAME_STACK_SIZE": 4,
        "FRAME_SKIP": 4,
        "TOTAL_TIMESTEPS": total_steps,
        **BATCH,
        "GAMMA": 0.99,
        "SKILL_GAMMA": 0.99,
        "LAMBDA": 0.65,
        "SKILL_LAMBDA": 0.65,
        "LR": 0.0001,
        "MAX_GRAD_NORM": 0.8,
        "OFF_POLICY": True,
        "RELABEL_ALL_OBJECTS": False,
        "RELABEL_K_OBJECTS": None,
        "SKILL_BOOTSTRAP": "q_lambda",
        "EPS_START": 0.2,
        "EPS_FINISH": 0.01,
        "EPS_DECAY": 0.2,
        "HEURISTIC_MODE": meta["HEURISTIC_MODE"],
        "SKILL_INPUT_MODE": _skill_mode_key(skill_mode),
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "SKILL_PIXEL_CNN_CHANNELS": [16, 32],
        **DESTROY_SHAPING,
        "USE_EGO_SHOT_FEATURES": False,
    }
    if skill_mode == "ego":
        alg.update(
            {
                "EGO_DISTANCE_CLIP_MAX_PX": 50 if total_steps == STEPS_5M else None,
            }
        )
    return alg


def _rq2_tags(env: str, phase: str, skill_mode: str, *, extra: str | None = None) -> list[str]:
    tags = [
        "suite:paper_eval",
        "rq:2",
        f"env:{env}",
        f"phase:{phase}",
        f"skill_input:{_skill_mode_key(skill_mode)}",
        "probe:thesis",
    ]
    if extra:
        tags.append(extra)
    return tags


def materialize_rq2() -> None:
    rq2 = OUT / "rq2"
    for env in ENVS:
        for mode in SKILL_INPUTS_TRANSFER:
            rq1_stem = _rq1_stem(mode)
            stem = f"{env}_transfer_{mode}"
            data = {
                "PAPER_EVAL_RQ": 2,
                "PAPER_EVAL_PHASE": "eval",
                "PAPER_EVAL_KIND": "transfer",
                "PAPER_EVAL_MODS": env,
                "PAPER_EVAL_REQUIRES_RQ1": [rq1_stem],
                "PAPER_EVAL_SKILLS_CKPT": _rq1_ckpt(mode),
                "NUM_SEEDS": 1,
                "ALG_NAME": "hierarchical_pqn",
                "TRAIN_MODE": "skills_only",
                "ENV_NAME": env,
                "SAVE_PATH": f"./outputs/models/paper_eval/rq2/{stem}",
                "WANDB_RUN_NAME": f"{env[:2]}_paper_rq2_transfer_{mode}",
                "WANDB_TAGS": _rq2_tags(env, "zero_shot", mode),
                **_base_skill_flags(env, mode, transfer_from_seaquest=True),
                "alg": _alg_skills(env, total_steps=STEPS_5M, skill_mode=mode),
            }
            if mode == "ego":
                data["PAPER_EVAL_ARCHITECTURE_OF"] = RQ1_EGO_STEM
            data["alg"]["SKILL_INPUT_MODE"] = _skill_mode_key(mode)
            _dump(rq2 / f"{stem}.yaml", data)

            stem_ft = f"{env}_finetune_{mode}"
            data_ft = {
                "PAPER_EVAL_RQ": 2,
                "PAPER_EVAL_PHASE": "finetune",
                "PAPER_EVAL_KIND": "skills",
                "PAPER_EVAL_MODS": env,
                "PAPER_EVAL_REQUIRES_RQ1": [rq1_stem],
                "NUM_SEEDS": 1,
                "ALG_NAME": "hierarchical_pqn",
                "TRAIN_MODE": "skills_only",
                "ENV_NAME": env,
                "INIT_MODEL_PATH": _rq1_ckpt(mode),
                "SAVE_PATH": f"./outputs/models/paper_eval/rq2/{stem_ft}",
                "WANDB_RUN_NAME": f"{env[:2]}_paper_rq2_finetune_{mode}",
                "WANDB_TAGS": _rq2_tags(env, "finetune", mode),
                **_base_skill_flags(env, mode, transfer_from_seaquest=True),
                "alg": _alg_skills(env, total_steps=STEPS_5M, skill_mode=mode),
            }
            if mode == "ego":
                data_ft["PAPER_EVAL_ARCHITECTURE_OF"] = RQ1_EGO_STEM
            data_ft["alg"]["SKILL_INPUT_MODE"] = _skill_mode_key(mode)
            data_ft["alg"]["INIT_MODEL_PATH"] = _rq1_ckpt(mode)
            _dump(rq2 / f"{stem_ft}.yaml", data_ft)

        for mode in SKILL_INPUTS_FROM_SCRATCH:
            stem_fs = f"{env}_from_scratch_{mode}"
            data_fs = {
                "PAPER_EVAL_RQ": 2,
                "PAPER_EVAL_PHASE": "train",
                "PAPER_EVAL_KIND": "skills",
                "PAPER_EVAL_MODS": env,
                "PAPER_EVAL_REQUIRES_RQ1": [],
                "NUM_SEEDS": 1,
                "ALG_NAME": "hierarchical_pqn",
                "TRAIN_MODE": "skills_only",
                "ENV_NAME": env,
                "SAVE_PATH": f"./outputs/models/paper_eval/rq2/{stem_fs}",
                "WANDB_RUN_NAME": f"{env[:2]}_paper_rq2_scratch_{mode}",
                "WANDB_TAGS": _rq2_tags(env, "from_scratch", mode),
                **_base_skill_flags(env, mode),
                "alg": _alg_skills(env, total_steps=STEPS_25M, skill_mode=mode),
            }
            data_fs["alg"]["SKILL_INPUT_MODE"] = _skill_mode_key(mode)
            _dump(rq2 / f"{stem_fs}.yaml", data_fs)


def _seaquest_skill_alg_block(*, skill_input: str) -> dict:
    return {
        "NUM_SKILLS": 3,
        "SKILL_NAMES": ["DESTROY", "COLLECT", "AVOID"],
        "MAX_OBJECTS": 35,
        "EGO_ACTION_DIM": 18,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 3,
        "SKILL_EMBED_DIM": 256,
        "FRAME_SKIP": 4,
        "FRAME_STACK_SIZE": 4,
        "HEURISTIC_MODE": "env_specific",
        "SKILL_INPUT_MODE": skill_input,
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "DESTROY_CONTACT_FAIL": True,
        "DESTROY_LIVING_PENALTY": True,
        "USE_EGO_SHOT_FEATURES": False,
        "SKILL_PIXEL_CNN_CHANNELS": [16, 32],
    }


def _rq3_base(*, meta_mode: str, kind: str, train_mode: str, stem: str, short: str) -> dict:
    skill_lr = 0.0 if kind == "meta" else 0.0005
    off_policy = kind != "meta"
    data = {
        "PAPER_EVAL_RQ": 3,
        "PAPER_EVAL_PHASE": "train",
        "PAPER_EVAL_KIND": kind,
        "PAPER_EVAL_MODS": "seaquest",
        "PAPER_EVAL_REQUIRES_RQ1": [RQ1_EGO_STEM],
        "NUM_SEEDS": 1,
        "ALG_NAME": "hierarchical_pqn",
        "TRAIN_MODE": train_mode,
        "META_ARCHITECTURE": "per_object_mlp",
        "ENV_NAME": "seaquest",
        "INIT_MODEL_PATH": RQ1_EGO,
        "SAVE_PATH": f"./outputs/models/paper_eval/rq3/{stem}",
        "WANDB_RUN_NAME": f"sq_paper_rq3_{kind}_{short}_ego_clip50",
        "WANDB_TAGS": [
            "suite:paper_eval",
            "rq:3",
            "env:seaquest",
            f"train_mode:{train_mode}",
            f"meta_input:{meta_mode}",
            "skill_input:ego",
            "probe:thesis",
        ],
        "USE_LEGACY_EGO_OBS": False,
        "USE_EGO_SIZE_FEATURES": False,
        "USE_EGO_VELOCITY_FEATURES": False,
        "USE_EGO_BLOCKING_FEATURES": True,
        "USE_EGO_WEAPON_FEATURES": True,
        "USE_EGO_SHOT_FEATURES": False,
        "USE_LEGACY_SEAQUEST_ACTION_MAPPER": False,
        "USE_LEGACY_SEAQUEST_ASCENT_BLOCK": False,
        "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD": False,
        "GAME_MODS": None,
        "EVAL_GAME_MODS": None,
        "META_INPUT_MODE": meta_mode,
        "SKILL_INPUT_MODE": "ego",
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "META_PIXEL_DO_RESIZE": True,
        "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
        "META_PIXEL_RESIZE_SHAPE": [84, 84],
        "META_PIXEL_GRAYSCALE": True,
        "RECORD_VIDEO": False,
        **WINNER_EGO_FLAGS,
        "alg": {
            "INIT_MODEL_PATH": RQ1_EGO,
            "NUM_SKILLS": 3,
            "SKILL_NAMES": ["DESTROY", "COLLECT", "AVOID"],
            "MAX_OBJECTS": 35,
            "EGO_ACTION_DIM": 18,
            "HIDDEN_SIZE": 128,
            "NUM_LAYERS": 3,
            "META_HIDDEN_SIZE": 256,
            "META_PER_OBJECT_NUM_LAYERS": 3,
            "META_FLAT_NUM_LAYERS": 3,
            "META_EMBED_DIM": 256,
            "SKILL_EMBED_DIM": 256,
            "META_LR": 0.0005,
            "SKILL_LR": skill_lr,
            "META_USE_NOOP_TARGET": True,
            "META_NOOP_TARGET_IDX": None,
            "META_SWITCH_PENALTY": -0.05,
            "META_DEATH_PENALTY": 0.0,
            "META_REWARD_LAMBDA": 1.0,
            "CLIP_REWARD": True,
            "FIRST_FIRE": True,
            "STICKY_ACTIONS": False,
            "EPISODIC_LIFE": True,
            "FRAME_STACK_SIZE": 4,
            "FRAME_SKIP": 4,
            "TOTAL_TIMESTEPS": STEPS_25M,
            **BATCH,
            "GAMMA": 0.99,
            "LAMBDA": 0.65,
            "LR": 0.0001,
            "MAX_GRAD_NORM": 0.8,
            "OFF_POLICY": off_policy,
            "RELABEL_ALL_OBJECTS": False,
            "EPS_START": 0.1,
            "EPS_FINISH": 0.01,
            "EPS_DECAY": 0.2,
            "META_INPUT_MODE": meta_mode,
            "SKILL_INPUT_MODE": "ego",
            "META_PIXEL_CNN_CHANNELS": [16, 32],
            "SKILL_DEATH_PENALIZE_COLLECT": True,
            "DESTROY_CONTACT_FAIL": True,
            "DESTROY_LIVING_PENALTY": True,
            "USE_EGO_SHOT_FEATURES": False,
        },
    }
    if kind == "tandem":
        data["alg"].update(
            {
                "SKILL_LR": 0.0005,
                "SKILL_LR_FINISH": 0.0001,
                "SKILL_LR_DECAY": 0.5,
                "OFF_POLICY": True,
                "SKILL_BOOTSTRAP": "q_lambda",
                "SKILL_LAMBDA": 0.65,
                "SKILL_GAMMA": 0.99,
            }
        )
    return data


def materialize_rq3() -> None:
    rq3 = OUT / "rq3"
    for meta_mode, short in (("pixel_embed", "pixel"), ("oc_embed", "oc")):
        stem_meta = f"seaquest_meta_{short}_ego_skills_clip50"
        _dump(rq3 / f"{stem_meta}.yaml", _rq3_base(
            meta_mode=meta_mode, kind="meta", train_mode="meta_only",
            stem=stem_meta, short=short,
        ))

        stem_eval = f"{stem_meta}_eval"
        meta_train_stem = stem_meta
        _dump(
            rq3 / f"{stem_eval}.yaml",
            {
                "PAPER_EVAL_RQ": 3,
                "PAPER_EVAL_PHASE": "eval",
                "PAPER_EVAL_KIND": "meta",
                "PAPER_EVAL_MODS": "seaquest",
                "PAPER_EVAL_REQUIRES_RQ1": [meta_train_stem],
                "PAPER_EVAL_SKILLS_CKPT": (
                    f"./outputs/models/paper_eval/rq3/{meta_train_stem}/seaquest/"
                    "hierarchical_pqn_seaquest_seed44_skills.safetensors"
                ),
                "PAPER_EVAL_META_CKPT": (
                    f"./outputs/models/paper_eval/rq3/{meta_train_stem}/seaquest/"
                    "hierarchical_pqn_seaquest_seed44_meta.safetensors"
                ),
                "PAPER_EVAL_ARCHITECTURE_OF": meta_train_stem,
                "NUM_SEEDS": 1,
                "ALG_NAME": "hierarchical_pqn",
                "TRAIN_MODE": "meta_only",
                "ENV_NAME": "seaquest",
                "SAVE_PATH": f"./outputs/models/paper_eval/rq3/{meta_train_stem}",
                "WANDB_RUN_NAME": f"sq_paper_rq3_eval_meta_{short}_ego_clip50",
                "WANDB_TAGS": [
                    "suite:paper_eval",
                    "rq:3",
                    "env:seaquest",
                    "phase:mod_eval",
                    "train_mode:meta_only",
                    f"meta_input:{meta_mode}",
                    "skill_input:ego",
                    "probe:thesis",
                ],
                "SKILL_INPUT_MODE": "ego",
                "META_INPUT_MODE": meta_mode,
                "HEURISTIC_MODE": "env_specific",
                "SKILL_DEATH_PENALIZE_COLLECT": True,
                **WINNER_EGO_FLAGS,
            },
        )

        stem_t = f"seaquest_tandem_{short}_ego_clip50_angbins"
        _dump(
            rq3 / f"{stem_t}.yaml",
            _rq3_base(
                meta_mode=meta_mode,
                kind="tandem",
                train_mode="tandem",
                stem=stem_t,
                short=short,
            ),
        )
        _dump(
            rq3 / f"{stem_t}_eval.yaml",
            {
                "PAPER_EVAL_RQ": 3,
                "PAPER_EVAL_PHASE": "eval",
                "PAPER_EVAL_KIND": "tandem",
                "PAPER_EVAL_MODS": "seaquest",
                "PAPER_EVAL_REQUIRES_RQ1": [stem_t],
                "PAPER_EVAL_SKILLS_CKPT": (
                    f"./outputs/models/paper_eval/rq3/{stem_t}/seaquest/"
                    "hierarchical_pqn_seaquest_seed44_skills.safetensors"
                ),
                "PAPER_EVAL_META_CKPT": (
                    f"./outputs/models/paper_eval/rq3/{stem_t}/seaquest/"
                    "hierarchical_pqn_seaquest_seed44_meta.safetensors"
                ),
                "PAPER_EVAL_ARCHITECTURE_OF": stem_t,
                "NUM_SEEDS": 1,
                "ALG_NAME": "hierarchical_pqn",
                "TRAIN_MODE": "tandem",
                "ENV_NAME": "seaquest",
                "SAVE_PATH": f"./outputs/models/paper_eval/rq3/{stem_t}",
                "WANDB_RUN_NAME": f"sq_paper_rq3_eval_tandem_{short}_ego_clip50",
                "WANDB_TAGS": [
                    "suite:paper_eval",
                    "rq:3",
                    "env:seaquest",
                    "phase:mod_eval",
                    "train_mode:tandem",
                    f"meta_input:{meta_mode}",
                    "skill_input:ego",
                    "probe:thesis",
                ],
                "SKILL_INPUT_MODE": "ego",
                "META_INPUT_MODE": meta_mode,
                "HEURISTIC_MODE": "env_specific",
                "SKILL_DEATH_PENALIZE_COLLECT": True,
                **WINNER_EGO_FLAGS,
            },
        )


def materialize_rq1_eval_markers() -> None:
    """Eval companions for RQ1 skill / PQN arms."""
    rq1 = OUT / "rq1"
    for stem, ckpt_mode in [
        ("seaquest_skills_pixel", "pixel"),
        ("seaquest_skills_oc", "oc"),
    ]:
        path = rq1 / f"{stem}_eval.yaml"
        data = {
            "PAPER_EVAL_RQ": 1,
            "PAPER_EVAL_PHASE": "eval",
            "PAPER_EVAL_KIND": "skills",
            "PAPER_EVAL_MODS": "seaquest",
            "PAPER_EVAL_REQUIRES_RQ1": [stem],
            "PAPER_EVAL_SKILLS_CKPT": _rq1_ckpt(ckpt_mode),
            "PAPER_EVAL_ARCHITECTURE_OF": stem,
            "NUM_SEEDS": 1,
            "ALG_NAME": "hierarchical_pqn",
            "TRAIN_MODE": "skills_only",
            "ENV_NAME": "seaquest",
            "SAVE_PATH": f"./outputs/models/paper_eval/rq1/{stem}",
            "WANDB_RUN_NAME": f"sq_paper_rq1_eval_{ckpt_mode}",
            "WANDB_TAGS": [
                "suite:paper_eval",
                "rq:1",
                "env:seaquest",
                "phase:mod_eval",
                f"skill_input:{_skill_mode_key(ckpt_mode)}",
                "probe:thesis",
            ],
            "USE_LEGACY_EGO_OBS": False,
            "USE_EGO_SIZE_FEATURES": False,
            "USE_EGO_VELOCITY_FEATURES": False,
            "USE_EGO_BLOCKING_FEATURES": True,
            "USE_EGO_WEAPON_FEATURES": True,
            "EGO_DISTANCE_CLIP_MAX_PX": None,
            "USE_LEGACY_SEAQUEST_ACTION_MAPPER": False,
            "USE_LEGACY_SEAQUEST_ASCENT_BLOCK": False,
            "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD": False,
            "GAME_MODS": None,
            "SKILL_INPUT_MODE": _skill_mode_key(ckpt_mode),
            "HEURISTIC_MODE": "env_specific",
            "SKILL_DEATH_PENALIZE_COLLECT": True,
            **DESTROY_SHAPING,
            "alg": _seaquest_skill_alg_block(skill_input=_skill_mode_key(ckpt_mode)),
        }
        if ckpt_mode == "pixel":
            data.update(
                {
                    "META_PIXEL_DO_RESIZE": True,
                    "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
                    "META_PIXEL_RESIZE_SHAPE": [84, 84],
                    "META_PIXEL_GRAYSCALE": True,
                }
            )
        _dump(path, data)

    _dump(
        rq1 / "seaquest_pqn_eval.yaml",
        {
            "PAPER_EVAL_RQ": 1,
            "PAPER_EVAL_PHASE": "eval",
            "PAPER_EVAL_KIND": "flat_pqn",
            "PAPER_EVAL_MODS": "seaquest",
            "PAPER_EVAL_REQUIRES_RQ1": ["seaquest_pqn"],
            "PAPER_EVAL_FLAT_PQN_CKPT": (
                "./outputs/models/paper_eval/rq1/seaquest_pqn/seaquest/"
                "flat_pqn_seaquest_pixel_seed44_vmap0.safetensors"
            ),
            "PAPER_EVAL_OBJECT_CENTRIC": False,
            "NUM_SEEDS": 1,
            "ALG_NAME": "flat_pqn",
            "ENV_NAME": "seaquest",
            "OBJECT_CENTRIC": False,
            "SAVE_PATH": "./outputs/models/paper_eval/rq1/seaquest_pqn",
            "WANDB_RUN_NAME": "sq_paper_rq1_pqn_eval",
            "WANDB_TAGS": [
                "suite:paper_eval",
                "rq:1",
                "env:seaquest",
                "phase:mod_eval",
                "train_mode:flat_pqn",
                "probe:thesis",
            ],
        },
    )


def main() -> None:
    for p in (OUT / "rq2" / "_placeholder.yaml", OUT / "rq3" / "_placeholder.yaml"):
        if p.exists():
            p.unlink()
            print(f"removed {p.relative_to(ROOT)}")
    materialize_rq2()
    materialize_rq3()
    materialize_rq1_eval_markers()
    print("done")


if __name__ == "__main__":
    main()

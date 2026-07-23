#!/usr/bin/env python3
"""Write Hydra experiment YAMLs for input-mode and Q-stickiness ablation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "configs" / "experiment" / "input_ablation"
QSTICK_DIR = ROOT / "configs" / "experiment" / "q_stickiness_eval"

COMMON_ENV = {
    "ALG_NAME": "hierarchical_pqn",
    "TRAIN_MODE": "tandem",
    "META_ARCHITECTURE": "per_object_mlp",
    "USE_LEGACY_EGO_OBS": False,
    "USE_EGO_SIZE_FEATURES": False,
    "USE_EGO_VELOCITY_FEATURES": True,
    "EGO_DISTANCE_CLIP_MAX_PX": 20,
    "GAME_MODS": None,
    "EVAL_GAME_MODS": None,
}

SEAQUEST_ENV = {
    **COMMON_ENV,
    "ENV_NAME": "seaquest",
    "USE_LEGACY_SEAQUEST_ACTION_MAPPER": False,
    "USE_LEGACY_SEAQUEST_ASCENT_BLOCK": False,
    "USE_LEGACY_SEAQUEST_SURFACING_DEATH_GUARD": False,
}

SPACEINVADERS_ENV = {
    **COMMON_ENV,
    "ENV_NAME": "spaceinvaders",
    "USE_EGO_WEAPON_FEATURES": True,
}

FREEWAY_ENV = {
    **COMMON_ENV,
    "ENV_NAME": "freeway",
    "EGO_DISTANCE_CLIP_MAX_PX": 40,
}

BEAMRIDER_ENV = {
    **COMMON_ENV,
    "ENV_NAME": "beamrider",
    "USE_EGO_WEAPON_FEATURES": True,
}

ABLATION_ENVS = {
    "seaquest": {
        "env": SEAQUEST_ENV,
        "alg": {"MAX_OBJECTS": 35, "EGO_ACTION_DIM": 18},
        "prefix": "sq",
    },
    "spaceinvaders": {
        "env": SPACEINVADERS_ENV,
        "alg": {"MAX_OBJECTS": 41, "EGO_ACTION_DIM": 18},
        "prefix": "si",
    },
    "freeway": {
        "env": FREEWAY_ENV,
        "alg": {
            "MAX_OBJECTS": 12,
            "EGO_ACTION_DIM": 18,
            "FIRST_FIRE": False,
            "EPISODIC_LIFE": False,
            "META_DEATH_PENALTY": -1.0,
            "META_EPS_START": 0.8,
        },
        "prefix": "fw",
    },
    "beamrider": {
        "env": BEAMRIDER_ENV,
        "alg": {
            "MAX_OBJECTS": 31,
            "EGO_ACTION_DIM": 18,
            "FIRST_FIRE": True,
            "EPISODIC_LIFE": True,
            "META_DEATH_PENALTY": -1.0,
            "META_USE_FIRE_TARGET": True,
            "META_FIRE_TARGET_IDX": 1,
        },
        "prefix": "br",
    },
}

INPUT_ALG = {
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
    "SKILL_LR": 0.0005,
    "META_USE_NOOP_TARGET": True,
    "META_NOOP_TARGET_IDX": 0,
    "META_SWITCH_PENALTY": -0.05,
    "META_DEATH_PENALTY": -10.0,
    "META_REWARD_LAMBDA": 1.0,
    "SKILL_REWARD_LAMBDA": 0.0,
    "META_Q_STICKINESS_ENABLED": False,
    "RESTRICT_FIRE_FOR_NON_DESTROY": False,
    "FIRST_FIRE": True,
    "STICKY_ACTIONS": False,
    "EPISODIC_LIFE": True,
    "FRAME_STACK_SIZE": 4,
    "TOTAL_TIMESTEPS": 25_000_000,
    "NUM_ENVS": 512,
    "NUM_STEPS": 64,
    "NUM_EPOCHS": 2,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "SKILL_GAMMA": 0.99,
    "LAMBDA": 0.65,
    "SKILL_LAMBDA": 0.65,
    "LR": 0.0001,
    "MAX_GRAD_NORM": 0.8,
    "OFF_POLICY": True,
    "RELABEL_ALL_OBJECTS": True,
    "RELABEL_K_OBJECTS": 15,
    "SKILL_BOOTSTRAP_CHUNK_STEPS": 16,
    "SKILL_BOOTSTRAP": "q_lambda",
    "EPS_START": 0.3,
    "EPS_FINISH": 0.05,
    "EPS_DECAY": 0.35,
    "META_EPS_START": 0.1,
    "META_NUM_EPOCHS": 2,
    "SKILL_WARMUP_UPDATES": 0,
    "META_LEARN_DURING_WARMUP": False,
    "HEURISTIC_MODE": "env_specific",
}


def _align_total_timesteps(total: int, num_envs: int, num_steps: int) -> int:
    batch = int(num_envs) * int(num_steps)
    if batch <= 0:
        return int(total)
    return (int(total) // batch) * batch


INPUT_VARIANTS = [
    ("flat", "ego", "baseline"),
    ("flat", "oc_embed", None),
    ("flat", "pixel_embed", None),
    ("oc_embed", "ego", None),
    ("oc_embed", "oc_embed", None),
    ("oc_embed", "pixel_embed", None),
    ("pixel_embed", "ego", None),
    ("pixel_embed", "oc_embed", None),
    ("pixel_embed", "pixel_embed", None),
]

QSTICK_ALG = {
    **INPUT_ALG,
    "TOTAL_TIMESTEPS": 30_000_000,
    "LR": 0.00001,
    "SKILL_LR": 0.00001,
    "SKILL_WARMUP_UPDATES": 200,
    "SKILL_EPS_START": 0.0,
    "SKILL_EPS_FINISH": 0.0,
    "SKILL_EPS_DECAY": 0.5,
    "EPS_DECAY": 0.5,
    "INIT_MODEL_PATH": (
        "./outputs/models/q_stickiness_eval/phase1_skills/seaquest/"
        "hierarchical_pqn_seaquest_seed44_skills.safetensors"
    ),
}

QSTICK_VARIANTS = [
    {
        "key": "baseline_no_qsticky",
        "run_name": "sq_qstk_baseline",
        "tags": [
            "suite:q_stickiness_eval",
            "phase:2",
            "env:seaquest",
            "train_mode:tandem",
            "q_sticky:off",
            "switch_penalty:off",
            "variant:baseline",
        ],
        "overrides": {
            "META_Q_STICKINESS_ENABLED": False,
            "META_SWITCH_PENALTY": 0.0,
        },
    },
    {
        "key": "qsticky_t0",
        "run_name": "sq_qstk_t0",
        "tags": [
            "suite:q_stickiness_eval",
            "phase:2",
            "env:seaquest",
            "train_mode:tandem",
            "q_sticky:on",
            "q_thresh:0",
            "switch_penalty:off",
            "variant:qsticky_t0",
        ],
        "overrides": {
            "META_Q_STICKINESS_ENABLED": True,
            "META_Q_STICKINESS_THRESHOLD": 0.0,
            "META_SWITCH_PENALTY": 0.0,
        },
    },
    {
        "key": "qsticky_t025",
        "run_name": "sq_qstk_t025",
        "tags": [
            "suite:q_stickiness_eval",
            "phase:2",
            "env:seaquest",
            "train_mode:tandem",
            "q_sticky:on",
            "q_thresh:0.25",
            "switch_penalty:off",
            "variant:qsticky_t025",
        ],
        "overrides": {
            "META_Q_STICKINESS_ENABLED": True,
            "META_Q_STICKINESS_THRESHOLD": 0.25,
            "META_SWITCH_PENALTY": 0.0,
        },
    },
    {
        "key": "qsticky_t0_switch_penalty",
        "run_name": "sq_qstk_t0_swpen",
        "tags": [
            "suite:q_stickiness_eval",
            "phase:2",
            "env:seaquest",
            "train_mode:tandem",
            "q_sticky:on",
            "q_thresh:0",
            "switch_penalty:on",
            "variant:qsticky_t0_switch_penalty",
        ],
        "overrides": {
            "META_Q_STICKINESS_ENABLED": True,
            "META_Q_STICKINESS_THRESHOLD": 0.0,
            "META_SWITCH_PENALTY": -0.05,
        },
    },
    {
        "key": "qsticky_t_neg1",
        "run_name": "sq_qstk_tneg1",
        "tags": [
            "suite:q_stickiness_eval",
            "phase:2",
            "env:seaquest",
            "train_mode:tandem",
            "q_sticky:on",
            "q_thresh:-1",
            "switch_penalty:off",
            "variant:qsticky_t_neg1",
        ],
        "overrides": {
            "META_Q_STICKINESS_ENABLED": True,
            "META_Q_STICKINESS_THRESHOLD": -1.0,
            "META_SWITCH_PENALTY": 0.0,
        },
    },
]


def _input_variant_doc(
    env_name: str,
    meta: str,
    skill: str,
    label: str | None,
) -> str:
    title = f"meta={meta}, skill={skill}"
    if label:
        title += f" ({label})"
    return (
        f"# input_ablation/{env_name}_mi_{meta}_si_{skill}.yaml\n"
        f"# {env_name.title()} tandem input ablation: {title}.\n"
        "# Full scratch tandem (25M steps, 512x64 rollout, relabel_k=15). "
        "Filter: suite:input_ablation.\n"
    )


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def _pixel_embed_wrapper_settings(meta: str, skill: str) -> dict[str, Any]:
    if meta != "pixel_embed" and skill != "pixel_embed":
        return {}
    return {
        "META_PIXEL_DO_RESIZE": True,
        "META_PIXEL_USE_NATIVE_DOWNSCALING": True,
        "META_PIXEL_RESIZE_SHAPE": [84, 84],
        "META_PIXEL_GRAYSCALE": True,
    }


def _input_alg_for_variant(meta: str, skill: str, profile: dict) -> dict:
    alg = {**INPUT_ALG, **profile["alg"]}
    if meta == "pixel_embed" or skill == "pixel_embed":
        small_cnn = [16, 32]
        alg["META_PIXEL_CNN_CHANNELS"] = small_cnn
        alg["SKILL_PIXEL_CNN_CHANNELS"] = small_cnn
    return alg


def materialize_input_ablation() -> None:
    for env_name, profile in ABLATION_ENVS.items():
        env_prefix = profile["prefix"]
        for meta, skill, label in INPUT_VARIANTS:
            key = f"mi_{meta}_si_{skill}"
            tags = [
                "suite:input_ablation",
                f"env:{env_name}",
                "train_mode:tandem",
                f"meta_input:{meta}",
                f"skill_input:{skill}",
            ]
            if label:
                tags.append(f"variant:{label}")
            else:
                tags.append(f"variant:{key}")

            alg = {
                **_input_alg_for_variant(meta, skill, profile),
                "META_INPUT_MODE": meta,
                "SKILL_INPUT_MODE": skill,
            }
            alg["TOTAL_TIMESTEPS"] = _align_total_timesteps(
                int(INPUT_ALG["TOTAL_TIMESTEPS"]),
                int(alg["NUM_ENVS"]),
                int(alg["NUM_STEPS"]),
            )
            payload = {
                **profile["env"],
                **_pixel_embed_wrapper_settings(meta, skill),
                "SAVE_PATH": f"./outputs/models/input_ablation/{env_name}_{key}",
                "WANDB_RUN_NAME": f"{env_prefix}_abl_{key}",
                "WANDB_TAGS": tags,
                "META_INPUT_MODE": meta,
                "SKILL_INPUT_MODE": skill,
                "alg": alg,
            }
            path = INPUT_DIR / f"{env_name}_{key}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            header = _input_variant_doc(env_name, meta, skill, label)
            with path.open("w", encoding="utf-8") as f:
                f.write(header)
                yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def materialize_q_stickiness() -> None:
    phase1 = {
        **SEAQUEST_ENV,
        "TRAIN_MODE": "skills_only",
        "SAVE_PATH": "./outputs/models/q_stickiness_eval/phase1_skills",
        "WANDB_RUN_NAME": "sq_qstk_phase1_skills",
        "WANDB_TAGS": [
            "suite:q_stickiness_eval",
            "phase:1",
            "env:seaquest",
            "train_mode:skills_only",
            "skill_input:ego",
            "variant:phase1_skills",
        ],
        "SKILL_INPUT_MODE": "ego",
        "alg": {
            k: v
            for k, v in INPUT_ALG.items()
            if k
            not in {
                "META_LR",
                "META_USE_NOOP_TARGET",
                "META_NOOP_TARGET_IDX",
                "META_SWITCH_PENALTY",
                "META_DEATH_PENALTY",
                "META_REWARD_LAMBDA",
                "META_Q_STICKINESS_ENABLED",
                "META_NUM_EPOCHS",
                "META_LEARN_DURING_WARMUP",
                "HEURISTIC_MODE",
            }
        }
        | {
            "TOTAL_TIMESTEPS": 25_000_000,
            "NUM_ENVS": 512,
            "NUM_STEPS": 32,
            "LR": 0.0005,
            "SKILL_LR": 0.0005,
        },
    }
    _write_yaml(
        QSTICK_DIR / "phase1_skills.yaml",
        phase1,
    )

    for variant in QSTICK_VARIANTS:
        alg = {
            **QSTICK_ALG,
            "META_INPUT_MODE": "flat",
            "SKILL_INPUT_MODE": "ego",
            **variant["overrides"],
        }
        payload = {
            **SEAQUEST_ENV,
            "SAVE_PATH": f"./outputs/models/q_stickiness_eval/{variant['key']}",
            "WANDB_RUN_NAME": variant["run_name"],
            "WANDB_TAGS": variant["tags"],
            "META_INPUT_MODE": "flat",
            "SKILL_INPUT_MODE": "ego",
            **{k: v for k, v in variant["overrides"].items() if k.startswith("META_")},
            "alg": alg,
        }
        path = QSTICK_DIR / f"seaquest_{variant['key']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# q_stickiness_eval/seaquest_{variant['key']}.yaml\n"
            f"# Requires phase1 skills at alg.INIT_MODEL_PATH.\n"
        )
        with path.open("w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    materialize_input_ablation()
    materialize_q_stickiness()
    print(f"Wrote input ablation configs to {INPUT_DIR}")
    print(f"Wrote Q-stickiness configs to {QSTICK_DIR}")


if __name__ == "__main__":
    main()

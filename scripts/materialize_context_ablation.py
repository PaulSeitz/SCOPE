#!/usr/bin/env python3
"""Skill-context ablations of the thesis reference (Seaquest, Name This Game).

Two arms, each derived from the reference ``tf_{env}_skills_ego_seed{s}`` YAMLs
(which match the frozen checkpoints' saved configs). Only ``SKILL_INPUT_MODE``,
save paths, W&B names/tags and eval wiring change, so the heuristic selector,
skill rewards, ego action mapping, 3x128 MLP and PQN optimisation are the
reference ones.

``skills_ego_scenerel`` (SKILL_INPUT_MODE: ego_scene_rel)
    Reference target features + every object slot in the same player-relative
    facing-frame encoding (four frames, per-frame active, is_target). No
    absolute coordinates.
``skills_ego_abspos`` (SKILL_INPUT_MODE: ego_abs_pos)
    Reference input + normalized player centre (x/160, y/210) for each of the
    four stacked frames.
``skills_ego_scenerows`` (SKILL_INPUT_MODE: ego_scene_rows)
    The scene rows only, without the reference prefix.
``skills_ego_ocnative`` (SKILL_INPUT_MODE: ego_oc_native, OC_NATIVE_STACK: true)
    Reference input + the complete oc_native skill input (four-frame native
    vector and target one-hot). Tests whether absolute object coordinates and
    the extra fields still harm layout robustness once the target-relative
    input is present.

Home mod eval only (scene width depends on the game's slot count, so the scene
arm cannot transfer; the abs-pos arm is kept home-only for a matched pair).
Name This Game evals all seven keys present in the frozen reference JSONs.

Usage::

    python scripts/materialize_context_ablation.py            # write YAMLs
    python scripts/materialize_context_ablation.py --check    # dry run
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "configs" / "experiment" / "paper_eval" / "thesis_final"
OUT_DIR = ROOT / "configs" / "experiment" / "paper_eval" / "context_ablation"
SEEDS = (44, 45, 46, 47, 48)
ENVS = ("seaquest", "namethisgame")

ARMS = {
    "skills_ego_scenerel": (
        "ego_scene_rel",
        "Context ablation: reference ego target features plus all object slots "
        "in the same player-relative facing-frame encoding (4 frames, active, "
        "is_target). No absolute coordinates.",
    ),
    "skills_ego_abspos": (
        "ego_abs_pos",
        "Context ablation: reference ego input plus the normalized player "
        "centre (x/160, y/210) for each of the 4 stacked frames.",
    ),
    "skills_ego_recheck": (
        "ego",
        "Control: the reference retrained unchanged on the current code, to check the "
        "comparison with the frozen reference across code dates.",
    ),
    "skills_ego_scenerows": (
        "ego_scene_rows",
        "Context ablation: the relative scene rows of skills_ego_scenerel without the "
        "reference prefix. The target is identified by is_target only.",
    ),
    "skills_ego_ocnative": (
        "ego_oc_native",
        "Context ablation: reference ego input plus the complete oc_native skill "
        "input (native vector and target one-hot), including absolute coordinates.",
    ),
    "skills_ocnative_targetfront": (
        "oc_native_target_front",
        "Context ablation: the complete oc_native skill input behind the absolute target "
        "row in fixed input slots. No target-relative information.",
    ),
}
# Modes that read the aligned native OC vector.
NATIVE_STACK_MODES = {"ego_oc_native", "oc_native_target_front"}

# Frozen reference JSON key order (NTG includes the later-spliced mods).
EVAL_MOD_KEYS = {
    "seaquest": [
        "base", "no_enemies", "swap_diver_enemy_labels", "swap_diver_shark_roles",
        "gravity", "faster_enemies", "shift_lanes", "random_spawns",
        "change_background_color",
    ],
    "namethisgame": [
        "base", "shift_tentacle_spawn_x", "slower_player", "faster_shark",
        "no_shark_short_tentacles", "change_background_color", "no_shark",
    ],
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _retag(tags: list[str], arm: str, mode: str) -> list[str]:
    out = []
    for t in tags:
        if t == "rq:1":
            t = "rq:1abl"
        elif t.startswith("arm:"):
            t = f"arm:{arm}"
        elif t.startswith("skill_input:"):
            t = f"skill_input:{mode}"
        out.append(t)
    if not any(t.startswith("skill_input:") for t in out):
        out.append(f"skill_input:{mode}")
    out.append("ablation:skill_context")
    return out


def _set_mode(doc: dict, mode: str) -> None:
    doc["SKILL_INPUT_MODE"] = mode
    doc["alg"]["SKILL_INPUT_MODE"] = mode
    if mode in NATIVE_STACK_MODES:
        doc["OC_NATIVE_STACK"] = True


def build(env: str, seed: int, arm: str) -> dict[str, tuple[str, dict]]:
    mode, note = ARMS[arm]
    ref_stem = f"tf_{env}_skills_ego_seed{seed}"
    stem = f"tf_{env}_{arm}_seed{seed}"
    ref_train = _load(REF_DIR / f"{ref_stem}.yaml")
    ref_eval = _load(REF_DIR / f"{ref_stem}_eval.yaml")
    assert ref_train["SKILL_INPUT_MODE"] == "ego", ref_stem
    assert ref_train["alg"]["OFF_POLICY"] is True, ref_stem

    train = copy.deepcopy(ref_train)
    _set_mode(train, mode)
    train["SAVE_PATH"] = f"./outputs/models/paper_eval/thesis_final/{stem}"
    train["WANDB_RUN_NAME"] = ref_train["WANDB_RUN_NAME"].replace(ref_stem, stem)
    train["WANDB_TAGS"] = _retag(ref_train["WANDB_TAGS"], arm, mode)

    ev = copy.deepcopy(ref_eval)
    _set_mode(ev, mode)
    ev["PAPER_EVAL_MOD_KEYS"] = list(EVAL_MOD_KEYS[env])
    ev["PAPER_EVAL_REQUIRES_RQ1"] = [stem]
    ev["PAPER_EVAL_ARCHITECTURE_OF"] = stem
    ev["PAPER_EVAL_SKILLS_CKPT"] = ref_eval["PAPER_EVAL_SKILLS_CKPT"].replace(
        f"/{ref_stem}/", f"/{stem}/"
    )
    ev["SAVE_PATH"] = f"./outputs/models/paper_eval/thesis_final/{stem}_eval"
    ev["WANDB_RUN_NAME"] = ref_eval["WANDB_RUN_NAME"].replace(ref_stem, stem)
    ev["WANDB_TAGS"] = _retag(ref_eval["WANDB_TAGS"], arm, mode)
    if mode == "ego_scene_rel" and env == "seaquest":
        # NOTE: no effect on the 2026-11-04 runs; the harness still used 9 parallel
        # workers at 0.078. Those OOMed on a card with ~26 GB free, passed with ~45 GB.
        ev["EVAL_PARALLEL_MODS"] = False
    assert f"/{stem}/" in ev["PAPER_EVAL_SKILLS_CKPT"]

    header = (
        f"# {note}\n# Derived from {ref_stem} by "
        "scripts/materialize_context_ablation.py; do not edit by hand.\n"
    )
    return {
        stem: (header, train),
        f"{stem}_eval": (header, ev),
    }


FLAT_DIR = ROOT / "configs" / "experiment" / "paper_eval" / "rewardfix_wave"
FLAT_ARM = "pqn_ego_scenerel"
# Reference settings that define the input encoding and the action mapping.
_ENCODING_PREFIXES = ("USE_EGO", "USE_POV", "EGO_", "USE_LEGACY", "SEAQUEST_")


def build_flat(env: str, seed: int) -> dict[str, tuple[str, dict]]:
    """Flat PQN on the relative-scene rows: same encoding and actions, no hierarchy."""
    src = f"tf_{env}_pqn_ocnative_seed{seed}"
    stem = f"tf_{env}_{FLAT_ARM}_seed{seed}"
    ref = _load(REF_DIR / f"tf_{env}_skills_ego_seed{seed}.yaml")
    ref_flat = {**ref, **ref["alg"]}
    train = _load(FLAT_DIR / f"{src}.yaml")
    ev = _load(FLAT_DIR / f"{src}_eval.yaml")
    assert train["PAPER_EVAL_KIND"] == "flat_pqn" and ev["PAPER_EVAL_KIND"] == "flat_pqn", src
    # Matches the 3x128 MLP of the skill arm; the update settings already agree.
    for key in ("LR", "LAMBDA", "EPS_DECAY", "NUM_ENVS", "NUM_STEPS", "NUM_EPOCHS", "GAMMA",
                "NUM_MINIBATCHES", "TOTAL_TIMESTEPS", "MAX_GRAD_NORM", "EPS_FINISH"):
        assert train[key] == ref_flat[key], (key, train[key], ref_flat[key])
    save = f"./outputs/models/paper_eval/thesis_final/{stem}"
    enc = {k: v for k, v in ref_flat.items() if k.startswith(_ENCODING_PREFIXES)}
    common = {**enc, "OC_INPUT_MODE": "ego_scene_rel_flat", "NUM_LAYERS": 3,
              "HIDDEN_SIZE": 128, "MAX_OBJECTS": ref_flat["MAX_OBJECTS"]}
    train.update(common)
    train["SAVE_PATH"] = save
    train["WANDB_RUN_NAME"] = train["WANDB_RUN_NAME"].replace(src, stem)
    train["WANDB_TAGS"] = _retag(train["WANDB_TAGS"], FLAT_ARM, "ego_scene_rel_flat")
    ev.update(common)
    ev["PAPER_EVAL_MOD_KEYS"] = list(EVAL_MOD_KEYS[env])
    ev["PAPER_EVAL_REQUIRES_RQ1"] = [stem]
    ev["PAPER_EVAL_ARCHITECTURE_OF"] = stem
    ev["PAPER_EVAL_FLAT_PQN_CKPT"] = f"{save}/{env}/flat_pqn_{env}_oc_scene_rel_seed{seed}_vmap0.safetensors"
    ev["SAVE_PATH"] = f"{save}_eval"
    ev["WANDB_RUN_NAME"] = ev["WANDB_RUN_NAME"].replace(src, stem)
    ev["WANDB_TAGS"] = _retag(ev["WANDB_TAGS"], FLAT_ARM, "ego_scene_rel_flat")
    header = (
        f"# Flat PQN on the relative-scene rows, derived from {src} and the reference "
        "encoding\n# by scripts/materialize_context_ablation.py; do not edit by hand.\n"
    )
    return {stem: (header, train), f"{stem}_eval": (header, ev)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="dry run, write nothing")
    ap.add_argument("--arms", nargs="+", choices=sorted([*ARMS, FLAT_ARM]),
                    default=sorted([*ARMS, FLAT_ARM]),
                    help="limit writing to these arms")
    args = ap.parse_args()
    docs: dict[str, tuple[str, dict]] = {}
    for env in ENVS:
        for arm in args.arms:
            for seed in SEEDS:
                docs.update(build_flat(env, seed) if arm == FLAT_ARM else build(env, seed, arm))
    if not args.check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, (header, doc) in sorted(docs.items()):
        text = header + yaml.safe_dump(doc, sort_keys=False)
        if not args.check:
            (OUT_DIR / f"{stem}.yaml").write_text(text)
        print(("would write " if args.check else "wrote ") + f"{stem}.yaml")
    print(f"{len(docs)} configs -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

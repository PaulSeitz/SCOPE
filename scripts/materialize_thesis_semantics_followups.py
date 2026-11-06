#!/usr/bin/env python3
"""Materialize the post-audit Seaquest reward and screen-frame follow-ups.

This script intentionally lives outside ``materialize_thesis_final.py`` so
refreshing the locked thesis suite cannot delete or silently rewrite these
diagnostic runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiment" / "paper_eval" / "thesis_final"
MANIFEST = OUT / "_thesis_semantics_followups_manifest.yaml"
NOTES = ROOT / "docs" / "thesis_semantics_followup_runs.md"
SEEDS = (44, 45, 46)
TRANSFER_TARGETS = (
    "asteroids",
    "beamrider",
    "breakout",
    "freeway",
    "namethisgame",
    "phoenix",
    "pong",
    "spaceinvaders",
    "timepilot",
)
CROSS_SKILL_TRANSFER_TARGETS = ("beamrider", "phoenix", "pong")

PRIORITY_TRAIN = "tf_seaquest_skills_ego_reward_semantics_v2_seed44"
PRIORITY_EVAL = f"{PRIORITY_TRAIN}_eval"


def _load(stem: str) -> dict[str, Any]:
    with (OUT / f"{stem}.yaml").open(encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"Expected mapping in {stem}.yaml")
    return doc


def _rename(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_rename(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _rename(item, old, new) for key, item in value.items()}
    return value


def _set_tag(doc: dict[str, Any], prefix: str, value: str) -> None:
    tags = list(dict.fromkeys(str(tag) for tag in doc.get("WANDB_TAGS", [])))
    tags = [tag for tag in tags if not tag.startswith(f"{prefix}:")]
    tags.append(f"{prefix}:{value}")
    doc["WANDB_TAGS"] = tags


def _set_both(doc: dict[str, Any], key: str, value: Any) -> None:
    doc[key] = value
    alg = doc.setdefault("alg", {})
    if isinstance(alg, dict):
        alg[key] = value


def _write(stem: str, doc: dict[str, Any], description: str) -> None:
    text = f"# {description}\n" + yaml.safe_dump(doc, sort_keys=False, width=100)
    (OUT / f"{stem}.yaml").write_text(text, encoding="utf-8")


def _reward_semantics_v2() -> tuple[str, str]:
    source = "tf_seaquest_skills_ego_seed44"
    train = _rename(_load(source), source, PRIORITY_TRAIN)
    _set_both(train, "COLLECT_DEATH_PRECEDENCE", "success_first")
    _set_both(train, "AVOID_KILLED_EVENT_MODE", "avoid_label_only")
    _set_tag(train, "arm", "skills_ego_reward_semantics_v2")
    _set_tag(train, "reward_semantics", "v2")
    _set_tag(train, "priority", "high")
    _write(
        PRIORITY_TRAIN,
        train,
        "High-priority Seaquest seed44: corrected off-policy reward precedence.",
    )

    source_eval = f"{source}_eval"
    eval_doc = _rename(_load(source_eval), source, PRIORITY_TRAIN)
    _set_both(eval_doc, "COLLECT_DEATH_PRECEDENCE", "success_first")
    _set_both(eval_doc, "AVOID_KILLED_EVENT_MODE", "avoid_label_only")
    _set_tag(eval_doc, "arm", "skills_ego_reward_semantics_v2")
    _set_tag(eval_doc, "reward_semantics", "v2")
    _set_tag(eval_doc, "priority", "high")
    _write(
        PRIORITY_EVAL,
        eval_doc,
        "Full Seaquest mod eval for corrected reward-semantics seed44.",
    )
    return PRIORITY_TRAIN, PRIORITY_EVAL


def _screen_frame_runs() -> list[str]:
    stems: list[str] = []
    for seed in SEEDS:
        source = f"tf_seaquest_skills_nopov_seed{seed}"
        target = f"tf_seaquest_skills_screen_seed{seed}"
        train = _rename(_load(source), source, target)
        _set_both(train, "EGO_OBSERVATION_FRAME", "screen")
        _set_both(train, "EGO_ACTION_FRAME", "screen")
        _set_tag(train, "arm", "skills_screen")
        _set_tag(train, "mapper", "screen_fixed")
        _set_tag(train, "observation_frame", "screen")
        _set_tag(train, "action_frame", "screen")
        _write(
            target,
            train,
            f"Seaquest fixed-screen observation/action ablation, seed{seed}.",
        )
        stems.append(target)

        source_eval = f"{source}_eval"
        target_eval = f"{target}_eval"
        eval_doc = _rename(_load(source_eval), source, target)
        _set_both(eval_doc, "EGO_OBSERVATION_FRAME", "screen")
        _set_both(eval_doc, "EGO_ACTION_FRAME", "screen")
        _set_tag(eval_doc, "arm", "skills_screen")
        _set_tag(eval_doc, "mapper", "screen_fixed")
        _set_tag(eval_doc, "observation_frame", "screen")
        _set_tag(eval_doc, "action_frame", "screen")
        _write(
            target_eval,
            eval_doc,
            f"Full Seaquest mod eval for fixed-screen ablation, seed{seed}.",
        )
        stems.append(target_eval)
    return stems


def _screen_transfer_runs() -> list[str]:
    """Derive clean fixed-screen transfers from the tracked ego transfer matrix."""
    stems: list[str] = []
    for env_name in TRANSFER_TARGETS:
        for seed in SEEDS:
            source = f"tf_zs_skills_ego_seaquest_to_{env_name}_seed{seed}"
            target = f"tf_zs_skills_screen_seaquest_to_{env_name}_seed{seed}"
            doc = _rename(_load(source), "skills_ego", "skills_screen")
            _set_both(doc, "USE_POV_RELATIVE_MAPPER", False)
            _set_both(doc, "USE_POV_FIXED_BILATERAL", False)
            _set_both(doc, "USE_POV_FACING_FRAME", False)
            _set_both(doc, "EGO_ANGLE_BINS", 8)
            _set_both(doc, "EGO_PSI_ANGLE_BINS", None)
            _set_both(doc, "EGO_OBSERVATION_FRAME", "screen")
            _set_both(doc, "EGO_ACTION_FRAME", "screen")
            _set_tag(doc, "arm", "skills_screen")
            _set_tag(doc, "mapper", "screen_fixed")
            _set_tag(doc, "observation_frame", "screen")
            _set_tag(doc, "action_frame", "screen")
            _write(
                target,
                doc,
                f"Zero-shot fixed-screen Seaquest skills to {env_name}, seed{seed}.",
            )
            stems.append(target)
    return stems


def _cross_skill_ablation_runs() -> list[str]:
    """Final-recipe seed44 with only cross-skill update eligibility disabled."""
    source = "tf_seaquest_skills_ego_seed44"
    target = "tf_seaquest_skills_ego_no_cross_skill_seed44"

    train = _rename(_load(source), source, target)
    _set_both(train, "OFF_POLICY", False)
    _set_tag(train, "arm", "skills_ego_no_cross_skill")
    _set_tag(train, "ablation", "cross_skill_updates_off")
    _set_tag(train, "priority", "diagnostic")
    _write(
        target,
        train,
        "Seaquest seed44: final recipe without cross-skill updates.",
    )

    source_eval = f"{source}_eval"
    target_eval = f"{target}_eval"
    eval_doc = _rename(_load(source_eval), source, target)
    _set_both(eval_doc, "OFF_POLICY", False)
    _set_tag(eval_doc, "arm", "skills_ego_no_cross_skill")
    _set_tag(eval_doc, "ablation", "cross_skill_updates_off")
    _set_tag(eval_doc, "priority", "diagnostic")
    _write(
        target_eval,
        eval_doc,
        "Full Seaquest mod eval without cross-skill updates, seed44.",
    )

    stems = [target, target_eval]
    for env_name in CROSS_SKILL_TRANSFER_TARGETS:
        transfer_source = f"tf_zs_skills_ego_seaquest_to_{env_name}_seed44"
        transfer_target = (
            f"tf_zs_skills_ego_no_cross_skill_seaquest_to_{env_name}_seed44"
        )
        doc = _rename(_load(transfer_source), "skills_ego", "skills_ego_no_cross_skill")
        _set_both(doc, "OFF_POLICY", False)
        _set_tag(doc, "arm", "skills_ego_no_cross_skill")
        _set_tag(doc, "ablation", "cross_skill_updates_off")
        _set_tag(doc, "priority", "diagnostic")
        _write(
            transfer_target,
            doc,
            f"Zero-shot no-cross-skill Seaquest skills to {env_name}, seed44.",
        )
        stems.append(transfer_target)
    return stems


def _write_manifest(
    priority: tuple[str, str],
    screen: list[str],
    transfers: list[str],
    cross_skill: list[str],
) -> None:
    doc = {
        "priority": list(priority),
        "screen_frame": screen,
        "screen_transfer": transfers,
        "cross_skill_ablation": cross_skill,
        "decision_rule": (
            "Reward-semantics-v2 was rejected after its seed44 base score fell "
            "50 percent and no_enemies fell 65 percent. Retain the thesis-final "
            "rewards; do not propagate the v2 flags. Run screen-frame with the "
            "legacy rewards, then gate transfers on source competence."
        ),
    }
    MANIFEST.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    priority_args = " ".join(priority)
    screen_args = " ".join(screen)
    transfer_args = " ".join(transfers)
    cross_skill_args = " ".join(cross_skill)
    NOTES.write_text(
        "# Thesis semantics follow-up runs\n\n"
        "Generated by `scripts/materialize_thesis_semantics_followups.py`.\n\n"
        "## 1. Completed reward diagnostic (rejected)\n\n"
        "```bash\n"
        f"python scripts/run_paper_eval.py --only {priority_args} "
        "--gpus 0 --wait-deps --skip-rq-gate --lean --resume\n"
        "```\n\n"
        "This retains the trained POV/facing action frame and changes only:\n\n"
        "- `COLLECT_DEATH_PRECEDENCE: success_first`\n"
        "- `AVOID_KILLED_EVENT_MODE: avoid_label_only`\n\n"
        "This run completed and was rejected: base return fell about 50% and "
        "`no_enemies` about 65%. Keep it as an audit artifact. Do not propagate "
        "these flags to the thesis-final arms.\n\n"
        "## 2. Clean no-POV action-frame ablation\n\n"
        "```bash\n"
        f"python scripts/run_paper_eval.py --only {screen_args} "
        "--gpus 0,1,2 --wait-deps --skip-rq-gate --lean --resume\n"
        "```\n\n"
        "These runs use screen/world observations and `EGO_ACTION_FRAME: screen`; "
        "the mapper cannot inspect the selected target. They retain the legacy "
        "reward semantics to isolate the action-frame correction.\n\n"
        "## 3. Clean no-POV zero-shot transfer\n\n"
        "Run after all three screen-frame source checkpoints exist.\n\n"
        "```bash\n"
        f"python scripts/run_paper_eval.py --only {transfer_args} "
        "--gpus 0,1,2 --wait-deps --skip-rq-gate --lean --resume\n"
        "```\n\n"
        "## 4. Cross-skill update ablation\n\n"
        "One final-recipe seed with `OFF_POLICY: false`. This retains the original "
        "rewards and disables only cross-skill update eligibility. It runs the full "
        "Seaquest mod strip and zero-shot mod evaluations on BeamRider, Phoenix, "
        "and Pong.\n\n"
        "```bash\n"
        f"python scripts/run_paper_eval.py --only {cross_skill_args} "
        "--gpus 0,1,2 --wait-deps --skip-rq-gate --lean --resume\n"
        "```\n\n"
        "## Decision rule\n\n"
        "The reward gate is decided: retain the thesis-final reward contract and "
        "cancel the conditional full reward rerun. Run the screen-frame jobs with "
        "their existing legacy-reward configs. Launch the 27 transfers only after "
        "the three source checkpoints show nontrivial Seaquest competence.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    priority = _reward_semantics_v2()
    screen = _screen_frame_runs()
    transfers = _screen_transfer_runs()
    cross_skill = _cross_skill_ablation_runs()
    _write_manifest(priority, screen, transfers, cross_skill)
    print(
        f"Wrote {len(priority) + len(screen) + len(transfers) + len(cross_skill)} configs"
    )
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"Run notes: {NOTES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

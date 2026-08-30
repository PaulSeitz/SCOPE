#!/usr/bin/env python3
"""Emit ``_v2`` paper-eval YAMLs for post-facingfix Seaquest skill retraining.

Old facing-frame checkpoints are invalid (obs frame was rotated vs mapper). These
configs keep the same hyperparameters but write to new SAVE_PATHs so pre-fix
artifacts remain for comparison.

Usage:
    python scripts/materialize_pov_facing_v2_configs.py
    python scripts/materialize_pov_facing_v2_configs.py --phase 1
    python scripts/materialize_pov_facing_v2_configs.py --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"


def _train_stems(phase: str) -> tuple[str, ...]:
    stems = [
        "seaquest_skills_ego_pov_clip50_facing_psibins8",
        "seaquest_skills_ego_pov_clip50_facing_psibins8_mid",
    ]
    if phase in ("2", "all"):
        stems.append("seaquest_skills_ego_pov_clip50_facing_xy")
    return tuple(stems)


def _transfer_stems(phase: str) -> tuple[str, ...]:
    stems = [
        "pong_transfer_ego_pov_clip50_facing_psibins8",
        "pong_transfer_ego_pov_clip50_facing_psibins8_mid",
    ]
    if phase in ("2", "all"):
        stems.extend(
            [
                "pong_transfer_ego_pov_clip50_facing_xy",
                "beamrider_transfer_ego_pov_clip50_facing_psibins8",
                "beamrider_transfer_ego_pov_clip50_facing_xy",
            ]
        )
    return tuple(stems)


def _all_stems(phase: str) -> list[str]:
    out: list[str] = []
    for train in _train_stems(phase):
        out.append(train)
        out.append(f"{train}_eval_full")
    out.extend(_transfer_stems(phase))
    return out


def _v2_stem(stem: str) -> str:
    if stem.endswith("_eval_full"):
        return f"{stem[: -len('_eval_full')]}_v2_eval_full"
    return f"{stem}_v2"


def _seaquest_train_stem_for_transfer(transfer_stem: str) -> str:
    """``pong_transfer_ego_pov_*`` → matching ``seaquest_skills_ego_pov_*`` train stem."""
    prefix = "pong_transfer_ego_"
    if not transfer_stem.startswith(prefix):
        prefix = "beamrider_transfer_ego_"
    if not transfer_stem.startswith(prefix):
        raise ValueError(f"unknown transfer stem: {transfer_stem}")
    return "seaquest_skills_ego_" + transfer_stem[len(prefix) :]


def _patch_yaml(text: str, old_stem: str, new_stem: str) -> str:
    sq_old = _seaquest_train_stem_for_transfer(old_stem) if "transfer" in old_stem else None
    sq_new = _v2_stem(sq_old) if sq_old else None
    out = text.replace(old_stem, new_stem)
    if sq_old and sq_new:
        out = out.replace(sq_old, sq_new)
    if out.startswith("#"):
        first_nl = out.find("\n")
        header = out[:first_nl]
        if "facingfix" not in header.lower():
            out = header + " (facingfix v2 retrain)\n" + out[first_nl + 1 :]
    else:
        out = f"# facingfix v2 retrain ({new_stem})\n{out}"
    out = out.replace("suite:paper_eval", "suite:pov_facing_v2")
    out = re.sub(
        r"(WANDB_RUN_NAME: [^\n]+)",
        lambda m: m.group(1) + "_v2" if not m.group(1).endswith("_v2") else m.group(1),
        out,
    )
    return out


def _src_path(stem: str) -> Path:
    if stem.startswith(("pong_transfer", "beamrider_transfer")):
        return _RQ2 / f"{stem}.yaml"
    return _RQ1 / f"{stem}.yaml"


def materialize(stem: str, *, dry_run: bool) -> None:
    src = _src_path(stem)
    if not src.is_file():
        raise FileNotFoundError(src)
    new_stem = _v2_stem(stem)
    dst = src.parent / f"{new_stem}.yaml"
    body = _patch_yaml(src.read_text(), stem, new_stem)
    if dry_run:
        print(f"would write {dst.relative_to(_ROOT)}")
    else:
        dst.write_text(body)
        print(f"wrote {dst.relative_to(_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("1", "2", "all"),
        default="all",
        help="1=φ-bins8+mid+pong only; 2/all=add xy+beamrider",
    )
    args = parser.parse_args()
    for stem in _all_stems(args.phase):
        materialize(stem, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

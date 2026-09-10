#!/usr/bin/env python3
"""clip10 continuous near-field + exclusive mid/far tiers (surfacesub5).

Keeps min(d,10)/10 like clip10, then adds one-hots for bands after 10px:
  10–20 / 20–50 / 50–80 / 80–150 / ≥150  (d<10 has no tier bit — continuous only).

Source: clip10 noweapon surfacesub5 family.

    python scripts/materialize_surfacesub5_neartiers.py
    python scripts/run_paper_eval.py --suite ego_min_surfacesub5_neartiers \\
        --gpus 0 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip10_{SPINE}"
DST_TOK = f"neartiers_{SPINE}"
SUITE = "ego_min_surfacesub5_neartiers"
EDGES = "[10, 20, 50, 80, 150]"


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _family_paths(train_stem: str) -> dict[str, Path]:
    tok = train_stem.removeprefix("seaquest_skills_ego_pov_")
    return {
        "train": _RQ1 / f"{train_stem}.yaml",
        "eval": _RQ1 / f"{train_stem}_eval_full.yaml",
        "pong": _RQ2 / f"pong_transfer_ego_pov_{tok}.yaml",
        "beam": _RQ2 / f"beamrider_transfer_ego_pov_{tok}.yaml",
        "phoenix": _RQ2 / f"phoenix_transfer_ego_pov_{tok}.yaml",
    }


def _transform(text: str) -> str:
    src_tok = SRC_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, DST_TOK)
    out = out.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    # Keep clip10; add near_tiers mode + hard edges after the clip line.
    needle = "EGO_DISTANCE_CLIP_MAX_PX: 10\n"
    if needle not in out:
        raise SystemExit("expected EGO_DISTANCE_CLIP_MAX_PX: 10 in source YAML")
    out = out.replace(
        needle,
        (
            "EGO_DISTANCE_CLIP_MAX_PX: 10\n"
            "EGO_DISTANCE_MODE: near_tiers\n"
            f"EGO_DISTANCE_HARD_EDGES_PX: {EDGES}\n"
        ),
    )
    out = out.replace("- ego:clip10\n", "- ego:clip10\n- ego:neartiers\n")
    return out


def materialize(*, dry: bool) -> None:
    src = _family_paths(SRC_TRAIN)
    dst = _family_paths(f"seaquest_skills_ego_pov_{DST_TOK}")
    for key, src_path in src.items():
        if not src_path.exists():
            raise SystemExit(f"missing source {src_path}")
        _write(dst[key], _transform(src_path.read_text(encoding="utf-8")), dry=dry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()

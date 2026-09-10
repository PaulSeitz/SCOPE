#!/usr/bin/env python3
"""Three distance-block probes on clip50 surfacesub5 (φ8 kept).

1. deltaxy — body-frame (Δfwd, Δlat) / MAX_SPEED instead of scalar distance
2. deltasign — sign(Δfwd), sign(Δlat) ∈ {-1,0,1} (0.5px deadzone)
3. posxy — absolute body-frame (fwd, lat) clip-normalized (still + φ8)

Source spine: clip50 noweapon surfacesub5. Trains-first suite for 3-GPU shard.

    python scripts/materialize_surfacesub5_distpose.py
    python scripts/run_paper_eval.py --suite ego_min_surfacesub5_distpose \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SPINE = "facing_psibins8_v2_noblocking_noweapon_surfacesub5"
SRC_TRAIN = f"seaquest_skills_ego_pov_clip50_{SPINE}"
SUITE = "ego_min_surfacesub5_distpose"

ARMS = (
    {
        "tok": f"deltaxy_{SPINE}",
        "mode": "delta_xy",
        "tag": "- ego:deltaxy",
    },
    {
        "tok": f"deltasign_{SPINE}",
        "mode": "delta_xy_sign",
        "tag": "- ego:deltasign",
    },
    {
        "tok": f"posxy_{SPINE}",
        "mode": "xy",
        "tag": "- ego:posxy",
    },
)


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


def _transform(text: str, arm: dict) -> str:
    src_tok = SRC_TRAIN.removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, arm["tok"])
    out = out.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    out = out.replace("- suite:ego_min_noweapon_heu_ablation\n", "")
    # Keep clip50 as xy normalizer / unused scalar fallback; inject mode after it.
    out = out.replace(
        "EGO_DISTANCE_CLIP_MAX_PX: 50\n",
        f"EGO_DISTANCE_CLIP_MAX_PX: 50\nEGO_DISTANCE_MODE: {arm['mode']}\n",
    )
    out = out.replace("- ego:clip50", f"- ego:clip50\n{arm['tag']}")
    return out


def materialize(*, dry: bool) -> None:
    src = _family_paths(SRC_TRAIN)
    for arm in ARMS:
        dst_train = f"seaquest_skills_ego_pov_{arm['tok']}"
        dst = _family_paths(dst_train)
        for key, src_path in src.items():
            _write(
                dst[key],
                _transform(src_path.read_text(encoding="utf-8"), arm),
                dry=dry,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()

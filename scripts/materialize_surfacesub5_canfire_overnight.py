#!/usr/bin/env python3
"""Overnight can_fire re-intro batch on surfacesub5.

Arms (trains first → 3-GPU friendly):
  1. clip10 + can_fire          (already materialized; verified here)
  2. clip50 + can_fire
  3. neartiers + can_fire
  4. deltaxy + can_fire (φ8)
  5. deltaxy + can_fire + φ sin/cos  (the one angle variant)

Weapon spine matches EVAL_PLAN §0 lock: USE_EGO_WEAPON_FEATURES true,
USE_EGO_FACING_ENEMY_FEATURES false (can_fire only). Blocking stays off.

    python scripts/materialize_surfacesub5_canfire_overnight.py
    python scripts/run_paper_eval.py --suite ego_min_surfacesub5_canfire_overnight \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "ego_min_surfacesub5_canfire_overnight"

# (src_train_stem, dst_train_stem, suite_tag_replace, extra_transforms)
# src/dst are full seaquest_skills_ego_pov_* stems without path.
ARMS: list[dict] = [
    {
        "name": "clip10_canfire",
        "src": "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
        "dst": "seaquest_skills_ego_pov_clip10_facing_psibins8_v2_noblocking_canfire_surfacesub5",
        "already_done": True,
    },
    {
        "name": "clip50_canfire",
        "src": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
        "dst": "seaquest_skills_ego_pov_clip50_facing_psibins8_v2_noblocking_canfire_surfacesub5",
        "src_suite_tags": (
            "suite:ego_min_noweapon_surfacesub5_clipsweep",
            "suite:ego_min_noweapon_heu_ablation",
        ),
    },
    {
        "name": "neartiers_canfire",
        "src": "seaquest_skills_ego_pov_neartiers_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
        "dst": "seaquest_skills_ego_pov_neartiers_facing_psibins8_v2_noblocking_canfire_surfacesub5",
        "src_suite_tags": ("suite:ego_min_surfacesub5_neartiers",),
    },
    {
        "name": "deltaxy_canfire",
        "src": "seaquest_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
        "dst": "seaquest_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_canfire_surfacesub5",
        "src_suite_tags": ("suite:ego_min_surfacesub5_distpose",),
    },
    {
        "name": "deltaxy_sincos_canfire",
        "src": "seaquest_skills_ego_pov_deltaxy_facing_psibins8_v2_noblocking_noweapon_surfacesub5",
        "dst": "seaquest_skills_ego_pov_deltaxy_facing_sincos_v2_noblocking_canfire_surfacesub5",
        "src_suite_tags": ("suite:ego_min_surfacesub5_distpose",),
        "sincos": True,
    },
]


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


def _to_canfire(text: str, *, src_stem: str, dst_stem: str, sincos: bool) -> str:
    src_tok = src_stem.removeprefix("seaquest_skills_ego_pov_")
    dst_tok = dst_stem.removeprefix("seaquest_skills_ego_pov_")
    out = text.replace(src_tok, dst_tok)
    # Suite tags → overnight suite
    for tag in (
        "suite:ego_min_noweapon_surfacesub5_clipsweep",
        "suite:ego_min_noweapon_heu_ablation",
        "suite:ego_min_surfacesub5_neartiers",
        "suite:ego_min_surfacesub5_distpose",
        "suite:ego_min_surfacesub5_clip10_canfire",
    ):
        out = out.replace(tag, f"suite:{SUITE}")
    out = out.replace("USE_EGO_WEAPON_FEATURES: false", "USE_EGO_WEAPON_FEATURES: true")
    if "USE_EGO_FACING_ENEMY_FEATURES" not in out:
        out = out.replace(
            "USE_EGO_WEAPON_FEATURES: true\n",
            "USE_EGO_WEAPON_FEATURES: true\nUSE_EGO_FACING_ENEMY_FEATURES: false\n",
        )
    else:
        out = out.replace(
            "USE_EGO_FACING_ENEMY_FEATURES: true",
            "USE_EGO_FACING_ENEMY_FEATURES: false",
        )
    out = out.replace("- ego:noweapon\n", "- ego:can_fire_only\n")
    if sincos:
        out = out.replace("EGO_PSI_ANGLE_BINS: 8\n", "EGO_PSI_ANGLE_BINS: null\n")
        out = out.replace("- ego:phi_bins8\n", "- ego:phi_sincos\n")
        # stem already has facing_sincos; ensure tags consistent
    return out


def materialize(*, dry: bool) -> None:
    for arm in ARMS:
        src = _family_paths(arm["src"])
        dst = _family_paths(arm["dst"])
        if arm.get("already_done"):
            missing = [k for k, p in dst.items() if not p.exists()]
            if missing:
                raise SystemExit(
                    f"{arm['name']}: expected existing configs missing: {missing}"
                )
            print(f"OK exists {arm['name']} → {arm['dst']}")
            # Retag suite name on existing clip10_canfire family for overnight batch?
            # Keep original suite tag on disk; overnight suite lists stems explicitly.
            continue
        sincos = bool(arm.get("sincos"))
        for key, src_path in src.items():
            if not src_path.exists():
                raise SystemExit(f"missing source {src_path}")
            text = _to_canfire(
                src_path.read_text(encoding="utf-8"),
                src_stem=arm["src"],
                dst_stem=arm["dst"],
                sincos=sincos,
            )
            _write(dst[key], text, dry=dry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    materialize(dry=args.dry)


if __name__ == "__main__":
    main()

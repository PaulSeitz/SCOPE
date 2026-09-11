#!/usr/bin/env python3
"""Re-train clip10/clip50 lock spine after skill-mechanics (ungated) rewards.

Fresh stems so we don't resume old ckpts. Same YAML knobs as the lock; only
SAVE_PATH / deps / suite tags change. Plus one clip10 arm with
RELABEL_ALL_OBJECTS (train on every on-screen object like the selected target).

Regression family (× clip10, clip50):
  SQ train → SQ eval_full → Pong/BR/PH ZS

All-objects probe:
  clip10 + RELABEL_ALL_OBJECTS → eval_full → Pong/BR/PH ZS

    python scripts/materialize_surfacesub5_skillmech_rw.py
    python scripts/run_paper_eval.py --suite ego_min_surfacesub5_skillmech_rw \\
        --gpus 0,1,2 --wait-deps --lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RQ1 = _ROOT / "configs/experiment/paper_eval/rq1"
_RQ2 = _ROOT / "configs/experiment/paper_eval/rq2"

SUITE = "ego_min_surfacesub5_skillmech_rw"
# Lock token replaced inside stems/paths (must be unique suffix of the lock spine).
SPINE_LOCK = "noblocking_noweapon_surfacesub5"
REGRESSION_TAG = "skillmech"
RELABEL_TAG = "skillmech_relabel"
CLIPS = ("clip10", "clip50")
TRANSFER_ENVS = ("pong", "beamrider", "phoenix")


def _src_train(clip: str) -> str:
    return f"seaquest_skills_ego_pov_{clip}_facing_psibins8_v2_{SPINE_LOCK}"


def _dst_train(clip: str, tag: str) -> str:
    return f"{_src_train(clip)}_{tag}"


def _tok(train_stem: str) -> str:
    return train_stem.removeprefix("seaquest_skills_ego_pov_")


def _family(train_stem: str) -> dict[str, Path]:
    tok = _tok(train_stem)
    return {
        "train": _RQ1 / f"{train_stem}.yaml",
        "eval": _RQ1 / f"{train_stem}_eval_full.yaml",
        "pong": _RQ2 / f"pong_transfer_ego_pov_{tok}.yaml",
        "beamrider": _RQ2 / f"beamrider_transfer_ego_pov_{tok}.yaml",
        "phoenix": _RQ2 / f"phoenix_transfer_ego_pov_{tok}.yaml",
    }


def _write(path: Path, text: str, *, dry: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"DRY {path.relative_to(_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(_ROOT)}")


def _rewrite(text: str, *, tag: str, relabel: bool) -> str:
    spine_dst = f"{SPINE_LOCK}_{tag}"
    if spine_dst in text:
        raise ValueError(f"source already contains {spine_dst!r}")
    out = text.replace(SPINE_LOCK, spine_dst)
    # SPINE_LOCK also sits inside suite:ego_min_noweapon_surfacesub5_clipsweep?
    # No — that suite string uses noweapon_surfacesub5 without noblocking_.
    out = out.replace(
        "suite:ego_min_noweapon_surfacesub5_clipsweep", f"suite:{SUITE}"
    )
    if f"- ablation:{tag}" not in out:
        out = out.replace(
            "- ablation:ego_min_channels\n",
            f"- ablation:ego_min_channels\n- ablation:{tag}\n",
        )
    if relabel:
        out = out.replace(
            "RELABEL_ALL_OBJECTS: false", "RELABEL_ALL_OBJECTS: true"
        )
        if "- ablation:relabel_all" not in out:
            out = out.replace(
                f"- ablation:{tag}\n",
                f"- ablation:{tag}\n- ablation:relabel_all\n",
            )
    return out


def materialize(*, dry: bool) -> None:
    for clip in CLIPS:
        src_train = _src_train(clip)
        src = _family(src_train)
        for key, src_path in src.items():
            if not src_path.exists():
                raise FileNotFoundError(src_path)

        dst_train = _dst_train(clip, REGRESSION_TAG)
        dst = _family(dst_train)
        for key, src_path in src.items():
            body = _rewrite(
                src_path.read_text(encoding="utf-8"),
                tag=REGRESSION_TAG,
                relabel=False,
            )
            _write(dst[key], body, dry=dry)

    src_train = _src_train("clip10")
    src = _family(src_train)
    dst_train = _dst_train("clip10", RELABEL_TAG)
    dst = _family(dst_train)
    for key, src_path in src.items():
        body = _rewrite(
            src_path.read_text(encoding="utf-8"),
            tag=RELABEL_TAG,
            relabel=True,
        )
        _write(dst[key], body, dry=dry)


def stem_lists() -> list[str]:
    stems: list[str] = []
    # Trains first so --gpus 0,1,2 shards one train per GPU.
    for tag, clips in (
        (REGRESSION_TAG, CLIPS),
        (RELABEL_TAG, ("clip10",)),
    ):
        for clip in clips:
            stems.append(_dst_train(clip, tag))
    for tag, clips in (
        (REGRESSION_TAG, CLIPS),
        (RELABEL_TAG, ("clip10",)),
    ):
        for clip in clips:
            train = _dst_train(clip, tag)
            stems.append(f"{train}_eval_full")
            tok = _tok(train)
            for env in TRANSFER_ENVS:
                stems.append(f"{env}_transfer_ego_pov_{tok}")
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--print-stems", action="store_true")
    args = parser.parse_args()
    if args.print_stems:
        for s in stem_lists():
            print(s)
        return
    materialize(dry=args.dry)
    stems = stem_lists()
    print(f"\n{len(stems)} stems ({SUITE})")


if __name__ == "__main__":
    main()

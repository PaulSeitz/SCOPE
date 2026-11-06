#!/usr/bin/env python3
"""Eval-only transfer of existing Seaquest checkpoints onto T9.

Arms (no new training):
  skills_nopov, skills_pixel_outline, tandem_pixel skills → all 9 targets
  tandem_oc skills → the 7 targets that are not already in thesis_final
  (Name This Game and Time Pilot tandem-OC transfers already exist)

    python scripts/materialize_extended_transfer_eval.py
    CUDA_VISIBLE_DEVICES=1 python scripts/run_paper_eval.py \\
        --suite extended_transfer_fast --wait-deps --skip-rq-gate --lean
    CUDA_VISIBLE_DEVICES=2 python scripts/run_paper_eval.py \\
        --suite extended_transfer_pixel --wait-deps --skip-rq-gate --lean
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from materialize_thesis_final import (  # noqa: E402
    SEEDS,
    Run,
    _OUT,
    _skills_eval_body,
    _tags,
    zs_stem,
)

T9 = (
    "namethisgame",
    "timepilot",
    "spaceinvaders",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "phoenix",
    "beamrider",
)
# tandem-OC → these two is already in thesis_final.
TANDEM_OC_DONE = {"namethisgame", "timepilot"}

# (zs arm name, train stem prefix, skill-input tag, targets)
ARMS = (
    ("skills_nopov", "tf_seaquest_skills_nopov", "ego", T9),
    ("tandem_oc_skills", "tf_seaquest_tandem_oc", "ego", tuple(e for e in T9 if e not in TANDEM_OC_DONE)),
    ("skills_pixel_outline", "tf_seaquest_skills_pixel_outline", "pixel_embed", T9),
    ("tandem_pixel_skills", "tf_seaquest_tandem_pixel", "ego", T9),
)


def _load_train_doc(prefix: str, seed: int) -> dict:
    path = _OUT / f"{prefix}_seed{seed}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise SystemExit(f"Bad train yaml {path}")
    return raw


def _train_run(prefix: str, seed: int, doc: dict) -> Run:
    return Run(
        stem=f"{prefix}_seed{seed}",
        tier="extras",
        rq="2",
        phase="train",
        arm=prefix,
        env="seaquest",
        seed=seed,
        hours=0.0,
        measured=False,
        doc=doc,
        header="",
    )


def iter_runs():
    for zs_arm, prefix, _skill, targets in ARMS:
        tandem = zs_arm.startswith("tandem_")
        for seed in SEEDS:
            doc = _load_train_doc(prefix, seed)
            train = _train_run(prefix, seed, doc)
            for dst in targets:
                stem = zs_stem(zs_arm, "seaquest", dst, seed)
                tags = _tags(
                    "suite:extended_transfer",
                    "rq:2",
                    f"env:{dst}",
                    "src:seaquest",
                    f"arm:{zs_arm}",
                    "phase:zero_shot",
                    "budget:50m_optim",
                )
                body = _skills_eval_body(
                    train,
                    eval_env=dst,
                    kind="transfer",
                    stem=stem,
                    rq=2,
                    tags=tags,
                    tandem=tandem,
                    meta=None,
                )
                header = (
                    f"# Extended ZS — {train.stem} skills + {dst} heuristic.\n"
                )
                yield stem, header + yaml.safe_dump(body, sort_keys=False, width=100)


def fast_stems() -> tuple[str, ...]:
    """no-POV transfers, then remaining tandem-OC transfers."""
    stems = []
    for zs_arm, _prefix, _skill, targets in ARMS[:2]:
        for seed in SEEDS:
            for dst in targets:
                stems.append(zs_stem(zs_arm, "seaquest", dst, seed))
    return tuple(stems)


def pixel_stems() -> tuple[str, ...]:
    """Pixel-skill transfers, then tandem-pixel extracted ego-skill transfers."""
    stems = []
    for zs_arm, _prefix, _skill, targets in ARMS[2:]:
        for seed in SEEDS:
            for dst in targets:
                stems.append(zs_stem(zs_arm, "seaquest", dst, seed))
    return tuple(stems)


def materialize() -> int:
    n = 0
    for stem, text in iter_runs():
        path = _OUT / f"{stem}.yaml"
        path.write_text(text, encoding="utf-8")
        n += 1
    return n


def main() -> None:
    n = materialize()
    print(f"wrote {n} eval yamls")
    print(f"fast queue {len(fast_stems())}  pixel queue {len(pixel_stems())}")


if __name__ == "__main__":
    main()

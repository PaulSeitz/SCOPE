#!/usr/bin/env python3
"""Print a raw-OC tandem train/home-eval/transfer chain; execute only with --run."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "seaquest",
    "namethisgame",
    "spaceinvaders",
    "beamrider",
    "asteroids",
    "pong",
    "breakout",
    "freeway",
    "phoenix",
)


def stems(env: str, seed: int):
    source = f"tf_{env}_tandem_ocraw_seed{seed}"
    return [(source, "train"), (source + "_eval", "eval")] + [
        (f"tf_zs_tandem_ocraw_skills_{env}_to_{target}_seed{seed}", "eval")
        for target in TARGETS
        if target != env
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["seaquest", "namethisgame"], required=True)
    parser.add_argument("--seed", type=int, choices=[44, 45, 46], required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    for stem, phase in stems(args.env, args.seed):
        command = [
            sys.executable,
            str(ROOT / "scripts/run_paper_eval.py"),
            "--only",
            stem,
            "--phase",
            phase,
            "--resume",
            "--abort-on-failure",
        ]
        print(" ".join(command), flush=True)
        if args.run:
            subprocess.run(command, cwd=ROOT, check=True)

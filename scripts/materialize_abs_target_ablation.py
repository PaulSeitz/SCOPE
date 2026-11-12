#!/usr/bin/env python3
"""Materialize the ten absolute-target trains from their matching SCOPE configs.

Only input mode and run identity change. No training or evaluation is launched.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/experiment/paper_eval/thesis_final"
OUTPUT = ROOT / "configs/experiment/paper_eval/abs_target_ablation"


def materialize(reference=REFERENCE, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for game in ("seaquest", "namethisgame"):
        for seed in range(44, 49):
            old = f"tf_{game}_skills_ego_seed{seed}"
            new = f"tf_{game}_skills_ego_abstarget_seed{seed}"
            cfg = copy.deepcopy(yaml.safe_load((reference / f"{old}.yaml").read_text()))
            cfg["SKILL_INPUT_MODE"] = cfg["alg"]["SKILL_INPUT_MODE"] = "ego_abs_target"
            cfg["SAVE_PATH"] = f"./outputs/models/paper_eval/abs_target_ablation/{new}"
            cfg["WANDB_RUN_NAME"] = cfg["WANDB_RUN_NAME"].replace(old, new)
            cfg["WANDB_TAGS"] = [
                {"arm:skills_ego": "arm:skills_ego_abstarget",
                 "skill_input:ego": "skill_input:ego_abs_target"}.get(tag, tag)
                for tag in cfg.get("WANDB_TAGS", [])
            ]
            path = output / f"{new}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path in materialize(args.reference, args.output):
        print(path)

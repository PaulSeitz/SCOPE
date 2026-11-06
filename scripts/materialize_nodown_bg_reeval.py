#!/usr/bin/env python3
"""Name This Game background re-eval of the nodown pixel checkpoints.

The nodown home evals omit change_background_color, which the frozen pixel
evals had through ntg_bg_reeval. These evaluation-only configs add that key for
the target-marked pixel skills and the pixel tandem agent, and train nothing.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/experiment/paper_eval/probes_nodown_sticky"
OUT = ROOT / "configs/experiment/paper_eval/ntg_bg_reeval"
ARMS = ("skills_pixel_outline", "tandem_pixel_slots")
SEEDS = (44, 45, 46, 47, 48)

for arm in ARMS:
    for seed in SEEDS:
        src = f"tf_namethisgame_{arm}_seed{seed}_nodown_eval"
        stem = f"{src}_bg"
        doc = yaml.safe_load((SRC / f"{src}.yaml").read_text())
        assert doc["META_PIXEL_USE_NATIVE_DOWNSCALING"] is False, src
        doc["PAPER_EVAL_MOD_KEYS"] = ["change_background_color"]
        doc["PAPER_EVAL_REQUIRES_RQ1"] = []
        doc["SAVE_PATH"] = f"{doc['SAVE_PATH']}_bg"
        doc["WANDB_RUN_NAME"] = f"{doc['WANDB_RUN_NAME']}_bg"
        header = f"# Background re-eval of {src}; evaluation only.\n"
        (OUT / f"{stem}.yaml").write_text(header + yaml.safe_dump(doc, sort_keys=False))
        print(stem)

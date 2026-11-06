#!/usr/bin/env python3
"""Action-frame sensitivity check for the broader skill inputs (Seaquest only).

Native-OC and target-marked pixel skills keep their inputs, heuristic selector,
rewards, network and optimisation, and switch only to fixed-screen actions.
Name This Game is omitted since the player never turns there. Reported in the
appendix whatever it shows; it does not replace the main arms.

Usage:
    python materialize_action_frame_sensitivity.py --root REPO --out OUTDIR [--check]
"""
import argparse, copy
from pathlib import Path
import yaml

SEEDS = (44, 45, 46, 47, 48)
# arm: (config subdir of the source, source stem pattern)
ARMS = {
    "skills_ocnative_screenact": ("rewardfix_wave", "tf_seaquest_skills_ocnative_seed{s}"),
    "skills_pixel_outline_nodown_screenact": ("probes_nodown_sticky", "tf_seaquest_skills_pixel_outline_seed{s}_nodown"),
}

def retag(tags, arm):
    out = [f"arm:{arm}" if t.startswith("arm:") else ("mapper:world" if t.startswith("mapper:") else t) for t in tags]
    return [("rq:1abl" if t == "rq:1" else t) for t in out] + ["ablation:action_frame_sensitivity"]

def apply_frame(doc):
    # Top-level and alg copy, as in the fixed-screen configs. The observation
    # frame and USE_POV_* stay untouched, so the selector and inputs do not change.
    doc["EGO_ACTION_FRAME"] = "screen"
    doc.setdefault("alg", {})["EGO_ACTION_FRAME"] = "screen"

def build(root, arm, sub, pat, s):
    src = pat.format(s=s); stem = f"tf_seaquest_{arm}_seed{s}"
    d = root / "configs/experiment/paper_eval" / sub
    tr = yaml.safe_load((d / f"{src}.yaml").read_text())
    ev = yaml.safe_load((d / f"{src}_eval.yaml").read_text())
    apply_frame(tr); apply_frame(ev)
    tr["SAVE_PATH"] = f"./outputs/models/paper_eval/thesis_final/{stem}"
    tr["WANDB_RUN_NAME"] = f"sq_{stem}"; tr["WANDB_TAGS"] = retag(tr["WANDB_TAGS"], arm)
    ev["PAPER_EVAL_REQUIRES_RQ1"] = [stem]; ev["PAPER_EVAL_ARCHITECTURE_OF"] = stem
    assert f"/{src}/" in ev["PAPER_EVAL_SKILLS_CKPT"], src
    ev["PAPER_EVAL_SKILLS_CKPT"] = ev["PAPER_EVAL_SKILLS_CKPT"].replace(f"/{src}/", f"/{stem}/")
    ev["SAVE_PATH"] = f"./outputs/models/paper_eval/thesis_final/{stem}_eval"
    ev["WANDB_RUN_NAME"] = f"sq_{stem}_eval"; ev["WANDB_TAGS"] = retag(ev["WANDB_TAGS"], arm)
    return {stem: tr, f"{stem}_eval": ev}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); n = 0
    for arm, (sub, pat) in ARMS.items():
        for s in SEEDS:
            for stem, doc in build(a.root, arm, sub, pat, s).items():
                n += 1
                if not a.check:
                    a.out.mkdir(parents=True, exist_ok=True)
                    (a.out / f"{stem}.yaml").write_text(f"# Action-frame sensitivity check, screen actions only.\n" + yaml.safe_dump(doc, sort_keys=False))
                print(stem)
    print(n, "configs")

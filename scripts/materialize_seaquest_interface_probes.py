#!/usr/bin/env python3
"""Materialize three seed-44 probes without rewriting the main thesis suite."""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/experiment/paper_eval/thesis_final"
SOURCE = "tf_seaquest_skills_ego_avsep_seed44"
TARGETS = ("namethisgame", "spaceinvaders", "asteroids", "pong",
           "breakout", "freeway", "phoenix", "beamrider")
VARIANTS = (
    ("a_early", True, "centre_ray"),
    ("b_closest", False, "closest_points"),
    ("c_combined", True, "closest_points"),
)

def load(stem):
    return yaml.safe_load((OUT / f"{stem}.yaml").read_text())

def rename(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rename(x, old, new) for x in value]
    if isinstance(value, dict):
        return {k: rename(v, old, new) for k, v in value.items()}
    return value

def both(doc, key, value):
    doc[key] = value
    doc.setdefault("alg", {})[key] = value

def configure(doc, variant, early, geometry, source_env):
    for key, value in {
        "COLLECT_EARLY_WHEN_NO_THREATS": early if source_env else False,
        "EGO_AABB_GEOM": geometry,
        "AVOID_SPARSE_ONLY": False,
        "AVOID_ESCAPE_BONUS": 0.01,
        "AVOID_KILLED_EVENT_MODE": "none",
        "COLLECT_DEATH_PRECEDENCE": "death_first",
        "COLLECT_SPARSE_ONLY": True,
        "SKILL_DEATH_PENALIZE_COLLECT": True,
        "DESTROY_CONTACT_FAIL": True,
        "OFF_POLICY": True,
    }.items():
        both(doc, key, value)
    doc["WANDB_TAGS"] = list(dict.fromkeys(
        [*doc.get("WANDB_TAGS", []), "probe:seaquest_interface", f"variant:{variant}"]
    ))
    return doc

def main():
    manifest = {"source_reference": SOURCE, "variants": {}}
    for variant, early, geometry in VARIANTS:
        token = f"skills_ego_probe_{variant}"
        train = f"tf_seaquest_{token}_seed44"
        stems = []
        for original, stem in ((SOURCE, train), (SOURCE + "_eval", train + "_eval")):
            doc = rename(load(original), SOURCE, train)
            configure(doc, variant, early, geometry, True)
            (OUT / f"{stem}.yaml").write_text(
                f"# Seed-44 interface development probe {variant}.\n" +
                yaml.safe_dump(doc, sort_keys=False, width=100)
            )
            stems.append(stem)
        for target in TARGETS:
            old = f"tf_zs_skills_ego_avsep_seaquest_to_{target}_seed44"
            stem = f"tf_zs_{token}_seaquest_to_{target}_seed44"
            doc = rename(rename(load(old), old, stem), SOURCE, train)
            configure(doc, variant, early, geometry, False)
            doc["PAPER_EVAL_MOD_KEYS"] = ["base"]
            both(doc, "SAVE_PATH", f"./outputs/models/paper_eval/thesis_final/{stem}")
            doc["WANDB_RUN_NAME"] = stem
            (OUT / f"{stem}.yaml").write_text(
                f"# Frozen {variant} source skills to {target}. Same source geometry.\n" +
                yaml.safe_dump(doc, sort_keys=False, width=100)
            )
            stems.append(stem)
        manifest["variants"][variant] = {
            "early_collect": early, "geometry": geometry,
            "train": train, "runs_in_order": stems,
        }
        print(variant + ": " + " ".join(stems))
    (OUT / "_seaquest_interface_probes_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=100)
    )

if __name__ == "__main__":
    main()

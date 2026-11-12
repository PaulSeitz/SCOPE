#!/usr/bin/env python3
"""Add the results integrated after the 2026-11-10 suite to the replay specification.

Maintenance tool, like ``materialize_evaluation.py``. It extends the committed
suite with

* the absolute-target ablation (J): ten home-game checkpoints and their 80
  zero-shot transfers, appended to ``frozen_2026-11-07`` on 2026-11-11. Their
  checkpoints ship inside the snapshot under ``checkpoints/``, so their jobs
  are marked ``checkpoint_in_archive``;
* the Seaquest rescue-metric rerun of 2026-11-12 (NEXUS symbolic pixel, HRL
  pixel and HRL OC), recorded as supplementary inputs with hashes. Returns in the
  figures stay on the frozen evaluations; only the rescue counters come from it.

Existing runs and configs are left unchanged. The figure manifest decides which
inputs belong to the thesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation_config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
DROP = {"SAVE_PATH", "MODEL_PATH", "SKILLS_PATH", "META_PATH", "WANDB_MODE", "RTPT_ENABLED",
        "RTPT_EXPERIMENT_NAME", "RECORD_VIDEO", "VIDEO_AT_END"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def home_config(stem):
    """Same construction as scripts/run_abs_target_campaign.py:train_cfg."""
    root = load_yaml(ROOT / "configs/config.yaml")
    root.pop("defaults", None)
    raw = load_yaml(ROOT / f"configs/experiment/paper_eval/abs_target_ablation/{stem}.yaml")
    cfg = {**root, **{k: v for k, v in raw.items() if k != "alg"}, **raw["alg"]}
    assert cfg["SKILL_INPUT_MODE"] == "ego_abs_target"
    return {k: v for k, v in cfg.items() if not k.startswith("PAPER_EVAL_") and k not in DROP}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True, help="frozen_2026-11-07 directory")
    parser.add_argument("--rescue", type=Path, required=True, help="rescue_metrics_20261112 directory")
    parser.add_argument("--spec", type=Path, default=ROOT / "configs/reproduce/thesis.json")
    args = parser.parse_args()
    manifest = json.loads(args.figure_manifest.read_text())
    spec = json.loads(args.spec.read_text())
    frozen, rescue = args.frozen.resolve(), args.rescue.resolve()
    added, supplementary = 0, {}
    for source, digest in sorted(manifest["source_files_sha256"].items()):
        name = Path(source).name
        if "rescue_metrics_20261112" in Path(source).parts:
            path = rescue / "eval" / name
            if sha(path) != digest:
                raise ValueError(f"Rescue input hash mismatch: {name}")
            supplementary[name] = digest
            continue
        path = frozen / "eval" / name
        if sha(path) != digest:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        if name in spec["runs"]:
            if spec["runs"][name]["sha256"] != digest:
                raise ValueError(f"Suite hash differs from the figure manifest: {name}")
            continue
        data = json.loads(path.read_text())
        if data.get("skill_input_mode") != "ego_abs_target":
            raise ValueError(f"Unexpected new input outside the J ablation: {name}")
        ckpt = data["checkpoint_snapshot_path"]
        if not (frozen / ckpt).is_file() or sha(frozen / ckpt) != data.get("checkpoint_sha256", sha(frozen / ckpt)):
            raise ValueError(f"Bundled checkpoint missing or changed: {name}: {ckpt}")
        if data["agent_kind"] == "transfer":
            # The first batch (ABS_TARGET_TRANSFER_2026-11-11.json) records only its basis,
            # "matching frozen SCOPE transfer configs"; the later batch names the config.
            reference = data.get("reference_config") or \
                f"tf_zs_skills_ego_{data['source_env']}_to_{data['target_env']}_seed{data['source_seed']}"
            if reference not in spec["configs"]:
                raise ValueError(f"Reference SCOPE transfer config not in suite: {reference}")
            stem = reference.replace("tf_zs_skills_ego_", "tf_zs_skills_ego_abstarget_")
            config = dict(spec["configs"][reference], SKILL_INPUT_MODE="ego_abs_target")
        else:
            stem = name.removesuffix("_eval_mods.json")
            config = home_config(stem)
        if config["ENV_NAME"] != data["env"]:
            raise ValueError(f"Config environment differs: {name}")
        spec["configs"].setdefault(stem, config)
        if spec["configs"][stem] != config:
            raise ValueError(f"Conflicting config: {stem}")
        mods = {m["key"]: m for m in data["mods"]}
        jobs = {}
        for key, result in sorted(data["results"].items()):
            if not result.get("returns"):
                continue
            jobs[key] = {k: data[k] for k in ("base_seed", "num_episodes", "num_seeds", "max_frames")}
            jobs[key].update(kind=data["agent_kind"], config=stem, object_centric=data["object_centric"],
                             skills_path=ckpt, checkpoint_in_archive=True, mod=mods[key])
        spec["runs"][name] = {"sha256": digest, "env": data["env"], "agent_kind": data["agent_kind"], "jobs": jobs}
        added += 1
    if len(supplementary) != 15:
        raise ValueError(f"Expected 15 rescue-metric inputs, found {len(supplementary)}")
    spec["supplementary"] = {"rescue_metrics_20261112": {
        "description": "Seaquest rescue counters for NEXUS symbolic pixel, HRL pixel and HRL OC "
                       "(base, no_enemies), measured in a separate rerun on 2026-11-12. Figures keep "
                       "the frozen returns and take only the counters from these files.",
        "files": supplementary}}
    spec["missing_queries"] = manifest["missing_queries"]
    spec["main_layout"] = manifest["main_layout"]
    total = len(spec["runs"]) + len(supplementary)
    if total != len(manifest["source_files_sha256"]):
        raise ValueError(f"Suite covers {total} inputs, manifest lists {len(manifest['source_files_sha256'])}")
    tmp = args.spec.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(spec, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    tmp.replace(args.spec)
    print(f"Added {added} result files; suite now {len(spec['runs'])} files, "
          f"{sum(len(r['jobs']) for r in spec['runs'].values())} jobs, {len(supplementary)} supplementary")


if __name__ == "__main__":
    main()

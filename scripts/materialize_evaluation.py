#!/usr/bin/env python3
"""Build the portable thesis evaluation specification from the archived inputs.

This is a maintenance tool for the release specification. Replaying the suite
uses the committed specification directly and needs none of the audit inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from evaluation_config import architecture_config, load_yaml


def load(path):
    return json.loads(path.read_text())


def relative_checkpoint(value):
    if not value:
        return None
    # Historical checkpoint metadata may be absolute on either cluster.
    parts = Path(value).parts
    if "outputs" not in parts:
        raise ValueError(f"Unrecognized checkpoint path: {value}")
    return Path(*parts[parts.index("outputs"):]).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-manifest", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, action="append", required=True)
    parser.add_argument("--config-tree", type=Path, action="append", required=True,
                        help="Historical configs/experiment/paper_eval tree; repeat for each source")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.frozen.resolve()
    if args.output.resolve().is_relative_to(frozen):
        parser.error("Output must be outside the frozen snapshot")
    manifest = load(args.figure_manifest)
    with (frozen / "CHANGES.tsv").open(newline="") as f:
        changes = {(r["target"], r["mod"]): r["source"] for r in csv.DictReader(f, delimiter="\t")}
    cfgs = {}
    for tree in args.config_tree:
        for path in sorted(tree.rglob("*.yaml")):
            cfgs.setdefault(path.stem, []).append(path)

    def resolve_config(stem, config_root):
        paths = cfgs[stem]
        if len({p.read_bytes() for p in paths}) != 1:
            raise ValueError(f"Conflicting historical config: {stem}")
        path = paths[0]
        raw = load_yaml(path)
        config_tree = next(tree for tree in args.config_tree if path.is_relative_to(tree))
        config = architecture_config(raw, config_tree)
        config.setdefault("SEED", 44)
        config["ENV_NAME"] = config_root
        # The checkpoint inputs and output paths are set by the replay entry point.
        return {k: v for k, v in config.items() if not k.startswith("PAPER_EVAL_")
                and k not in {"SAVE_PATH", "MODEL_PATH", "SKILLS_PATH", "META_PATH"}}

    spec = {"schema": 1, "main_layout": manifest["main_layout"],
            "configs": {}, "runs": {}, "missing_queries": manifest["missing_queries"]}
    for source, expected_hash in sorted(manifest["source_files_sha256"].items()):
        name = Path(source).name
        path = frozen / "eval" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        data = load(path)
        run = {"sha256": expected_hash, "env": data["env"],
               "agent_kind": data["agent_kind"], "jobs": {}}
        spec["runs"][name] = run
        if data["agent_kind"] == "nexus":
            run["alg"] = data["alg"]
        for key, result in sorted(data["results"].items()):
            if key.startswith("swap_") or not result.get("returns"):
                continue  # Historical label-swap diagnostics are excluded from replay.
            origin = changes.get((name, key))
            actual = data
            if origin:
                actual = load(next(p / origin for p in args.rerun if (p / origin).is_file()))
                if actual["results"][key]["returns"] != result["returns"]:
                    raise ValueError(f"Rerun differs from frozen: {name} / {key}")
            kind = actual["agent_kind"]
            job = {k: actual[k] for k in ("base_seed", "num_episodes", "num_seeds", "max_frames")}
            job["kind"] = kind
            if kind == "nexus":
                job["checkpoint"] = relative_checkpoint(actual["ckpt"])
            else:
                stem = actual["config_stem"]
                if stem not in spec["configs"]:
                    spec["configs"][stem] = resolve_config(stem, actual["env"])
                job["config"] = stem
                job["object_centric"] = actual["object_centric"]
                for k in ("skills_path", "meta_path", "flat_pqn_path"):
                    val = actual.get(k)
                    if k == "skills_path" and origin:
                        val = actual.get("skills_real") or val
                    # The temporary rerun link for the selector preserves its name.
                    if origin and val and "outputs" not in Path(val).parts:
                        original = data.get(k)
                        if not original or Path(original).name != Path(val).name:
                            raise ValueError(f"Cannot resolve rerun checkpoint: {name} / {key}: {val}")
                        val = original
                    if val:
                        job[k] = relative_checkpoint(val)
                job["mod"] = next((m for m in actual["mods"] if m["key"] == key), None)
                if job["mod"] is None:
                    allowlists = [tree / "mods" / f"{actual['env']}.yaml" for tree in args.config_tree]
                    for allowlist in allowlists:
                        if allowlist.is_file():
                            entries = yaml.safe_load(allowlist.read_text())["eval_mods"]
                            job["mod"] = next((m for m in entries if m["key"] == key), None)
                            if job["mod"] is not None:
                                break
                    if job["mod"] is None:
                        raise ValueError(f"Missing mod specification: {name} / {key}")
            if origin:
                job["rerun_source"] = origin
            run["jobs"][key] = job
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(spec, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    print(f"{len(spec['runs'])} result files; {sum(len(r['jobs']) for r in spec['runs'].values())} jobs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce SCOPE thesis evaluation from the frozen data or trained checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/reproduce/thesis.json"
ANALYSIS = ROOT / "analysis"
RESCUE = "rescue_metrics_20261112"
CHECKPOINT_FIELDS = ("skills_path", "meta_path", "flat_pqn_path", "checkpoint")


def read(path):
    return json.loads(path.read_text())


def frozen_data(path):
    return path / "eval" if (path / "eval").is_dir() else path


def rescue_data(results, override=None):
    """eval/ of the 2026-11-12 rescue-metric rerun, shipped next to the frozen snapshot."""
    return (override or results.parent.parent / RESCUE / "eval").resolve()


def verify_archive(directory, spec, rescue=None):
    for name, run in spec["runs"].items():
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["sha256"]:
            raise ValueError(f"Frozen result hash mismatch: {path}")
    print(f"Verified {len(spec['runs'])} frozen result files", flush=True)
    if rescue is not None:
        files = spec["supplementary"][RESCUE]["files"]
        for name, digest in files.items():
            if hashlib.sha256((rescue / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Rescue-metric hash mismatch: {rescue / name}")
        print(f"Verified {len(files)} rescue-metric files", flush=True)


def worker(request_path):
    """Internal entry point: a single checkpoint/condition in a fresh process."""
    request = read(request_path)
    run, job, mod = request["run"], request["job"], request["mod"]
    output = Path(request["output"])
    protocol = {key: job[key] for key in ("base_seed", "num_episodes", "num_seeds", "max_frames")}
    if job["kind"] == "nexus":
        from nexus.mod_eval import run_nexus_mod_eval

        protocol["seed"] = protocol.pop("base_seed")
        payload = run_nexus_mod_eval(
            nexus_root=Path(request["nexus_root"]), jaxatari_root=None,
            alg=run["alg"], ckpt_dir=Path(job["checkpoint"]),
            mod_keys=[mod], output_json=output, **protocol,
        )
    else:
        from importlib import import_module
        import inspect

        import numpy as np
        from jaxtari.core import GAME_MODULES
        from jaxtari.environment import JaxEnvironment
        from concept_based_learning.paper_eval.eval_harness import run_mod_eval
        from concept_based_learning.paper_eval.mods import ModSpec

        # The thesis worker used host-side action tables before each evaluation.
        module = import_module(GAME_MODULES[run["env"]])
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is not JaxEnvironment and issubclass(cls, JaxEnvironment) and hasattr(cls, "ACTION_SET"):
                cls.ACTION_SET = np.asarray(cls.ACTION_SET, dtype=np.int32)
        spec = job["mod"]
        mod_spec = ModSpec(spec["key"], tuple(spec["mods"]), spec["category"], spec.get("relabel", False))
        payload = run_mod_eval(
            agent_kind=job["kind"], config=request["config"], env_name=run["env"],
            object_centric=job["object_centric"], mod_specs=[mod_spec],
            publish_report=False, wandb_mode="disabled", **protocol,
            **{key: job[key] for key in CHECKPOINT_FIELDS[:-1] if key in job},
        )
    result = payload["results"][mod]
    output.write_text(json.dumps(result, indent=2) + "\n")


def choose_jobs(spec, env, match, condition, exclude_external):
    return [(name, run, mod, job) for name, run in spec["runs"].items()
            if (not env or run["env"] == env) and fnmatch.fnmatch(name, match)
            for mod, job in run["jobs"].items()
            if (not condition or mod == condition) and (not exclude_external or job["kind"] != "nexus")]


def evaluate(args, spec, results):
    jobs = choose_jobs(spec, args.env, args.match, args.condition, args.exclude_external)
    if not jobs:
        raise ValueError("No evaluation jobs selected")
    paths = sorted({job[key] for _, _, _, job in jobs for key in CHECKPOINT_FIELDS if key in job})
    print(f"{len(jobs)} fresh processes; {len(paths)} checkpoint paths; "
          f"{sum(j['num_episodes'] * j['num_seeds'] for _, _, _, j in jobs)} episodes")
    print("Evaluators:", dict(Counter(job["kind"] for _, _, _, job in jobs)), flush=True)
    if args.dry_run:
        return 0
    if args.reference is None:
        raise ValueError("--reference is required to verify replayed episode returns")
    if args.checkpoint_root is None and not all(job.get("checkpoint_in_archive") for _, _, _, job in jobs):
        raise ValueError("--checkpoint-root is required unless only archive-bundled (J) checkpoints are replayed")
    base = args.checkpoint_root.resolve() if args.checkpoint_root else None
    archive = results.parent  # J checkpoints ship inside the frozen snapshot

    def root(job):
        return archive if job.get("checkpoint_in_archive") else base

    absent = [job[key] for _, _, _, job in jobs for key in CHECKPOINT_FIELDS
              if key in job and not (root(job) / job[key]).exists()]
    if absent:
        absent = sorted(set(absent))
        raise FileNotFoundError(f"{len(absent)} checkpoint paths missing under {base}: {', '.join(absent[:5])}")
    external = any(job["kind"] == "nexus" for _, _, _, job in jobs)
    if external and (args.nexus_root is None or not (args.nexus_root / "main.py").is_file()):
        raise ValueError("NEXUS/HRL replay needs --nexus-root; use --exclude-external for SCOPE and PQN only")
    verify_archive(results, spec)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    differences = []
    with tempfile.TemporaryDirectory(prefix="scope-replay-") as temp:
        scratch = Path(temp)
        nexus = scratch / "symbolic_options"
        if external:
            # The legacy baseline integration applies compatibility fixes to its copy.
            shutil.copytree(args.nexus_root, nexus, ignore=shutil.ignore_patterns(".git", ".venv", "outputs", "wandb", "__pycache__"))
            from run_nexus import _sync_alg_configs

            _sync_alg_configs(nexus)
        for i, (name, run, mod, job) in enumerate(jobs, start=1):
            out = args.output / Path(name).stem / f"{mod}.json"
            if not out.is_file() or not args.resume:
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp_out = out.with_suffix(".partial.json")
                request = scratch / "request.json"
                resolved = {k: str(root(job) / v) if k in CHECKPOINT_FIELDS else v for k, v in job.items()}
                request.write_text(json.dumps({"run": {k: v for k, v in run.items() if k != "jobs"},
                                               "job": resolved, "mod": mod, "output": str(tmp_out),
                                               "nexus_root": str(nexus),
                                               "config": spec["configs"].get(job.get("config"))}))
                env = os.environ | {"WANDB_MODE": "disabled", "CBL_EVAL_MOD_WORKER": "1"}
                print(f"[{i}/{len(jobs)}] {name} / {mod}", flush=True)
                subprocess.run([sys.executable, __file__, "_worker", str(request)],
                               cwd=ROOT, env=env, check=True)
                tmp_out.replace(out)
            expected = read(results / name)["results"][mod]["returns"]
            if read(out)["returns"] != expected:
                differences.append({"file": name, "condition": mod, "output": str(out)})
    report = {"jobs": len(jobs), "matched": len(jobs) - len(differences), "mismatched": differences}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Matched {report['matched']}/{len(jobs)} episode-return vectors")
    return int(bool(differences))


def figures(args, spec, results):
    rescue = rescue_data(results, args.rescue)
    verify_archive(results, spec, rescue)
    args.output.mkdir(parents=True)
    assets, tables = args.output / "assets", args.output / "tables"
    shutil.copytree(ANALYSIS / "illustrations", assets)
    cmd = [sys.executable, str(ANALYSIS / "generate.py"), "--results-root", str(results),
           "--controls-root", str(results), "--control-overrides", str(ANALYSIS / "no-overrides"),
           "--rescue-root", str(rescue), "--main-layout", spec["main_layout"], "--output", str(assets), "--tables-output", str(tables)]
    subprocess.run(cmd, check=True, env=os.environ | {"MPLCONFIGDIR": str(args.output / ".matplotlib")})
    subprocess.run([sys.executable, str(ANALYSIS / "verify.py"), "--output", str(assets),
                    "--tables-output", str(tables)], check=True)
    return 0


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_worker":
        worker(Path(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify", "figures", "evaluate"), default="verify")
    parser.add_argument("--reference", type=Path, help="Extracted frozen snapshot or its eval directory")
    parser.add_argument("--rescue", type=Path, help=f"Override the {RESCUE}/eval directory next to the snapshot")
    parser.add_argument("--checkpoint-root", type=Path, help="Directory containing the checkpoint outputs tree")
    parser.add_argument("--nexus-root", type=Path, help="Optional external symbolic_options checkout")
    parser.add_argument("--output", type=Path, default=Path("scope-reproduction"))
    parser.add_argument("--env", help="Filter replay by target environment")
    parser.add_argument("--match", default="*", help="Glob matching archived result filenames")
    parser.add_argument("--condition", help="Replay only this mod key")
    parser.add_argument("--exclude-external", action="store_true", help="Skip NEXUS and HRL")
    parser.add_argument("--dry-run", action="store_true", help="Print replay workload without evaluating")
    parser.add_argument("--resume", action="store_true", help="Reuse complete results in the chosen output")
    args = parser.parse_args()
    args.output = args.output.resolve()
    results = frozen_data(args.reference.resolve()) if args.reference else None
    if args.output.name.startswith("frozen_") or any(p.name.startswith("frozen_") for p in args.output.parents):
        parser.error("Output must be outside frozen results")
    if results and (args.output.is_relative_to(results) or results.is_relative_to(args.output)):
        parser.error("Input and output directories must be disjoint")
    spec = read(SPEC)
    if args.stage == "evaluate":
        return evaluate(args, spec, results)
    if results is None:
        parser.error("--reference is required")
    if args.stage == "figures":
        return figures(args, spec, results)
    verify_archive(results, spec, rescue_data(results, args.rescue))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

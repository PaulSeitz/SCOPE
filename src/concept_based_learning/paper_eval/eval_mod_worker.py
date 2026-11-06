"""One mod of a skill eval, in its own process.

Invoked by ``eval_harness._run_skill_mods_in_subprocesses``. Do not set
``EVAL_PARALLEL_MODS`` off here: the parent already did, and
``CBL_EVAL_MOD_WORKER=1`` stops another spawn.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from concept_based_learning.paper_eval.eval_harness import run_hierarchical_mod_eval
from concept_based_learning.paper_eval.mods import ModSpec
from concept_based_learning.pqn_visualizer import load_params_from_file


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(f"usage: python -m concept_based_learning.paper_eval.eval_mod_worker JOB.pkl")
    job = pickle.loads(Path(args[0]).read_bytes())
    mod = job["mod"]
    spec = ModSpec(
        key=str(mod["key"]),
        mods=tuple(mod.get("mods") or ()),
        category=str(mod.get("category") or ""),
        relabel=bool(mod.get("relabel")),
    )
    params = load_params_from_file(job["skills_path"])
    meta = job.get("meta_path")
    meta_params = load_params_from_file(meta) if meta and Path(meta).is_file() else None
    results = run_hierarchical_mod_eval(
        job["config"],
        params_tree=params,
        meta_params=meta_params,
        mod_specs=[spec],
        num_episodes=int(job["num_episodes"]),
        num_seeds=int(job["num_seeds"]),
        base_seed=int(job["base_seed"]),
        max_frames=int(job["max_frames"]),
    )
    Path(job["out"]).write_text(json.dumps(results), encoding="utf-8")
    print(f"  worker wrote {spec.key}", flush=True)


if __name__ == "__main__":
    main()

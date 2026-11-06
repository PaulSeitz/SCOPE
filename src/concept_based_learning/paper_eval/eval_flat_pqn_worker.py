"""One mod of a flat PQN eval, in its own process.

Same greedy eval as the serial path. ``CBL_EVAL_MOD_WORKER=1`` is set by the
parent so this process does not spawn further workers.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from concept_based_learning.baselines.flat_pqn import run_flat_pqn_greedy_eval


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit(
            "usage: python -m concept_based_learning.paper_eval.eval_flat_pqn_worker JOB.pkl"
        )
    job = pickle.loads(Path(args[0]).read_bytes())
    stats = run_flat_pqn_greedy_eval(
        model_path=str(job["model_path"]),
        game_mods=list(job.get("game_mods") or []),
        num_episodes=int(job["num_episodes"]),
        num_seeds=int(job["num_seeds"]),
        base_seed=int(job["base_seed"]),
        max_frames=int(job["max_frames"]),
        object_centric=bool(job.get("object_centric", True)),
        env_name=str(job["env_name"]),
        progress=True,
    )
    Path(job["out"]).write_text(json.dumps(stats), encoding="utf-8")
    print(f"  worker wrote {Path(args[0]).stem}", flush=True)


if __name__ == "__main__":
    main()

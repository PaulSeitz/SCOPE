"""One NEXUS / HRL mod eval, in its own process.

The parent sets ``CBL_EVAL_MOD_WORKER=1`` so this call stays on the serial path.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m nexus.mod_eval_worker JOB.pkl")
    job = pickle.loads(Path(args[0]).read_bytes())
    from nexus.mod_eval import run_nexus_mod_eval

    run_nexus_mod_eval(
        nexus_root=Path(job["nexus_root"]),
        jaxatari_root=Path(job["jaxatari_root"]),
        alg=str(job["alg"]),
        ckpt_dir=Path(job["ckpt_dir"]),
        seed=int(job["seed"]),
        num_episodes=int(job["num_episodes"]),
        num_seeds=int(job["num_seeds"]),
        max_frames=int(job["max_frames"]),
        trial=bool(job["trial"]),
        mod_keys=[str(job["mod_key"])],
        output_json=Path(job["out"]),
        use_vendored_seaquest=bool(job["use_vendored_seaquest"]),
    )
    print(f"  worker wrote {job['mod_key']}", flush=True)


if __name__ == "__main__":
    main()

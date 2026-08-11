"""Paper evaluation helpers (mod allowlists, checkpoints, greedy mod eval)."""

from concept_based_learning.paper_eval.env_metrics import EPISODE_METRIC_KEYS
from concept_based_learning.paper_eval.mods import load_mod_allowlist, mod_specs_for_env

__all__ = [
    "EPISODE_METRIC_KEYS",
    "load_mod_allowlist",
    "mod_specs_for_env",
    "resolve_rq1_skill_checkpoint",
    "resolve_flat_pqn_checkpoint",
    "run_mod_eval",
    "write_mod_eval_csvs",
]


def __getattr__(name: str):
    # Lazy imports avoid circular deps (pqn_visualizer ↔ eval_harness ↔ flat_pqn).
    if name == "resolve_rq1_skill_checkpoint":
        from concept_based_learning.paper_eval.checkpoints import (
            resolve_rq1_skill_checkpoint,
        )

        return resolve_rq1_skill_checkpoint
    if name == "resolve_flat_pqn_checkpoint":
        from concept_based_learning.paper_eval.checkpoints import (
            resolve_flat_pqn_checkpoint,
        )

        return resolve_flat_pqn_checkpoint
    if name == "run_mod_eval":
        from concept_based_learning.paper_eval.eval_harness import run_mod_eval

        return run_mod_eval
    if name == "write_mod_eval_csvs":
        from concept_based_learning.paper_eval.csv_export import write_mod_eval_csvs

        return write_mod_eval_csvs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

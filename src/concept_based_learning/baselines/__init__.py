"""Baseline agents used for comparison (e.g. flat PQN from JAXAtari benchmarks).

Imports are lazy so ``from concept_based_learning.baselines.fixed_ego_skills import ...``
does not pull in ``flat_pqn`` (which imports ``pqn_visualizer`` and would cycle).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CNN",
    "QNetwork",
    "FlatPQNCNN",
    "FlatPQNQNetwork",
    "load_flat_pqn_params",
    "run_flat_pqn_greedy_eval",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from concept_based_learning.baselines import flat_pqn

        return getattr(flat_pqn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

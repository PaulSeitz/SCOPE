"""Resolve paper_eval checkpoint paths (RQ1 skills / flat PQN)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PAPER_EVAL_ROOT = PROJECT_ROOT / "outputs" / "models" / "paper_eval"


def resolve_rq1_skill_checkpoint(
    stem: str = "seaquest_skills_ego",
    *,
    env_name: str = "seaquest",
    seed: int = 44,
    save_root: str | Path | None = None,
) -> Path:
    """
    ``skills_only`` save layout from ``main.py``:
    ``{SAVE_PATH}/{env}/hierarchical_pqn_{env}_seed{N}.safetensors``
    """
    root = Path(save_root) if save_root else PAPER_EVAL_ROOT / "rq1" / stem
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / env_name / f"hierarchical_pqn_{env_name}_seed{seed}.safetensors"


def resolve_flat_pqn_checkpoint(
    *,
    object_centric: bool,
    env_name: str = "seaquest",
    seed: int = 44,
    save_root: str | Path | None = None,
    vmap_index: int = 0,
) -> Path:
    """
    Flat PQN save layout from ``baselines.flat_pqn_train``:
    ``{SAVE_PATH}/{env}/flat_pqn_{env}_{oc|pixel}_seed{N}_vmap{i}.safetensors``
    """
    root = Path(save_root) if save_root else PAPER_EVAL_ROOT / "rq1" / "seaquest_pqn"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    oc = "oc" if object_centric else "pixel"
    return (
        root
        / env_name
        / f"flat_pqn_{env_name}_{oc}_seed{seed}_vmap{vmap_index}.safetensors"
    )

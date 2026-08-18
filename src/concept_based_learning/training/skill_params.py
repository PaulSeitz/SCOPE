"""Helpers for stacked vs per-skill (heterogeneous input) Q-network params."""

from __future__ import annotations

from typing import Any

import numpy as np
from flax.traverse_util import flatten_dict


def as_per_skill_param_list(params_tree: Any) -> list | None:
    """Return a list of per-skill param dicts, or None if params are stacked/shared.

    Packed checkpoints use keys ``0`` / ``"0"`` … for each skill. Legacy files
    keep a stacked ``Dense_0.kernel`` with a leading skill axis.
    """
    if params_tree is None:
        return None
    if isinstance(params_tree, (tuple, list)):
        if params_tree and isinstance(params_tree[0], dict) and "Dense_0" in params_tree[0]:
            return list(params_tree)
        return None
    if not isinstance(params_tree, dict):
        return None
    if "Dense_0" in params_tree:
        return None
    keys: list = []
    for i in range(32):
        if i in params_tree:
            keys.append(i)
        elif str(i) in params_tree:
            keys.append(str(i))
        else:
            break
    if len(keys) < 2:
        return None
    packed = [params_tree[k] for k in keys]
    if packed and isinstance(packed[0], dict) and "Dense_0" in packed[0]:
        return packed
    return None


def flatten_skill_params(params: Any) -> dict:
    """Flatten stacked or per-skill params for safetensors."""
    packed = as_per_skill_param_list(params)
    if packed is not None:
        return flatten_dict({str(i): p for i, p in enumerate(packed)}, sep=",")
    return flatten_dict(params, sep=",")


def dense0_in_dim(params: Any) -> int:
    """First-layer input width of a single-skill param dict."""
    return int(np.asarray(params["Dense_0"]["kernel"]).shape[0])

"""Validate evaluation artifacts before treating a NEXUS batch job as complete."""
from __future__ import annotations

import json
import math
from pathlib import Path


def evaluation_complete(path, *, mod_keys=None, num_episodes=None, num_seeds=None):
    try:
        payload = json.loads(Path(path).read_text())
        if payload.get("status") == "failed":
            return False
        episodes = num_episodes or int(payload["num_episodes"])
        seeds = num_seeds or int(payload.get("num_seeds", 1))
        expected = episodes * seeds
        results = payload["results"]
        keys = mod_keys if mod_keys is not None else list(results)
        if not keys or expected <= 0:
            return False
        for key in keys:
            stats = results[key]
            if stats.get("error") or stats["n_episodes"] != expected:
                return False
            if not math.isfinite(float(stats["mean_return"])):
                return False
            returns = stats["returns"]
            if len(returns) != expected or not all(math.isfinite(float(x)) for x in returns):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False

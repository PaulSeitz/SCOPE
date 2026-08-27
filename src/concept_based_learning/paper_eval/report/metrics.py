"""Aggregation helpers for paper_eval retention tables."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Collection

import numpy as np

# Easy / no-threat mods where weak bases inflate retention.
POINT_FARMING_MODS: frozenset[str] = frozenset({"disable_enemies", "no_enemies"})


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    rq: int | None = None,
    env: str | None = None,
    exclude_base: bool = False,
    protocols: set[str] | None = None,
    labels: set[str] | None = None,
    exclude_mods: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if rq is not None:
        out = [r for r in out if int(r.get("paper_eval_rq") or 0) == int(rq)]
    if env is not None:
        env_l = env.lower()
        out = [r for r in out if r.get("env") == env_l]
    if exclude_base:
        out = [r for r in out if r.get("mod_key") != "base"]
    if protocols is not None:
        out = [r for r in out if r.get("protocol") in protocols]
    if labels is not None:
        out = [r for r in out if r.get("agent_label") in labels]
    if exclude_mods is not None:
        skip = set(exclude_mods)
        out = [r for r in out if r.get("mod_key") not in skip]
    return out


def split_point_farming(
    rows: list[dict[str, Any]],
    *,
    farming_mods: Collection[str] = POINT_FARMING_MODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    farming_set = set(farming_mods)
    main = [r for r in rows if r.get("mod_key") not in farming_set]
    farming = [
        r
        for r in rows
        if r.get("mod_key") in farming_set or r.get("mod_key") == "base"
    ]
    return main, farming


def base_returns_by_agent(rows: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """Return [(agent_label, mean, std), ...] for mod_key==base, sorted."""
    found: dict[str, tuple[float, float]] = {}
    for row in rows:
        if row.get("mod_key") != "base":
            continue
        label = str(row["agent_label"])
        found[label] = (float(row["mean"]), float(row.get("std") or 0.0))
    return sorted(found.items(), key=lambda kv: kv[0])


def mean_retention_by_category(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    agent_label → {category → mean retention over mods in that category}.

    Excludes ``base``. Categories with no finite retentions are omitted.
    """
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("mod_key") == "base":
            continue
        ret = row.get("retention")
        if ret is None or ret != ret:
            continue
        buckets[str(row["agent_label"])][str(row["mod_category"] or "other")].append(
            float(ret)
        )
    return {
        agent: {cat: sum(vals) / len(vals) for cat, vals in cats.items() if vals}
        for agent, cats in buckets.items()
    }


def retention_matrix(
    rows: list[dict[str, Any]],
    *,
    agents: list[str] | None = None,
    mods: list[str] | None = None,
) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Build (mod_keys, agent_labels, matrix[mod][agent]) of retention.

    ``base`` is excluded from rows. Missing cells are NaN.
    """
    by_pair: dict[tuple[str, str], float] = {}
    mod_set: set[str] = set()
    agent_set: set[str] = set()
    for row in rows:
        if row.get("mod_key") == "base":
            continue
        mod = str(row["mod_key"])
        agent = str(row["agent_label"])
        ret = row.get("retention")
        if ret is None or ret != ret:
            continue
        by_pair[(mod, agent)] = float(ret)
        mod_set.add(mod)
        agent_set.add(agent)

    mod_list = mods if mods is not None else sorted(mod_set)
    agent_list = agents if agents is not None else sorted(agent_set)
    matrix: list[list[float]] = []
    for mod in mod_list:
        matrix.append(
            [by_pair.get((mod, agent), float("nan")) for agent in agent_list]
        )
    return mod_list, agent_list, matrix


def mean_retention_by_agent(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Overall mean retention (all non-base mods) per agent."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("mod_key") == "base":
            continue
        ret = row.get("retention")
        if ret is None or ret != ret:
            continue
        buckets[str(row["agent_label"])].append(float(ret))
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def median_retention_by_agent(
    rows: list[dict[str, Any]],
    *,
    category: str | None = None,
    exclude_mods: Collection[str] | None = POINT_FARMING_MODS,
) -> dict[str, float]:
    skip = set(exclude_mods or ())
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("mod_key") == "base" or row.get("mod_key") in skip:
            continue
        if category is not None and str(row.get("mod_category") or "") != category:
            continue
        ret = row.get("retention")
        if ret is None or ret != ret:
            continue
        buckets[str(row["agent_label"])].append(float(ret))
    return {
        agent: float(np.median(np.asarray(vals, dtype=float)))
        for agent, vals in buckets.items()
        if vals
    }


def transfer_base_matrix(
    rows: list[dict[str, Any]],
    *,
    envs: list[str] | None = None,
    modes: list[str] | None = None,
) -> tuple[list[str], list[str], list[list[float]]]:
    env_list = envs or ["pong", "tennis", "beamrider", "freeway", "asterix", "phoenix"]
    mode_list = modes or ["ego", "pixel", "oc", "pqn"]
    by_pair: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("mod_key") != "base":
            continue
        label = str(row.get("agent_label") or "")
        if not label.startswith("transfer/"):
            continue
        mode = label.split("/", 1)[1]
        env = str(row.get("env") or "")
        mean = row.get("mean")
        if mean is None or mean != mean:
            continue
        by_pair[(env, mode)] = float(mean)
    matrix = [
        [by_pair.get((env, mode), float("nan")) for mode in mode_list]
        for env in env_list
    ]
    return env_list, mode_list, matrix


def envs_in_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(r.get("env") or "") for r in rows if r.get("env")})

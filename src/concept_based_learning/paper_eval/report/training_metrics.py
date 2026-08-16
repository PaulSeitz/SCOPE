"""Load training_metrics.csv peaks for train-vs-eval report figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


_GAME_RETURN_COLS = (
    "charts/avg_game_return",
    "returned_episode_returns",
    "returned_game_returns",
)


def _stem_to_agent_label(stem: str) -> str | None:
    s = stem.lower()
    if "heuristic" in s:
        return "heuristic"
    if "pqn" in s and ("_oc" in s or s.endswith("oc") or "/oc" in s):
        return "pqn/oc"
    if "pqn" in s:
        return "pqn/pixel"
    if "tandem" in s:
        if "pixel" in s:
            return "tandem/pixel"
        if "_oc" in s or "oc_" in s:
            return "tandem/oc"
        return "tandem/mixed"
    if "meta" in s and "skills" not in s.split("_")[0:2]:
        # seaquest_meta_pixel_ego_skills etc.
        if "pixel" in s:
            return "meta/pixel"
        if "_oc" in s or "oc_" in s:
            return "meta/oc"
        return "meta/mixed"
    # skills / transfer / finetune / scratch
    tag = "ego"
    if "pixel" in s:
        tag = "pixel"
    elif "_oc" in s or s.endswith("_oc") or "_oc_" in s:
        tag = "oc"
    if "transfer" in s:
        return f"transfer/{tag}"
    if "finetune" in s:
        return f"finetune/{tag}"
    if "from_scratch" in s or "scratch" in s:
        return f"scratch/{tag}"
    if "skills" in s or s.startswith("seaquest_"):
        return f"skills/{tag}"
    return None


def _peak_from_csv(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    series: list[tuple[float, float]] = []  # (step_or_update, value)
    for i, row in enumerate(rows):
        val = None
        for col in _GAME_RETURN_COLS:
            raw = row.get(col, "")
            if raw in ("", None):
                continue
            try:
                val = float(raw)
                break
            except (TypeError, ValueError):
                continue
        if val is None or val != val:
            continue
        step_raw = row.get("env/step") or row.get("update_steps") or str(i)
        try:
            step = float(step_raw)
        except (TypeError, ValueError):
            step = float(i)
        series.append((step, val))
    if not series:
        return None
    peak_step, peak_val = max(series, key=lambda t: t[1])
    last_val = series[-1][1]
    return {
        "peak": float(peak_val),
        "peak_step": float(peak_step),
        "last": float(last_val),
        "n_rows": len(series),
        "path": str(path),
    }


def discover_training_peaks(
    root: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Scan ``root`` for ``**/training_metrics.csv`` and return
    ``agent_label → {peak, last, path, stem, ...}``.

    When multiple CSVs map to the same label, keep the highest peak.
    """
    root = Path(root)
    if not root.is_dir():
        return {}

    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("training_metrics.csv")):
        # .../rq1/seaquest_pqn/seaquest/training_metrics.csv → seaquest_pqn
        stem = path.parent.parent.name if path.parent.parent != path.parent else path.parent.name
        label = _stem_to_agent_label(stem)
        if label is None:
            continue
        peak = _peak_from_csv(path)
        if peak is None:
            continue
        peak["stem"] = stem
        peak["agent_label"] = label
        prev = found.get(label)
        if prev is None or float(peak["peak"]) > float(prev["peak"]):
            found[label] = peak
    return found


def train_eval_pairs(
    eval_rows: list[dict[str, Any]],
    train_peaks: dict[str, dict[str, Any]],
    *,
    env: str | None = "seaquest",
    rq: int | None = 1,
) -> list[dict[str, Any]]:
    """Join eval base means with training peaks for the same agent_label."""
    from concept_based_learning.paper_eval.report.metrics import (
        base_returns_by_agent,
        filter_rows,
    )

    rows = filter_rows(eval_rows, rq=rq, env=env)
    bases = dict(base_returns_by_agent(rows))
    out: list[dict[str, Any]] = []
    labels = sorted(set(bases) | set(train_peaks))
    for label in labels:
        base = bases.get(label)
        peak = train_peaks.get(label)
        if base is None and peak is None:
            continue
        eval_mean = float(base[0]) if base is not None else float("nan")
        eval_std = float(base[1]) if base is not None else float("nan")
        train_peak = float(peak["peak"]) if peak is not None else float("nan")
        gap = (
            train_peak - eval_mean
            if train_peak == train_peak and eval_mean == eval_mean
            else float("nan")
        )
        ratio = (
            eval_mean / train_peak
            if train_peak == train_peak and abs(train_peak) > 1e-8 and eval_mean == eval_mean
            else float("nan")
        )
        out.append(
            {
                "agent_label": label,
                "train_peak": train_peak,
                "eval_mean": eval_mean,
                "eval_std": eval_std,
                "gap": gap,
                "eval_over_train": ratio,
                "train_path": peak.get("path") if peak else "",
                "train_stem": peak.get("stem") if peak else "",
            }
        )
    return out

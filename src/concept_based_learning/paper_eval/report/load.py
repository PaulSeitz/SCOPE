"""Load and normalize paper_eval mod-eval summary tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_agent_label(row: dict[str, Any]) -> str:
    """Human-readable variant label used as the plot legend / heatmap column."""
    kind = str(row.get("paper_eval_kind") or row.get("agent_kind") or "").lower().strip()
    mode = str(row.get("skill_input_mode") or "").lower().strip()
    stem = str(row.get("config_stem") or row.get("run_name") or "").lower()
    oc = str(row.get("object_centric") or "").strip().lower() in {"1", "true", "yes"}

    if kind in {"heuristic_random", "heuristic"} or "heuristic" in stem:
        return "heuristic"
    if kind in {"flat_pqn"} or "pqn" in stem:
        if oc or mode == "oc_embed" or "_oc" in stem:
            return "pqn/oc"
        return "pqn/pixel"
    if kind in {"meta"}:
        return f"meta/{mode or 'mixed'}"
    if kind in {"tandem"}:
        return f"tandem/{mode or 'mixed'}"
    if kind in {"transfer"}:
        tag = mode or "ego"
        if tag in {"ego", "ego_embed"}:
            tag = "ego"
        elif tag.startswith("pixel"):
            tag = "pixel"
        elif tag.startswith("oc"):
            tag = "oc"
        return f"transfer/{tag}"
    # skills / hierarchical / finetune / from_scratch
    tag = mode or "ego"
    if tag in {"ego", "ego_embed"}:
        tag = "ego"
    elif tag.startswith("pixel"):
        tag = "pixel"
    elif tag.startswith("oc"):
        tag = "oc"
    if "finetune" in stem:
        return f"finetune/{tag}"
    if "from_scratch" in stem or "scratch" in stem:
        return f"scratch/{tag}"
    return f"skills/{tag}"


def _infer_protocol(row: dict[str, Any]) -> str:
    """Coarse protocol bucket: base | transfer | finetune | scratch | meta | tandem | heuristic | pqn."""
    label = _infer_agent_label(row)
    if label == "heuristic":
        return "heuristic"
    if label.startswith("pqn/"):
        return "pqn"
    if label.startswith("meta/"):
        return "meta"
    if label.startswith("tandem/"):
        return "tandem"
    if label.startswith("transfer/"):
        return "transfer"
    if label.startswith("finetune/"):
        return "finetune"
    if label.startswith("scratch/"):
        return "scratch"
    return "skills"


def normalize_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["mean"] = _to_float(row.get("mean"))
    out["std"] = _to_float(row.get("std"), 0.0)
    out["drop_from_base"] = _to_float(row.get("drop_from_base"))
    out["drop_frac_from_base"] = _to_float(row.get("drop_frac_from_base"))
    base_mean = _to_float(row.get("base_mean"))
    # Retention: prefer explicit; else derive from drop_frac; else nan until joined.
    ret = _to_float(row.get("retention"), default=float("nan"))
    if ret != ret:  # NaN
        dfrac = out["drop_frac_from_base"]
        if dfrac == dfrac:
            ret = 1.0 + dfrac
    out["retention"] = ret
    out["base_mean"] = base_mean
    try:
        out["paper_eval_rq"] = int(float(row.get("paper_eval_rq") or 0))
    except (TypeError, ValueError):
        out["paper_eval_rq"] = 0
    out["env"] = str(row.get("env") or "").lower()
    out["mod_key"] = str(row.get("mod_key") or "")
    out["mod_category"] = str(row.get("mod_category") or "other") or "other"
    out["config_stem"] = str(row.get("config_stem") or row.get("run_name") or "")
    out["agent_label"] = str(row.get("agent_label") or _infer_agent_label(out))
    out["protocol"] = str(row.get("protocol") or _infer_protocol(out))
    return out


def _attach_base_and_retention(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join each row to its same-run base mean and fill retention if missing."""
    base_by_run: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["mod_key"] != "base":
            continue
        key = (row["config_stem"], row["env"], row["agent_label"])
        if row["mean"] == row["mean"]:
            base_by_run[key] = float(row["mean"])

    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["config_stem"], row["env"], row["agent_label"])
        base = base_by_run.get(key, row.get("base_mean", float("nan")))
        row = dict(row)
        row["base_mean"] = float(base) if base == base else float("nan")
        if row["mod_key"] == "base":
            row["retention"] = 1.0
            row["drop_frac_from_base"] = 0.0
            row["drop_from_base"] = 0.0
        elif row["retention"] != row["retention"]:
            if row["base_mean"] == row["base_mean"] and abs(row["base_mean"]) > 1e-8:
                row["retention"] = float(row["mean"]) / abs(float(row["base_mean"]))
            elif row["drop_frac_from_base"] == row["drop_frac_from_base"]:
                row["retention"] = 1.0 + float(row["drop_frac_from_base"])
        out.append(row)
    return out


def load_summary_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [normalize_summary_row(r) for r in reader]
    return _attach_base_and_retention(rows)


def load_summary_json(path: Path) -> list[dict[str, Any]]:
    """Load a single ``*_mods.json`` payload into summary-shaped rows."""
    from concept_based_learning.paper_eval.csv_export import results_to_summary_rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [normalize_summary_row(r) for r in results_to_summary_rows(payload)]
    return _attach_base_and_retention(rows)


def load_mod_eval_summary(
    source: str | Path,
    *,
    rq: int | None = None,
    env: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load mod-eval summary rows from:
      - ``all_mod_eval_summary.csv``
      - a single ``*_summary.csv``
      - a single ``*_mods.json``
      - a directory (prefers master CSV, else globs ``*_summary.csv`` / ``*_mods.json``)
    """
    path = Path(source)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        if path.suffix.lower() == ".json":
            rows = load_summary_json(path)
        else:
            rows = load_summary_csv(path)
    elif path.is_dir():
        master = path / "all_mod_eval_summary.csv"
        if master.is_file() and master.stat().st_size > 0:
            rows = load_summary_csv(master)
        else:
            summaries = sorted(path.glob("*_summary.csv"))
            if summaries:
                for p in summaries:
                    rows.extend(load_summary_csv(p))
            else:
                for p in sorted(path.glob("*_mods.json")):
                    rows.extend(load_summary_json(p))
            rows = _attach_base_and_retention(rows)
    else:
        raise FileNotFoundError(f"No mod-eval summary at {path}")

    if rq is not None:
        rows = [r for r in rows if int(r.get("paper_eval_rq") or 0) == int(rq)]
    if env is not None:
        env_l = env.lower()
        rows = [r for r in rows if r.get("env") == env_l]
    return rows

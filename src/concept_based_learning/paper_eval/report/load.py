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


def _short_input_mode(mode: str) -> str:
    m = str(mode or "").lower().strip()
    if m in {"ego", "ego_embed", "ego_embedding"}:
        return "ego"
    if m.startswith("pixel") or m in {"cnn"}:
        return "pixel"
    if m.startswith("oc") or m in {"object_embed", "object_embedding"}:
        return "oc"
    return m


def _infer_meta_input_mode(row: dict[str, Any]) -> str:
    """Meta/tandem observation mode (not skill input)."""
    explicit = _short_input_mode(str(row.get("meta_input_mode") or ""))
    if explicit:
        return explicit
    stem = str(row.get("config_stem") or row.get("run_name") or "").lower()
    if "meta_pixel" in stem or "tandem_pixel" in stem:
        return "pixel"
    if "meta_oc" in stem or "tandem_oc" in stem:
        return "oc"
    return ""


def _infer_agent_label(row: dict[str, Any]) -> str:
    """Human-readable variant label used as the plot legend / heatmap column.

    Naming cheat-sheet
    ------------------
    ``skills/{ego|pixel|oc}``
        Hierarchical skills + symbolic heuristic (in-domain Seaquest).
    ``transfer/{ego|pixel|oc}``
        Same Seaquest-trained skills zero-shot on another env (+ env heuristic).
    ``finetune/…`` / ``scratch/…``
        RQ2 continue-train / train-from-scratch skills on the target env.
    ``pqn/pixel`` / ``pqn/oc``
        Flat PQN baseline (in-domain).
    ``transfer/pqn``
        Seaquest-trained flat PQN evaluated zero-shot on another env.
    ``meta({obs})+skills({in})``
        Learned meta-policy on observation ``obs``, selecting among **frozen**
        skills with skill input ``in`` (usually ego). Example:
        ``meta(pixel)+skills(ego)`` = pixel meta + frozen Seaquest ego skills.
    ``tandem({obs})+skills({in})``
        Joint meta+skill training; same ``obs`` / ``in`` reading as meta.
    ``heuristic``
        Random / symbolic baseline with no learned skills.
    """
    kind = str(row.get("paper_eval_kind") or row.get("agent_kind") or "").lower().strip()
    mode = _short_input_mode(str(row.get("skill_input_mode") or ""))
    stem = str(row.get("config_stem") or row.get("run_name") or "").lower()
    oc = str(row.get("object_centric") or "").strip().lower() in {"1", "true", "yes"}

    if kind in {"heuristic_random", "heuristic"} or "heuristic" in stem:
        return "heuristic"
    if kind in {"flat_pqn"} or (
        "pqn" in stem and "skills" not in stem and "tandem" not in stem and "meta" not in stem
    ):
        tag = "pqn/oc" if (oc or mode == "oc" or "_oc" in stem) else "pqn/pixel"
        if "transfer" in stem:
            return "transfer/pqn"
        return tag
    if kind in {"meta"}:
        meta_m = _infer_meta_input_mode(row) or "mixed"
        skill_m = mode or "ego"
        return f"meta({meta_m})+skills({skill_m})"
    if kind in {"tandem"}:
        meta_m = _infer_meta_input_mode(row) or "mixed"
        skill_m = mode or "ego"
        return f"tandem({meta_m})+skills({skill_m})"
    if kind in {"transfer"}:
        tag = mode or "ego"
        return f"transfer/{tag}"
    # skills / hierarchical / finetune / from_scratch
    tag = mode or "ego"
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
    if label.startswith("pqn/") or label == "transfer/pqn":
        return "pqn" if label.startswith("pqn/") else "transfer"
    if label.startswith("meta("):
        return "meta"
    if label.startswith("tandem("):
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
    reward_mean = _to_float(row.get("mean"))
    reward_std = _to_float(row.get("std"), 0.0)
    score_mean = _to_float(row.get("mean_env_score_final"))
    score_std = _to_float(row.get("std_env_score_final"))

    # Prefer unclipped ALE score when present and non-zero. Hierarchical CSVs
    # often store clipped ``mean`` (~count of +1 events) with a real
    # ``mean_env_score_final``. Pixel / some wrappers report score=0 while
    # ``mean`` already holds the unclipped return — do not overwrite that.
    use_score = score_mean == score_mean and (
        abs(score_mean) > 1e-8
        or (reward_mean != reward_mean or abs(reward_mean) <= 1e-8)
    )
    if use_score:
        out["mean"] = score_mean
        out["std"] = score_std if score_std == score_std else reward_std
        out["mean_reward_sum"] = reward_mean
        out["score_source"] = "env_score_final"
        # Drop/retention from clipped reward sums are invalid — recompute later.
        out["drop_from_base"] = float("nan")
        out["drop_frac_from_base"] = float("nan")
        out["retention"] = float("nan")
    else:
        out["mean"] = reward_mean
        out["std"] = reward_std
        out["score_source"] = "episode_return"
        out["drop_from_base"] = _to_float(row.get("drop_from_base"))
        out["drop_frac_from_base"] = _to_float(row.get("drop_frac_from_base"))
        ret = _to_float(row.get("retention"), default=float("nan"))
        if ret != ret:  # NaN
            dfrac = out["drop_frac_from_base"]
            if dfrac == dfrac:
                ret = 1.0 + dfrac
        out["retention"] = ret
    out["max_frames"] = _to_float(row.get("max_frames") or row.get("max_frames_per_episode"))

    base_mean = _to_float(row.get("base_mean"))
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
    """Join each row to its same-run base mean and (re)fill retention."""
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
        else:
            # Always recompute from the (possibly score-promoted) means so
            # clipped drop_frac in old CSVs cannot poison the report.
            if row["base_mean"] == row["base_mean"] and abs(row["base_mean"]) > 1e-8:
                if row["mean"] == row["mean"]:
                    row["drop_from_base"] = float(row["mean"]) - float(row["base_mean"])
                    row["drop_frac_from_base"] = row["drop_from_base"] / abs(
                        float(row["base_mean"])
                    )
                    row["retention"] = float(row["mean"]) / abs(float(row["base_mean"]))
            elif row["retention"] != row["retention"]:
                if row["drop_frac_from_base"] == row["drop_frac_from_base"]:
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


def _dedupe_longest_horizon(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep one row per (config_stem, env, agent_label, mod_key).

    When 800-frame and 27000-frame evals share a master CSV, prefer the longer
    horizon so truncated runs do not dilute the report.
    """
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("config_stem") or ""),
            str(row.get("env") or ""),
            str(row.get("agent_label") or ""),
            str(row.get("mod_key") or ""),
        )
        frames = _to_float(row.get("max_frames"), default=0.0)
        if frames != frames:
            frames = 0.0
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        prev_frames = _to_float(prev.get("max_frames"), default=0.0)
        if prev_frames != prev_frames:
            prev_frames = 0.0
        if frames >= prev_frames:
            best[key] = row
    return list(best.values())


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

    Mixed-horizon master CSVs are deduped to the longest ``max_frames`` per run.
    """
    path = Path(source)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        if path.suffix.lower() == ".json":
            rows = load_summary_json(path)
        else:
            rows = load_summary_csv(path)
    elif path.is_dir():
        # Prefer per-run JSON payloads when present (always one horizon each).
        jsons = sorted(path.glob("*_mods.json"))
        if jsons:
            for p in jsons:
                rows.extend(load_summary_json(p))
            rows = _attach_base_and_retention(rows)
        else:
            master = path / "all_mod_eval_summary.csv"
            if master.is_file() and master.stat().st_size > 0:
                rows = load_summary_csv(master)
            else:
                summaries = sorted(path.glob("*_summary.csv"))
                for p in summaries:
                    rows.extend(load_summary_csv(p))
                rows = _attach_base_and_retention(rows)
    else:
        raise FileNotFoundError(f"No mod-eval summary at {path}")

    rows = _dedupe_longest_horizon(rows)
    rows = _attach_base_and_retention(rows)

    if rq is not None:
        rows = [r for r in rows if int(r.get("paper_eval_rq") or 0) == int(rq)]
    if env is not None:
        env_l = env.lower()
        rows = [r for r in rows if r.get("env") == env_l]
    return rows

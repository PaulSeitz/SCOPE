"""Export paper_eval mod-eval payloads to CSV (summary + per-episode)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _scalarize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str)):
        return value
    return str(value)


def _meta_columns(payload: dict[str, Any], extra_meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = {
        "agent_kind": payload.get("agent_kind", ""),
        "env": payload.get("env", ""),
        "skills_path": payload.get("skills_path", ""),
        "meta_path": payload.get("meta_path", ""),
        "flat_pqn_path": payload.get("flat_pqn_path", ""),
        "object_centric": payload.get("object_centric", ""),
        "base_seed": payload.get("base_seed", ""),
        "num_episodes": payload.get("num_episodes", ""),
        "num_seeds": payload.get("num_seeds", ""),
        "max_frames": payload.get("max_frames", ""),
        "skill_input_mode": payload.get("skill_input_mode", ""),
        "meta_input_mode": payload.get("meta_input_mode", ""),
        "run_name": payload.get("run_name", ""),
        "config_stem": payload.get("config_stem", ""),
        "paper_eval_rq": payload.get("paper_eval_rq", ""),
        "paper_eval_phase": payload.get("paper_eval_phase", ""),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta[k] = v
    return {k: _scalarize(v) for k, v in meta.items()}


def results_to_summary_rows(
    payload: dict[str, Any],
    *,
    extra_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per mod with aggregate stats + metadata."""
    meta = _meta_columns(payload, extra_meta)
    rows: list[dict[str, Any]] = []
    for mod_key, stats in (payload.get("results") or {}).items():
        row = {
            **meta,
            "mod_key": mod_key,
            "mod_category": stats.get("category", ""),
            "mods": _scalarize(stats.get("mods", [])),
            "mean": stats.get("mean", ""),
            "std": stats.get("std", ""),
            "min": stats.get("min", ""),
            "max": stats.get("max", ""),
            "drop_from_base": stats.get("drop_from_base", ""),
            "drop_frac_from_base": stats.get("drop_frac_from_base", ""),
            "num_episodes_observed": stats.get("num_episodes", ""),
            "episodes_per_seed": stats.get("episodes_per_seed", ""),
            "max_frames_per_episode": stats.get("max_frames_per_episode", ""),
            "object_centric_eval": stats.get("object_centric", ""),
            "returns_json": json.dumps(stats.get("returns", [])),
        }
        # Pass through any extra scalar keys on the stats dict (incl. mean_env_*).
        skip = {"returns", "mods", "category", "episode_metrics"}
        for k, v in stats.items():
            if k in row or k in skip:
                continue
            if isinstance(v, (list, dict)):
                row[f"extra_{k}"] = json.dumps(v)
            else:
                row[k] = _scalarize(v)
        rows.append(row)
    return rows


def results_to_episode_rows(
    payload: dict[str, Any],
    *,
    extra_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per episode return (for distribution / violin plots)."""
    meta = _meta_columns(payload, extra_meta)
    rows: list[dict[str, Any]] = []
    for mod_key, stats in (payload.get("results") or {}).items():
        returns = list(stats.get("returns") or [])
        episode_metrics = list(stats.get("episode_metrics") or [])
        n_seeds = int(stats.get("num_seeds") or payload.get("num_seeds") or 1)
        ep_per_seed = int(
            stats.get("episodes_per_seed")
            or payload.get("num_episodes")
            or max(1, (len(returns) // max(1, n_seeds)))
        )
        for i, ret in enumerate(returns):
            seed_idx = i // max(1, ep_per_seed)
            ep_idx = i % max(1, ep_per_seed)
            row = {
                **meta,
                "mod_key": mod_key,
                "mod_category": stats.get("category", ""),
                "mods": _scalarize(stats.get("mods", [])),
                "episode_global_idx": i,
                "seed_idx": seed_idx,
                "episode_idx": ep_idx,
                "return": float(ret),
                "mod_mean": stats.get("mean", ""),
                "mod_std": stats.get("std", ""),
                "drop_from_base": stats.get("drop_from_base", ""),
                "drop_frac_from_base": stats.get("drop_frac_from_base", ""),
            }
            if i < len(episode_metrics) and isinstance(episode_metrics[i], dict):
                for k, v in episode_metrics[i].items():
                    row[k] = _scalarize(v)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"Wrote empty CSV → {path}")
        return
    # Stable union of keys across rows
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Wrote CSV ({len(rows)} rows) → {path}")


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows; rewrite header if new columns appear."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, Any]] = []
    old_fields: list[str] = []
    if path.is_file() and path.stat().st_size > 0:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fieldnames: list[str] = list(old_fields)
    seen = set(fieldnames)
    for row in existing_rows + rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows + rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Appended CSV (+{len(rows)} rows, total {len(existing_rows) + len(rows)}) → {path}")


def write_mod_eval_csvs(
    payload: dict[str, Any],
    *,
    output_stem: str | Path,
    extra_meta: dict[str, Any] | None = None,
    master_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Write:
      - ``{stem}_summary.csv`` — one row per mod (aggregates)
      - ``{stem}_episodes.csv`` — one row per episode return
    and optionally append into master tables under ``master_dir``.
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    summary_path = stem.with_name(stem.name + "_summary.csv")
    episodes_path = stem.with_name(stem.name + "_episodes.csv")

    summary_rows = results_to_summary_rows(payload, extra_meta=extra_meta)
    episode_rows = results_to_episode_rows(payload, extra_meta=extra_meta)
    _write_csv(summary_path, summary_rows)
    _write_csv(episodes_path, episode_rows)

    if master_dir is not None:
        master = Path(master_dir)
        _append_csv(master / "all_mod_eval_summary.csv", summary_rows)
        _append_csv(master / "all_mod_eval_episodes.csv", episode_rows)

    return {"summary": summary_path, "episodes": episodes_path}

"""Write mod-eval score summaries next to the evaluated checkpoint (+ optional wandb sync).

After greedy mod eval we always write a plain-text summary beside the checkpoint
directory that produced the weights. When that directory also has
``wandb_run.json`` (written at train time), we resume that run and attach the
same numbers to ``wandb.summary`` / a small artifact — best-effort, never fatal.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from concept_based_learning.paper_eval.wandb_run_meta import load_wandb_run_meta

MOD_EVAL_SUMMARY_LATEST = "mod_eval_summary.txt"


def resolve_evaluated_ckpt_dir(
    *,
    skills_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    flat_pqn_path: str | Path | None = None,
    save_path: str | Path | None = None,
    env_name: str | None = None,
) -> Path | None:
    """Directory that holds the evaluated checkpoint files."""
    for raw in (skills_path, meta_path, flat_pqn_path):
        if not raw:
            continue
        p = Path(str(raw))
        if p.is_file() or p.suffix:
            return p.resolve().parent
        if p.is_dir():
            return p.resolve()
    if save_path and env_name:
        root = Path(str(save_path))
        if not root.is_absolute():
            # Caller usually already resolved; keep relative as-is under cwd.
            root = root.resolve()
        return (root / str(env_name).lower()).resolve()
    if save_path:
        return Path(str(save_path)).resolve()
    return None


def format_mod_eval_summary_text(
    payload: dict[str, Any],
    *,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    """Human-readable mod table (raw text for disk / wandb summary)."""
    meta = dict(extra_meta or {})
    results = payload.get("results") or {}
    lines: list[str] = []
    lines.append("paper_eval mod evaluation summary")
    lines.append(f"written_at_utc: {datetime.now(timezone.utc).isoformat()}")
    for key in (
        "config_stem",
        "run_name",
        "paper_eval_phase",
        "paper_eval_kind",
        "paper_eval_rq",
        "env_name",
        "agent_kind",
    ):
        if key in meta and meta[key] not in (None, ""):
            lines.append(f"{key}: {meta[key]}")
    if payload.get("env_name"):
        lines.append(f"payload_env: {payload.get('env_name')}")
    lines.append("")
    lines.append(f"{'mod':28s} {'mean':>10s} {'std':>10s} {'drop':>10s}  category")
    lines.append("-" * 72)

    base_mean = None
    if "base" in results and isinstance(results["base"], dict):
        try:
            base_mean = float(results["base"].get("mean"))
        except (TypeError, ValueError):
            base_mean = None

    means: list[float] = []
    for key, stats in results.items():
        if not isinstance(stats, dict):
            continue
        try:
            mean = float(stats.get("mean"))
        except (TypeError, ValueError):
            continue
        means.append(mean)
        try:
            std = float(stats.get("std", 0.0))
        except (TypeError, ValueError):
            std = 0.0
        drop = stats.get("drop_from_base")
        if drop is None and base_mean is not None and key != "base":
            drop = mean - base_mean
        drop_s = f"{float(drop):+10.2f}" if drop is not None else f"{'':>10s}"
        cat = str(stats.get("category") or "")
        lines.append(f"{key:28s} {mean:10.2f} {std:10.2f} {drop_s}  {cat}")

    lines.append("-" * 72)
    if means:
        avg = sum(means) / len(means)
        lines.append(f"{'AVG_OVER_MODS':28s} {avg:10.2f}")
        lines.append(f"n_mods: {len(means)}")
    if base_mean is not None:
        lines.append(f"base_mean: {base_mean:.2f}")
    lines.append("")
    return "\n".join(lines)


def write_mod_eval_summary_files(
    payload: dict[str, Any],
    ckpt_dir: Path,
    *,
    config_stem: str,
    extra_meta: dict[str, Any] | None = None,
    home_eval: bool = False,
) -> list[Path]:
    """Write stem-specific (+ optional latest) summary text under ``ckpt_dir``."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    text = format_mod_eval_summary_text(payload, extra_meta=extra_meta)
    written: list[Path] = []
    stem_path = ckpt_dir / f"mod_eval_{config_stem}_summary.txt"
    stem_path.write_text(text, encoding="utf-8")
    written.append(stem_path)
    if home_eval:
        latest = ckpt_dir / MOD_EVAL_SUMMARY_LATEST
        latest.write_text(text, encoding="utf-8")
        written.append(latest)
    return written


def sync_mod_eval_summary_to_train_wandb(
    *,
    ckpt_dir: Path,
    summary_paths: list[Path],
    payload: dict[str, Any],
    config_stem: str,
    text: str,
    wandb_mode: str | None = None,
) -> bool:
    """Resume the train wandb run and attach mod-eval means + text artifact.

    Returns True on successful sync. Never raises.
    """
    meta = load_wandb_run_meta(ckpt_dir)
    if not meta:
        print(
            f"  mod_eval wandb sync skipped (no {ckpt_dir / 'wandb_run.json'}); "
            "text summary still on disk."
        )
        return False

    mode = (wandb_mode or os.environ.get("WANDB_MODE") or "online").lower()
    if mode == "disabled":
        print("  mod_eval wandb sync skipped (WANDB_MODE=disabled).")
        return False

    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: wandb import failed for mod_eval sync: {exc}")
        return False

    run = None
    try:
        init_kwargs: dict[str, Any] = {
            "id": str(meta["id"]),
            "project": meta.get("project") or "concept_based_learning",
            "resume": "allow",
            "mode": mode,
            "reinit": True,
        }
        if meta.get("entity"):
            init_kwargs["entity"] = meta["entity"]
        # Avoid clobbering the original run name on resume.
        if meta.get("name"):
            init_kwargs["name"] = meta["name"]
        run = wandb.init(**init_kwargs)

        results = payload.get("results") or {}
        means: list[float] = []
        summary = getattr(run, "summary", None)
        if summary is None:
            summary = wandb.summary
        for key, stats in results.items():
            if not isinstance(stats, dict):
                continue
            try:
                mean = float(stats["mean"])
            except (KeyError, TypeError, ValueError):
                continue
            means.append(mean)
            summary[f"mod_eval/{config_stem}/{key}/mean"] = mean
            if stats.get("std") is not None:
                try:
                    summary[f"mod_eval/{config_stem}/{key}/std"] = float(stats["std"])
                except (TypeError, ValueError):
                    pass
        if means:
            summary[f"mod_eval/{config_stem}/avg_over_mods"] = sum(means) / len(means)
            summary[f"mod_eval/{config_stem}/n_mods"] = len(means)
        # Cap text size in summary (wandb summary values have practical limits).
        summary[f"mod_eval/{config_stem}/table"] = text[:15000]

        art_name = f"mod-eval-{config_stem}".replace("/", "-")[:128]
        art = wandb.Artifact(art_name, type="mod_eval")
        for p in summary_paths:
            if p.is_file():
                art.add_file(str(p), name=p.name)
        run.log_artifact(art)
        print(
            f"  mod_eval wandb sync OK → run id={meta['id']} "
            f"({len(means)} mods, artifact={art_name})"
        )
        return True
    except Exception as exc:  # noqa: BLE001 — eval must not die on wandb
        print(f"  WARN: mod_eval wandb sync failed: {exc}")
        return False
    finally:
        try:
            if run is not None:
                import wandb as _wandb

                _wandb.finish()
        except Exception:
            pass


def publish_mod_eval_report(
    payload: dict[str, Any],
    *,
    skills_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    flat_pqn_path: str | Path | None = None,
    save_path: str | Path | None = None,
    env_name: str | None = None,
    config_stem: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    wandb_mode: str | None = None,
) -> list[Path]:
    """Write text summaries next to the ckpt; best-effort sync to train wandb run."""
    ckpt_dir = resolve_evaluated_ckpt_dir(
        skills_path=skills_path,
        meta_path=meta_path,
        flat_pqn_path=flat_pqn_path,
        save_path=save_path,
        env_name=env_name or payload.get("env_name"),
    )
    if ckpt_dir is None:
        print("  WARN: mod_eval summary skipped (could not resolve checkpoint dir).")
        return []

    stem = str(config_stem or (extra_meta or {}).get("config_stem") or "unknown")
    meta = dict(extra_meta or {})
    meta.setdefault("config_stem", stem)
    if env_name:
        meta.setdefault("env_name", env_name)
    # Home eval: same env as the checkpoint folder name.
    home_eval = bool(env_name) and ckpt_dir.name.lower() == str(env_name).lower()

    try:
        paths = write_mod_eval_summary_files(
            payload,
            ckpt_dir,
            config_stem=stem,
            extra_meta=meta,
            home_eval=home_eval,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: failed to write mod_eval summary under {ckpt_dir}: {exc}")
        return []

    for p in paths:
        print(f"  Wrote mod eval summary → {p}")

    text = paths[0].read_text(encoding="utf-8") if paths else ""
    sync_mod_eval_summary_to_train_wandb(
        ckpt_dir=ckpt_dir,
        summary_paths=paths,
        payload=payload,
        config_stem=stem,
        text=text,
        wandb_mode=wandb_mode,
    )
    return paths

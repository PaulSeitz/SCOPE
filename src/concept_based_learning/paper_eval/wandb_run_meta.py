"""Persist / load the training wandb run id next to checkpoints.

Written at train time (``main.single_run``) as ``wandb_run.json`` under
``{SAVE_PATH}/{env}/`` so later mod-eval can resume that run and attach
summary metrics without creating a sibling eval run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WANDB_RUN_META_FILENAME = "wandb_run.json"


def wandb_run_meta_path(save_dir: str | Path) -> Path:
    return Path(save_dir) / WANDB_RUN_META_FILENAME


def persist_wandb_run_meta(save_dir: str | Path, run: Any | None = None) -> Path | None:
    """Write ``wandb_run.json`` for ``run`` (defaults to ``wandb.run``).

    Returns the path written, or ``None`` if there is no active run / write fails.
    Never raises — training must not die on metadata I/O.
    """
    try:
        if run is None:
            import wandb

            run = wandb.run
        if run is None:
            return None
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": _jsonable(getattr(run, "id", None)),
            "name": _jsonable(getattr(run, "name", None)),
            "project": _jsonable(getattr(run, "project", None)),
            "entity": _jsonable(getattr(run, "entity", None)),
            "path": _jsonable(getattr(run, "path", None)),
            "url": _jsonable(getattr(run, "url", None)),
        }
        if not payload.get("id"):
            return None
        path = wandb_run_meta_path(out_dir)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
    except Exception as exc:  # noqa: BLE001 — never break training
        print(f"WARN: could not persist wandb run meta under {save_dir}: {exc}")
        return None


def _jsonable(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value) if not isinstance(value, str) else value
    # Real wandb Run fields are strings; reject MagicMock / weird objects.
    text = str(value)
    if "MagicMock" in type(value).__name__ or text.startswith("<MagicMock"):
        return None
    return text


def load_wandb_run_meta(save_dir: str | Path) -> dict[str, Any] | None:
    """Load ``wandb_run.json`` if present and well-formed (has ``id``)."""
    path = wandb_run_meta_path(save_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("id"):
        return None
    return payload

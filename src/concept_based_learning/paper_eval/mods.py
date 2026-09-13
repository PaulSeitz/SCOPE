"""Load paper_eval mod allowlists from ``configs/experiment/paper_eval/mods``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODS_DIR = PROJECT_ROOT / "configs" / "experiment" / "paper_eval" / "mods"


@dataclass(frozen=True)
class ModSpec:
    key: str
    mods: tuple[str, ...]
    category: str


def load_mod_allowlist(env: str) -> dict[str, Any]:
    # old_seaquest shares the seaquest robustness mod registry/allowlist.
    env_key = str(env).lower().strip()
    if env_key == "old_seaquest":
        env_key = "seaquest"
    path = MODS_DIR / f"{env_key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No mod allowlist for env={env!r}: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid mod allowlist: {path}")
    return data


def mod_specs_for_env(
    env: str,
    *,
    paper_core_only: bool = False,
    trial: bool = False,
    keys: list[str] | None = None,
) -> list[ModSpec]:
    raw = load_mod_allowlist(env)
    entries = list(raw.get("eval_mods") or [])
    specs = [
        ModSpec(
            key=str(e["key"]),
            mods=tuple(e.get("mods") or ()),
            category=str(e.get("category", "other")),
        )
        for e in entries
        if not e.get("skip_eval")
    ]
    if keys:
        wanted = set(keys)
        specs = [s for s in specs if s.key in wanted]
    elif str(raw.get("status", "")).lower() == "draft":
        # Unpublished mods — eval base only until JAXAtari implements them.
        specs = [s for s in specs if s.key == "base"] or specs[:1]
    elif trial:
        # Minimal OOD smoke: base + first non-control mod if present.
        base = [s for s in specs if s.key == "base"]
        rest = [s for s in specs if s.key != "base"]
        specs = base + rest[:1]
    elif paper_core_only and raw.get("paper_core"):
        wanted = set(raw["paper_core"])
        specs = [s for s in specs if s.key in wanted]
    return specs

"""Resolve NEXUS / JAXAtari roots for local and cluster runs.

Environment variables (cluster-friendly):
  NEXUS_ROOT              Path to symbolic_options checkout (develop).
  PAPER_EVAL_JAXATARI_ROOT / JAXATARI_ROOT
                          Path to the Seaquest robustness jaxatari fork used
                          by paper_eval. Prefer this over NEXUS's vendored copy.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NEXUS_ROOT = PROJECT_ROOT / "third_party" / "symbolic_options"



def resolve_nexus_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("NEXUS_ROOT")
        root = Path(env).expanduser().resolve() if env else DEFAULT_NEXUS_ROOT
    if not (root / "main.py").is_file():
        raise FileNotFoundError(
            f"NEXUS_ROOT missing main.py: {root}. Clone with:\n"
            "  git submodule update --init --recursive third_party/symbolic_options\n"
            "or set NEXUS_ROOT=/path/to/symbolic_options"
        )
    return root


def resolve_jaxatari_root(explicit: str | Path | None = None) -> Path | None:
    value = explicit or os.environ.get("JAXATARI_ROOT") or os.environ.get("PAPER_EVAL_JAXATARI_ROOT")
    if value is None:
        return None  # use the installed, pinned upstream package
    root = Path(value).expanduser().resolve()
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"JAXtari root missing src/: {root}")
    return root


def nexus_alg_config_dir(nexus_root: Path | None = None) -> Path:
    root = nexus_root or resolve_nexus_root()
    return root / "src" / "symbolic_options" / "config" / "alg"

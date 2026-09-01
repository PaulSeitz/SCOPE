#!/usr/bin/env python3
"""Backward-compatible wrapper for clip100 probe materialization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from materialize_clip_collectsurface_probe import materialize_clip  # noqa: E402


def main() -> None:
    materialize_clip(100, dry=False)
    print("done.")


if __name__ == "__main__":
    main()

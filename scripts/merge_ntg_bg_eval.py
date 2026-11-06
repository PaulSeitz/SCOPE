#!/usr/bin/env python3
"""Splice a one-mod Name This Game re-eval into the full mod file.

The re-eval stem is the full stem plus a suffix (``_bg`` by default). Rows
already stored under that mod are replaced, so running this twice does not
duplicate them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MOD = "change_background_color"


def _merge_json(bg: Path, full: Path, mod: str) -> None:
    payload = json.loads(bg.read_text())
    results = payload.get("results") or {}
    if mod not in results or payload.get("status") == "failed" or payload.get("failed_mods"):
        raise SystemExit(f"refusing incomplete re-eval {bg.name}")
    target = json.loads(full.read_text()) if full.is_file() else {}
    if not target.get("results"):
        raise SystemExit(f"full eval missing or empty: {full}")
    target.setdefault("results", {})[mod] = results[mod]
    mods = [m for m in (target.get("mods") or []) if m.get("key") != mod]
    extra = [m for m in (payload.get("mods") or []) if m.get("key") == mod]
    target["mods"] = mods + extra
    full.write_text(json.dumps(target, indent=2) + "\n")


def _merge_csv(bg: Path, full: Path, mod: str) -> None:
    if not bg.is_file() or not full.is_file():
        return
    bg_lines = bg.read_text().splitlines()
    if len(bg_lines) < 2:
        return
    kept = [ln for ln in full.read_text().splitlines() if f",{mod}," not in ln and not ln.endswith(f",{mod}")]
    if not kept:
        kept = [bg_lines[0]]
    full.write_text("\n".join(kept + bg_lines[1:]) + "\n")


def merge_stem(eval_dir: Path, stem: str, *, mod: str, suffix: str) -> None:
    """``stem`` is the re-eval stem, ending in ``suffix``."""
    if not stem.endswith(suffix):
        raise SystemExit(f"expected a {suffix} stem, got {stem}")
    full_stem = stem[: -len(suffix)]
    bg_json = eval_dir / f"{stem}_mods.json"
    full_json = eval_dir / f"{full_stem}_mods.json"
    if not bg_json.is_file():
        print(f"skip {stem}: re-eval json missing")
        return
    if not full_json.is_file():
        print(f"skip {stem}: full eval not written yet ({full_json.name})")
        return
    _merge_json(bg_json, full_json, mod)
    for csv_suffix in ("_mods_summary.csv", "_mods_episodes.csv"):
        _merge_csv(eval_dir / f"{stem}{csv_suffix}", eval_dir / f"{full_stem}{csv_suffix}", mod)
    print(f"spliced {mod} into {full_stem}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--mod", default=MOD)
    parser.add_argument("--suffix", default="_bg")
    parser.add_argument("stems", nargs="+")
    args = parser.parse_args()
    for stem in args.stems:
        merge_stem(args.eval_dir, stem, mod=args.mod, suffix=args.suffix)


if __name__ == "__main__":
    main()

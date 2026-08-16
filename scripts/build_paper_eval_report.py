#!/usr/bin/env python3
"""
Build a paper_eval PDF report from mod-eval CSVs / JSONs.

Examples:
  python scripts/build_paper_eval_report.py \\
    --source outputs/models/paper_eval/eval \\
    --output outputs/reports/paper_eval_report.pdf

  python scripts/build_paper_eval_report.py \\
    --source outputs/models/paper_eval/eval/all_mod_eval_summary.csv \\
    --rq 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models" / "paper_eval" / "eval",
        help=(
            "all_mod_eval_summary.csv, a *_summary.csv / *_mods.json, "
            "or a directory containing them (default: outputs/models/paper_eval/eval)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "paper_eval_report.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG dumps (default: next to the PDF).",
    )
    parser.add_argument(
        "--rq",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Only build pages for this research question (default: all present).",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help=(
            "Directory tree with **/training_metrics.csv (e.g. paper_eval/). "
            "Adds RQ1 train-peak vs eval bars when present."
        ),
    )
    args = parser.parse_args()

    from concept_based_learning.paper_eval.report import build_paper_eval_report

    build_paper_eval_report(
        args.source,
        output_pdf=args.output,
        fig_dir=args.fig_dir,
        rq=args.rq,
        train_root=args.train_root,
    )


if __name__ == "__main__":
    main()

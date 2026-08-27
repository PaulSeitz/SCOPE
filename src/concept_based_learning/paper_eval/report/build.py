"""Assemble paper_eval figures into a multi-page PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from concept_based_learning.paper_eval.report.figures import (
    plot_base_returns,
    plot_category_retention,
    plot_overall_retention_bars,
    plot_point_farming_scores,
    plot_retention_heatmap,
    plot_return_vs_retention_scatter,
    plot_train_peak_vs_eval,
    save_figure,
    _require_matplotlib,
)
from concept_based_learning.paper_eval.report.load import load_mod_eval_summary
from concept_based_learning.paper_eval.report.metrics import (
    POINT_FARMING_MODS,
    envs_in_rows,
    filter_rows,
    split_point_farming,
)
from concept_based_learning.paper_eval.report.training_metrics import (
    discover_training_peaks,
    train_eval_pairs,
)


def _title_page(pdf, *, title: str, subtitle: str, lines: list[str]) -> None:
    plt = _require_matplotlib()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.92, title, fontsize=18, weight="bold")
    fig.text(0.08, 0.87, subtitle, fontsize=11, color="0.35")
    y = 0.80
    for line in lines:
        size = 8.5 if line.startswith("  ") else 9.5
        weight = "bold" if line.endswith(":") and not line.startswith(" ") else "normal"
        fig.text(0.08, y, line, fontsize=size, family="monospace", weight=weight)
        y -= 0.028
        if y < 0.04:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _add_fig(pdf, fig, *, png_path: Path | None = None) -> None:
    plt = _require_matplotlib()
    if png_path is not None:
        save_figure(fig, png_path)
    pdf.savefig(fig)
    plt.close(fig)


def build_rq1_pages(
    pdf,
    rows: list[dict[str, Any]],
    *,
    fig_dir: Path | None = None,
    train_root: Path | None = None,
) -> int:
    rq1 = filter_rows(rows, rq=1)
    if not rq1:
        # Fall back: unlabeled rows that look like seaquest suite evals.
        rq1 = [
            r
            for r in rows
            if r.get("env") == "seaquest"
            and r.get("protocol") in {"skills", "pqn", "heuristic"}
        ]
    if not rq1:
        return 0

    n = 0
    fig = plot_base_returns(
        rq1, title="RQ1 — Base return (Seaquest greedy eval, unclipped score)"
    )
    _add_fig(pdf, fig, png_path=(fig_dir / "rq1_base_returns.png") if fig_dir else None)
    n += 1

    if train_root is not None and Path(train_root).is_dir():
        peaks = discover_training_peaks(train_root)
        pairs = train_eval_pairs(rq1, peaks, env="seaquest", rq=1)
        if any(p.get("train_peak") == p.get("train_peak") for p in pairs):
            fig = plot_train_peak_vs_eval(
                pairs,
                title="RQ1 — Train peak vs greedy eval (overfit / protocol gap)",
            )
            _add_fig(
                pdf,
                fig,
                png_path=(fig_dir / "rq1_train_vs_eval.png") if fig_dir else None,
            )
            n += 1

    # Easy no-threat mods explode retention for weak bases — plot separately.
    rq1_main, rq1_farm = split_point_farming(rq1)
    farm_keys = sorted(
        {
            str(r["mod_key"])
            for r in rq1_farm
            if r.get("mod_key") in POINT_FARMING_MODS
        }
    )

    fig = plot_category_retention(
        rq1_main,
        title=(
            "RQ1 — Mean retention by mod category (TLDR; "
            "excl. point-farming mods)"
        ),
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq1_category_retention.png") if fig_dir else None,
    )
    n += 1

    if farm_keys:
        fig = plot_point_farming_scores(
            rq1_farm,
            title=(
                "RQ1 — Point-farming mods (absolute score; "
                f"{', '.join(farm_keys)})"
            ),
            farming_mods=farm_keys,
        )
        _add_fig(
            pdf,
            fig,
            png_path=(fig_dir / "rq1_point_farming_scores.png")
            if fig_dir
            else None,
        )
        n += 1

    fig = plot_retention_heatmap(
        rq1_main, title="RQ1 — Retention heatmap (mod × agent; excl. point-farming)"
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq1_retention_heatmap.png") if fig_dir else None,
    )
    n += 1

    fig = plot_overall_retention_bars(
        rq1_main,
        title="RQ1 — Mean retention across mods (excl. point-farming)",
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq1_overall_retention.png") if fig_dir else None,
    )
    n += 1
    return n


def build_rq2_pages(
    pdf,
    rows: list[dict[str, Any]],
    *,
    fig_dir: Path | None = None,
) -> int:
    rq2 = filter_rows(rows, rq=2)
    if not rq2:
        return 0
    n = 0
    # Cross-env summary first
    fig = plot_overall_retention_bars(
        rq2, title="RQ2 — Mean retention across envs/mods (all protocols)"
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq2_overall_retention.png") if fig_dir else None,
    )
    n += 1

    for env in envs_in_rows(rq2):
        env_rows = filter_rows(rq2, env=env)
        fig = plot_base_returns(
            env_rows, title=f"RQ2 / {env} — Base return (transfer / finetune / scratch)"
        )
        _add_fig(
            pdf,
            fig,
            png_path=(fig_dir / f"rq2_{env}_base_returns.png") if fig_dir else None,
        )
        n += 1
        fig = plot_category_retention(
            env_rows, title=f"RQ2 / {env} — Retention by mod category"
        )
        _add_fig(
            pdf,
            fig,
            png_path=(fig_dir / f"rq2_{env}_category_retention.png")
            if fig_dir
            else None,
        )
        n += 1
        fig = plot_retention_heatmap(
            env_rows, title=f"RQ2 / {env} — Retention heatmap"
        )
        _add_fig(
            pdf,
            fig,
            png_path=(fig_dir / f"rq2_{env}_retention_heatmap.png")
            if fig_dir
            else None,
        )
        n += 1
    return n


def build_rq3_pages(
    pdf,
    rows: list[dict[str, Any]],
    *,
    fig_dir: Path | None = None,
) -> int:
    # RQ3 agents + RQ1 refs (heuristic / skills ego / pqn) for comparison.
    rq3 = filter_rows(rows, rq=3)
    rq1_refs = [
        r
        for r in filter_rows(rows, rq=1, env="seaquest")
        if r.get("agent_label")
        in {"heuristic", "skills/ego", "pqn/pixel", "pqn/oc"}
    ]
    combined = rq3 + rq1_refs
    if not rq3:
        return 0
    n = 0
    fig = plot_return_vs_retention_scatter(
        combined,
        title=(
            "RQ3 — In-domain return vs retention "
            "(learned meta/tandem vs RQ1 refs)"
        ),
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq3_return_vs_retention.png") if fig_dir else None,
    )
    n += 1
    fig = plot_category_retention(
        combined, title="RQ3 — Retention by category (with RQ1 references)"
    )
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq3_category_retention.png") if fig_dir else None,
    )
    n += 1
    fig = plot_base_returns(combined, title="RQ3 — Base return comparison")
    _add_fig(
        pdf,
        fig,
        png_path=(fig_dir / "rq3_base_returns.png") if fig_dir else None,
    )
    n += 1
    return n


def build_paper_eval_report(
    source: str | Path,
    *,
    output_pdf: str | Path,
    fig_dir: str | Path | None = None,
    rq: int | None = None,
    train_root: str | Path | None = None,
) -> Path:
    """
    Build a multi-page PDF (+ optional PNG dumps) from mod-eval summary data.

    ``source`` is a CSV, JSON, or the ``outputs/.../eval`` directory.
    ``train_root`` (optional) is scanned for ``**/training_metrics.csv`` to
    add train-peak vs eval bars on RQ1.
    """
    plt = _require_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    rows = load_mod_eval_summary(source)
    if not rows:
        raise SystemExit(f"No mod-eval rows loaded from {source}")

    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    figures = Path(fig_dir) if fig_dir is not None else out.with_suffix("").parent / (
        out.stem + "_figures"
    )
    figures.mkdir(parents=True, exist_ok=True)

    train_path = Path(train_root) if train_root is not None else None
    agents = sorted({r["agent_label"] for r in rows})
    envs = envs_in_rows(rows)
    train_note = (
        f"Train metrics root: {train_path}"
        if train_path is not None and train_path.is_dir()
        else "Train metrics: (not provided; pass --train-root for train-vs-eval)"
    )
    with PdfPages(out) as pdf:
        _title_page(
            pdf,
            title="Concept-Based Learning — Paper Eval Report",
            subtitle=f"Source: {Path(source)}",
            lines=[
                f"rows={len(rows)}  agents={len(agents)}  envs={', '.join(envs) or '(none)'}",
                f"agents: {', '.join(agents)}",
                "Metrics: unclipped ALE score (env_score_final); "
                "retention = mod_score / |base_score|  (1.0 = no change)",
                "Point-farming mods (disable_enemies, no_enemies) excluded from "
                "main retention plots; absolute scores shown separately.",
                "Primary evidence: frozen greedy mod-eval (not training curves).",
                train_note,
                "Train peak = max logged avg_game_return / returned_episode_returns.",
                "",
                "Label legend:",
                "  skills/X          = hierarchical skills (input X) + symbolic heuristic",
                "  transfer/X        = Seaquest skills (input X) zero-shot on target env",
                "  transfer/pqn      = Seaquest flat PQN zero-shot on target env",
                "  finetune/X        = continue-train skills on target env",
                "  scratch/X         = train skills from scratch on target env",
                "  pqn/pixel|oc      = flat PQN in-domain (Seaquest)",
                "  meta(M)+skills(S) = learned meta on obs M; frozen skills with input S",
                "  tandem(M)+skills(S) = joint meta+skill train (same M/S reading)",
                "  heuristic         = non-learned / random-skill control",
            ],
        )
        n_pages = 1
        if rq in (None, 1):
            n_pages += build_rq1_pages(
                pdf, rows, fig_dir=figures, train_root=train_path
            )
        if rq in (None, 2):
            n_pages += build_rq2_pages(pdf, rows, fig_dir=figures)
        if rq in (None, 3):
            n_pages += build_rq3_pages(pdf, rows, fig_dir=figures)

    print(f"Wrote report PDF ({n_pages} pages) → {out}")
    print(f"Figures directory → {figures}")
    return out

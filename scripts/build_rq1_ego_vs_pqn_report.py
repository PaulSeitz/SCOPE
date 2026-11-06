#!/usr/bin/env python3
"""RQ1 head-to-head: skills/ego vs pqn/pixel — focused comparison PDF + figures.

Uses the thesis_facing collectsurface ego skills eval and seaquest flat PQN
pixel eval from tmp/eval_thesis_facing_abc (or outputs/models/paper_eval/eval).

Example:
  python scripts/build_rq1_ego_vs_pqn_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concept_based_learning.paper_eval.report.figures import (  # noqa: E402
    _require_matplotlib,
    _style,
    plot_base_returns,
    plot_category_retention,
    plot_point_farming_scores,
    plot_retention_heatmap,
    save_figure,
)
from concept_based_learning.paper_eval.report.load import load_mod_eval_summary  # noqa: E402
from concept_based_learning.paper_eval.report.metrics import (  # noqa: E402
    POINT_FARMING_MODS,
    mean_retention_by_agent,
    median_retention_by_agent,
    split_point_farming,
)

AGENTS = ("skills/ego", "pqn/pixel")
COLORS = {"skills/ego": "#2a9d8f", "pqn/pixel": "#e76f51"}
# Inflates mean retention; show with/without for honesty.
EASY_INFLATORS = frozenset({"peaceful_enemies"})
STEMS = {
    "skills/ego": "seaquest_skills_ego_pov_clip50_facing_psibins8_collectsurface_eval_full",
    "pqn/pixel": "seaquest_pqn_eval",
}


def _filter_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        stem = str(r.get("config_stem") or "")
        label = str(r.get("agent_label") or "")
        if r.get("env") != "seaquest":
            continue
        if label == "skills/ego" and stem == STEMS["skills/ego"]:
            keep.append(r)
        elif label == "pqn/pixel" and stem == STEMS["pqn/pixel"]:
            keep.append(r)
    return keep


def _relabel(rows: list[dict]) -> list[dict]:
    """Stable display names for the two arms."""
    out = []
    for r in rows:
        rr = dict(r)
        lab = str(rr.get("agent_label") or "")
        if lab == "skills/ego":
            rr["agent_label"] = "skills/ego (facing φ8)"
        elif lab == "pqn/pixel":
            rr["agent_label"] = "pqn/pixel (flat)"
        out.append(rr)
    return out


DISPLAY = {
    "skills/ego": "skills/ego (facing φ8)",
    "pqn/pixel": "pqn/pixel (flat)",
}
DISPLAY_COLORS = {
    "skills/ego (facing φ8)": COLORS["skills/ego"],
    "pqn/pixel (flat)": COLORS["pqn/pixel"],
}


def _base_means(rows: list[dict]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        if r.get("mod_key") != "base":
            continue
        out[str(r["agent_label"])] = (
            float(r["mean"]),
            float(r.get("std") or 0.0),
        )
    return out


def plot_mean_vs_median(rows: list[dict], *, title: str, ax=None):
    plt = _require_matplotlib()
    main, _ = split_point_farming(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4.2))
    means = mean_retention_by_agent(main)
    medians = median_retention_by_agent(main)
    # Also mean without peaceful_enemies
    no_peace = [
        r
        for r in main
        if r.get("mod_key") not in EASY_INFLATORS
    ]
    means_np = mean_retention_by_agent(no_peace)

    agents = [DISPLAY["skills/ego"], DISPLAY["pqn/pixel"]]
    x = np.arange(len(agents))
    w = 0.25
    series = [
        ("Mean (all mods)", [means.get(a, np.nan) for a in agents], "#264653"),
        (
            "Mean (excl. peaceful_enemies)",
            [means_np.get(a, np.nan) for a in agents],
            "#457b9d",
        ),
        ("Median", [medians.get(a, np.nan) for a in agents], "#e9c46a"),
    ]
    for i, (lab, vals, color) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, width=w, label=lab, color=color)
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylabel("Retention (mod / |base|)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_paired_mod_retention(rows: list[dict], *, title: str, ax=None):
    """Per-mod retention bars for both agents, sorted by ego retention."""
    plt = _require_matplotlib()
    main, _ = split_point_farming(rows)
    by: dict[str, dict[str, float]] = {}
    for r in main:
        if r.get("mod_key") == "base":
            continue
        mod = str(r["mod_key"])
        ret = r.get("retention")
        if ret is None or ret != ret:
            continue
        by.setdefault(mod, {})[str(r["agent_label"])] = float(ret)

    ego = DISPLAY["skills/ego"]
    pqn = DISPLAY["pqn/pixel"]
    mods = sorted(by.keys(), key=lambda m: by[m].get(ego, -999), reverse=True)
    created = ax is None
    if created:
        fig_h = max(5.5, 0.28 * len(mods) + 1.5)
        _, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(mods))
    h = 0.38
    ax.barh(
        y + h / 2,
        [by[m].get(ego, np.nan) for m in mods],
        height=h,
        label=ego,
        color=DISPLAY_COLORS[ego],
    )
    ax.barh(
        y - h / 2,
        [by[m].get(pqn, np.nan) for m in mods],
        height=h,
        label=pqn,
        color=DISPLAY_COLORS[pqn],
    )
    ax.axvline(1.0, color="0.35", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(mods, fontsize=8)
    ax.set_xlabel("Retention (mod / |base|)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_retention_delta(rows: list[dict], *, title: str, ax=None):
    """ego − pqn retention per mod (positive = ego more robust)."""
    plt = _require_matplotlib()
    main, _ = split_point_farming(rows)
    by: dict[str, dict[str, float]] = {}
    for r in main:
        if r.get("mod_key") == "base":
            continue
        ret = r.get("retention")
        if ret is None or ret != ret:
            continue
        by.setdefault(str(r["mod_key"]), {})[str(r["agent_label"])] = float(ret)

    ego = DISPLAY["skills/ego"]
    pqn = DISPLAY["pqn/pixel"]
    deltas = []
    for mod, d in by.items():
        if ego in d and pqn in d:
            deltas.append((d[ego] - d[pqn], mod, d[ego], d[pqn]))
    deltas.sort()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10, max(5.0, 0.28 * len(deltas) + 1.2)))
    ys = np.arange(len(deltas))
    vals = [d[0] for d in deltas]
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    ax.barh(ys, vals, color=colors)
    ax.axvline(0.0, color="0.3", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([d[1] for d in deltas], fontsize=8)
    ax.set_xlabel("Retention(ego) − Retention(PQN)")
    ax.set_title(title)
    ax.invert_yaxis()
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_base_comparison(rows: list[dict], *, title: str, ax=None):
    plt = _require_matplotlib()
    bases = _base_means(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 4.2))
    agents = [DISPLAY["skills/ego"], DISPLAY["pqn/pixel"]]
    means = [bases[a][0] for a in agents]
    stds = [bases[a][1] for a in agents]
    x = np.arange(len(agents))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[DISPLAY_COLORS[a] for a in agents],
        ecolor="0.25",
    )
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + max(means) * 0.02, f"{m:.0f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Mean episode return (unclipped)")
    ax.set_title(title)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def _title_page(pdf, *, bases: dict, mean_ret: dict, med_ret: dict, mean_np: dict):
    plt = _require_matplotlib()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.08, 0.90, "RQ1 head-to-head", fontsize=18, weight="bold")
    fig.text(
        0.08,
        0.85,
        "skills/ego (facing φ8 · clip50 · COLLECT-surface)  vs  pqn/pixel (flat)",
        fontsize=11,
        color="0.35",
    )
    ego = DISPLAY["skills/ego"]
    pqn = DISPLAY["pqn/pixel"]
    lines = [
        "Seaquest greedy mod-eval (27k × 3 eps). Unclipped ALE score.",
        "Retention = mod_score / |base_score|; 1.0 = unchanged.",
        "Point-farming mods (disable_enemies, no_enemies) excluded from retention means.",
        "peaceful_enemies inflates mean retention — median / excl. variant shown.",
        "",
        f"{ego}:",
        f"  base = {bases[ego][0]:.1f} ± {bases[ego][1]:.1f}",
        f"  mean retention = {mean_ret.get(ego, float('nan')):.2f}",
        f"  mean excl. peaceful_enemies = {mean_np.get(ego, float('nan')):.2f}",
        f"  median retention = {med_ret.get(ego, float('nan')):.2f}",
        "",
        f"{pqn}:",
        f"  base = {bases[pqn][0]:.1f} ± {bases[pqn][1]:.1f}",
        f"  mean retention = {mean_ret.get(pqn, float('nan')):.2f}",
        f"  mean excl. peaceful_enemies = {mean_np.get(pqn, float('nan')):.2f}",
        f"  median retention = {med_ret.get(pqn, float('nan')):.2f}",
        "",
        "Story: ego skills ≈ flat PQN on base competence; compare robustness",
        "per mod / category to see whether hierarchical skills buy dynamics retention.",
    ]
    y = 0.78
    for line in lines:
        fig.text(0.08, y, line, fontsize=9.5, family="monospace")
        y -= 0.032
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    src_candidates = [
        ROOT / "tmp/eval_thesis_facing_abc",
        ROOT / "outputs/models/paper_eval/eval",
    ]
    source = next(p for p in src_candidates if p.is_dir())
    rows = _relabel(_filter_rows(load_mod_eval_summary(source)))
    if not rows:
        raise SystemExit("No skills/ego + pqn/pixel Seaquest rows found")

    out = ROOT / "outputs/reports/paper_eval_report_rq1_ego_vs_pqn.pdf"
    fig_dir = ROOT / "outputs/reports/paper_eval_report_rq1_ego_vs_pqn_figures"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    main_rows, _ = split_point_farming(rows)
    bases = _base_means(rows)
    mean_ret = mean_retention_by_agent(main_rows)
    med_ret = median_retention_by_agent(main_rows)
    mean_np = mean_retention_by_agent(
        [r for r in main_rows if r.get("mod_key") not in EASY_INFLATORS]
    )

    plt = _require_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    pages = []

    def add(fig, name: str):
        pages.append((fig, name))

    add(
        plot_base_comparison(
            rows,
            title="RQ1 — Base return: skills/ego vs flat PQN pixel",
        ),
        "base_returns",
    )
    add(
        plot_mean_vs_median(
            rows,
            title="RQ1 — Retention summary (mean vs median; peaceful_enemies effect)",
        ),
        "retention_summary",
    )
    add(
        plot_category_retention(
            main_rows,
            title="RQ1 — Retention by mod category",
        ),
        "category_retention",
    )
    add(
        plot_retention_heatmap(
            main_rows,
            title="RQ1 — Retention heatmap (mod × agent)",
            agents=[DISPLAY["skills/ego"], DISPLAY["pqn/pixel"]],
        ),
        "retention_heatmap",
    )
    add(
        plot_paired_mod_retention(
            rows,
            title="RQ1 — Per-mod retention (sorted by ego)",
        ),
        "per_mod_retention",
    )
    add(
        plot_retention_delta(
            rows,
            title="RQ1 — Where ego beats / loses to PQN (retention delta)",
        ),
        "retention_delta",
    )
    add(
        plot_point_farming_scores(
            rows,
            title="RQ1 — Point-farming mods (absolute score)",
        ),
        "point_farming",
    )

    with PdfPages(out) as pdf:
        _title_page(
            pdf,
            bases=bases,
            mean_ret=mean_ret,
            med_ret=med_ret,
            mean_np=mean_np,
        )
        for fig, name in pages:
            save_figure(fig, fig_dir / f"rq1_ego_vs_pqn_{name}.png")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {len(pages) + 1} pages → {out}")
    print(f"Figures → {fig_dir}")
    ego = DISPLAY["skills/ego"]
    pqn = DISPLAY["pqn/pixel"]
    print(
        f"  {ego}: base={bases[ego][0]:.1f}  mean_ret={mean_ret[ego]:.2f}  "
        f"median={med_ret[ego]:.2f}  mean_excl_peace={mean_np[ego]:.2f}"
    )
    print(
        f"  {pqn}: base={bases[pqn][0]:.1f}  mean_ret={mean_ret[pqn]:.2f}  "
        f"median={med_ret[pqn]:.2f}  mean_excl_peace={mean_np[pqn]:.2f}"
    )


if __name__ == "__main__":
    main()

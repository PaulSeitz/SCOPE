"""Matplotlib figures for the paper_eval PDF report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from concept_based_learning.paper_eval.report.metrics import (
    POINT_FARMING_MODS,
    base_returns_by_agent,
    mean_retention_by_agent,
    mean_retention_by_category,
    median_retention_by_agent,
    retention_matrix,
    transfer_base_matrix,
)


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for paper_eval reports. "
            "Install with: uv add matplotlib  (or pip install matplotlib)"
        ) from exc
    return plt


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def plot_base_returns(
    rows: list[dict[str, Any]],
    *,
    title: str = "Base return (greedy eval)",
    ax=None,
):
    plt = _require_matplotlib()
    data = base_returns_by_agent(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4.2))
    if not data:
        ax.text(0.5, 0.5, "No base-return rows", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax

    labels = [d[0] for d in data]
    means = [d[1][0] for d in data]
    stds = [d[1][1] for d in data]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color="#3d5a80", ecolor="#293241")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean episode return (unclipped score)")
    ax.set_title(title)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_category_retention(
    rows: list[dict[str, Any]],
    *,
    title: str = "Mean retention by mod category",
    ax=None,
):
    """Grouped bars: categories on x, one bar per agent."""
    plt = _require_matplotlib()
    table = mean_retention_by_category(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 4.5))
    if not table:
        ax.text(0.5, 0.5, "No retention rows", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax

    agents = sorted(table.keys())
    categories = sorted({c for cats in table.values() for c in cats})
    x = np.arange(len(categories))
    width = min(0.8 / max(1, len(agents)), 0.18)
    cmap = plt.get_cmap("tab10")
    for i, agent in enumerate(agents):
        vals = [table[agent].get(cat, np.nan) for cat in categories]
        offset = (i - (len(agents) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=agent, color=cmap(i % 10))
    ax.axhline(1.0, color="0.4", lw=1, ls="--", label="base (=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Retention (mod / base)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_retention_heatmap(
    rows: list[dict[str, Any]],
    *,
    title: str = "Retention heatmap (mod × agent)",
    agents: Sequence[str] | None = None,
    ax=None,
):
    plt = _require_matplotlib()
    mods, agent_list, matrix = retention_matrix(
        rows, agents=list(agents) if agents is not None else None
    )
    created = ax is None
    if created:
        h = max(4.0, 0.28 * len(mods) + 1.5)
        w = max(6.0, 0.9 * len(agent_list) + 2.0)
        _, ax = plt.subplots(figsize=(w, h))
    if not mods or not agent_list:
        ax.text(0.5, 0.5, "No retention matrix", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax

    arr = np.asarray(matrix, dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.2)
    ax.set_xticks(np.arange(len(agent_list)))
    ax.set_yticks(np.arange(len(mods)))
    ax.set_xticklabels(agent_list, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(mods, fontsize=7)
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("retention")
    # Annotate finite cells lightly when matrix is small.
    if arr.size <= 200:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_overall_retention_bars(
    rows: list[dict[str, Any]],
    *,
    title: str = "Mean retention across all mods",
    ax=None,
):
    plt = _require_matplotlib()
    means = mean_retention_by_agent(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4.0))
    if not means:
        ax.text(0.5, 0.5, "No retention rows", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax
    labels = sorted(means.keys())
    vals = [means[k] for k in labels]
    x = np.arange(len(labels))
    ax.bar(x, vals, color="#2a9d8f")
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean retention")
    ax.set_title(title)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_return_vs_retention_scatter(
    rows: list[dict[str, Any]],
    *,
    title: str = "In-domain return vs mean retention",
    ax=None,
):
    """One point per agent: x=base return, y=mean retention over mods."""
    plt = _require_matplotlib()
    bases = {k: v[0] for k, v in base_returns_by_agent(rows)}
    rets = mean_retention_by_agent(rows)
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 5))
    labels = sorted(set(bases) & set(rets))
    if not labels:
        ax.text(0.5, 0.5, "Need base + mod rows", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax
    xs = [bases[k] for k in labels]
    ys = [rets[k] for k in labels]
    ax.scatter(xs, ys, s=60, c="#e76f51", zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("Base return")
    ax.set_ylabel("Mean retention")
    ax.set_title(title)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_train_peak_vs_eval(
    pairs: list[dict[str, Any]],
    *,
    title: str = "Train peak vs greedy eval (base)",
    ylabel: str = "Unclipped game return",
    ax=None,
):
    """
    Grouped bars: highest logged training ``avg_game_return`` vs eval base mean.

    Large train≫eval gaps are a robustness / overfitting signal (different
    protocols: train = rolling mean over parallel envs; eval = few greedy eps).
    """
    plt = _require_matplotlib()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 4.5))
    usable = [
        p
        for p in pairs
        if (p.get("train_peak") == p.get("train_peak"))
        or (p.get("eval_mean") == p.get("eval_mean"))
    ]
    if not usable:
        ax.text(0.5, 0.5, "No train/eval pairs", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax

    labels = [str(p["agent_label"]) for p in usable]
    train_vals = [
        float(p["train_peak"]) if p.get("train_peak") == p.get("train_peak") else 0.0
        for p in usable
    ]
    eval_vals = [
        float(p["eval_mean"]) if p.get("eval_mean") == p.get("eval_mean") else 0.0
        for p in usable
    ]
    eval_err = [
        float(p["eval_std"]) if p.get("eval_std") == p.get("eval_std") else 0.0
        for p in usable
    ]
    x = list(range(len(labels)))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        train_vals,
        width=w,
        color="#457b9d",
        label="Train peak (logged)",
        zorder=2,
    )
    ax.bar(
        [i + w / 2 for i in x],
        eval_vals,
        width=w,
        yerr=eval_err,
        color="#e76f51",
        label="Eval base mean",
        capsize=3,
        zorder=2,
        error_kw={"elinewidth": 1},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def plot_point_farming_scores(
    rows: list[dict[str, Any]],
    *,
    title: str = "Point-farming mods — absolute score",
    farming_mods: Sequence[str] | None = None,
    ax=None,
):
    plt = _require_matplotlib()
    mods = list(farming_mods) if farming_mods is not None else sorted(POINT_FARMING_MODS)
    by_pair: dict[tuple[str, str], float] = {}
    agents: set[str] = set()
    for row in rows:
        mod = str(row.get("mod_key") or "")
        if mod not in mods:
            continue
        mean = row.get("mean")
        if mean is None or mean != mean:
            continue
        agent = str(row["agent_label"])
        by_pair[(mod, agent)] = float(mean)
        agents.add(agent)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 4.5))
    agent_list = sorted(agents)
    if not agent_list or not any(m in {k[0] for k in by_pair} for m in mods):
        ax.text(0.5, 0.5, "No point-farming rows", ha="center", va="center")
        ax.set_axis_off()
        return ax.figure if created else ax

    x = np.arange(len(mods))
    width = min(0.8 / max(1, len(agent_list)), 0.18)
    cmap = plt.get_cmap("tab10")
    for i, agent in enumerate(agent_list):
        vals = [by_pair.get((mod, agent), np.nan) for mod in mods]
        offset = (i - (len(agent_list) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=agent, color=cmap(i % 10))
    ax.set_xticks(x)
    ax.set_xticklabels(mods, rotation=15, ha="right")
    ax.set_ylabel("Mean episode return (unclipped score)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    _style(ax)
    if created:
        ax.figure.tight_layout()
    return ax.figure if created else ax


def _canonical_rq1_agents() -> list[str]:
    return ["skills/ego", "skills/pixel", "skills/oc", "pqn/pixel", "pqn/oc", "heuristic"]


def _canonical_rq3_agents() -> list[str]:
    return [
        "meta(pixel)+skills(ego)",
        "meta(oc)+skills(ego)",
        "tandem(pixel)+skills(ego)",
        "tandem(oc)+skills(ego)",
        "skills/ego",
        "heuristic",
    ]


def plot_rq1_summary(rows: list[dict[str, Any]], *, title: str = "RQ1 TLDR"):
    plt = _require_matplotlib()
    agents = _canonical_rq1_agents()
    bases = dict(base_returns_by_agent(rows))
    dyn_med = median_retention_by_agent(rows, category="dynamics")
    fig, (ax_base, ax_dyn) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(agents))
    ax_base.bar(
        x,
        [bases.get(a, (np.nan, 0.0))[0] for a in agents],
        yerr=[bases.get(a, (np.nan, 0.0))[1] for a in agents],
        capsize=3,
        color="#3d5a80",
    )
    ax_base.set_xticks(x)
    ax_base.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
    ax_base.set_ylabel("Base return")
    ax_base.set_title("Greedy eval")
    _style(ax_base)
    ax_dyn.bar(x, [dyn_med.get(a, np.nan) for a in agents], color="#2a9d8f")
    ax_dyn.axhline(1.0, color="0.4", lw=1, ls="--")
    ax_dyn.set_xticks(x)
    ax_dyn.set_xticklabels(agents, rotation=25, ha="right", fontsize=8)
    ax_dyn.set_ylabel("Median retention")
    ax_dyn.set_title("Dynamics mods")
    _style(ax_dyn)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_rq2_summary(rows: list[dict[str, Any]], *, title: str = "RQ2 TLDR"):
    plt = _require_matplotlib()
    envs, modes, matrix = transfer_base_matrix(rows)
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.55 * len(envs) + 1.5)))
    if not np.any(np.isfinite(arr)):
        ax.text(0.5, 0.5, "No zero-shot transfer base rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(modes)))
    ax.set_yticks(np.arange(len(envs)))
    ax.set_xticklabels([f"transfer/{m}" for m in modes], rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(envs)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Base return")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    return fig


def plot_rq2_retention_summary(rows: list[dict[str, Any]], *, title: str = "RQ2 TLDR retention"):
    plt = _require_matplotlib()
    transfer_rows = [
        r
        for r in rows
        if str(r.get("agent_label") or "").startswith("transfer/")
        and r.get("mod_key") != "base"
        and r.get("mod_key") not in POINT_FARMING_MODS
    ]
    envs = sorted({str(r["env"]) for r in transfer_rows if r.get("env")})
    modes = ["ego", "pixel", "oc", "pqn"]
    means: dict[tuple[str, str], float] = {}
    for env in envs:
        for mode in modes:
            label = f"transfer/{mode}"
            rets = [
                float(r["retention"])
                for r in transfer_rows
                if r.get("env") == env and r.get("agent_label") == label
                if r.get("retention") == r.get("retention")
            ]
            if rets:
                means[(env, mode)] = sum(rets) / len(rets)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not means:
        ax.text(0.5, 0.5, "No transfer retention rows", ha="center", va="center")
        ax.set_axis_off()
        return fig
    x = np.arange(len(envs))
    width = 0.18
    cmap = plt.get_cmap("tab10")
    for i, mode in enumerate(modes):
        vals = [means.get((env, mode), np.nan) for env in envs]
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=f"transfer/{mode}", color=cmap(i))
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15, ha="right")
    ax.set_ylabel("Mean retention")
    ax.set_title(title)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_rq3_summary(rows: list[dict[str, Any]], *, title: str = "RQ3 TLDR"):
    plt = _require_matplotlib()
    agents = _canonical_rq3_agents()
    bases = dict(base_returns_by_agent(rows))
    rets = mean_retention_by_agent(
        [r for r in rows if r.get("mod_key") not in POINT_FARMING_MODS]
    )
    fig, (ax_base, ax_ret) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(agents))
    ax_base.bar(
        x,
        [bases.get(a, (np.nan, 0.0))[0] for a in agents],
        yerr=[bases.get(a, (np.nan, 0.0))[1] for a in agents],
        capsize=3,
        color="#457b9d",
    )
    ax_base.set_xticks(x)
    ax_base.set_xticklabels(agents, rotation=30, ha="right", fontsize=7)
    ax_base.set_ylabel("Base return")
    _style(ax_base)
    ax_ret.bar(x, [rets.get(a, np.nan) for a in agents], color="#e76f51")
    ax_ret.axhline(1.0, color="0.4", lw=1, ls="--")
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(agents, rotation=30, ha="right", fontsize=7)
    ax_ret.set_ylabel("Mean retention")
    _style(ax_ret)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def save_figure(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

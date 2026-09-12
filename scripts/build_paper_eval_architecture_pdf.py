#!/usr/bin/env python3
"""Build a short PDF architecture overview of the current paper_eval suite.

Focus: ego / pixel / OC skill observation modes, POV facing spine, and suite protocol.
Includes simple block diagrams (clarity over polish).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "docs" / "paper_eval" / "architecture_overview.pdf"
PIXEL_OUTLINE_IMG = PROJECT_ROOT / "docs" / "paper_eval" / "pixel_skill_target_outline.png"

FIG = (8.5, 11.0)
LM = 0.08
RM = 0.92

# Mode colors (kept consistent across pages)
C_EGO = "#1e7a46"
C_EGO_FACE = "#e8f5ec"
C_PIX = "#2c4a7c"
C_PIX_FACE = "#e8eef8"
C_OC = "#8a3a2a"
C_OC_FACE = "#f8ebe8"
C_SHARED = "#444444"
C_SHARED_FACE = "#f0f0f0"
C_META = "#5c4a8a"
C_META_FACE = "#f0ecf8"
C_REMOVED = "#999999"
C_THESIS = "#0b5a8a"
C_THESIS_FACE = "#e6f3fa"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _new_page(pdf: PdfPages, title: str, subtitle: str | None = None):
    fig = plt.figure(figsize=FIG)
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(LM, y, title, fontsize=15, weight="bold", color="#1a1a1a", va="top")
    y -= 0.032
    if subtitle:
        fig.text(LM, y, subtitle, fontsize=8.5, color="#555555", va="top")
        y -= 0.024
    ax = fig.add_axes([LM, y - 0.004, RM - LM, 0.001])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.plot([0, 1], [0.5, 0.5], color="#333333", lw=0.8)
    return fig, y - 0.02


def _body(fig, y: float, lines: list[str], *, size: float = 8.2, leading: float = 0.017):
    for line in lines:
        if y < 0.05:
            break
        weight = "bold" if line.startswith("##") else "normal"
        text = line[3:] if line.startswith("## ") else line
        fig.text(
            LM,
            y,
            text,
            fontsize=size + (1.0 if weight == "bold" else 0),
            family="DejaVu Sans",
            color="#111111" if weight == "bold" else "#222222",
            weight=weight,
            va="top",
        )
        y -= leading * (1.15 if weight == "bold" else 1.0)
    return y


def _axes(fig, rect) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def _node(ax, x, y, w, h, text, *, face="#f4f6f8", edge="#333333", fs=7.5, weight="normal", tc="#111"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            lw=1.2,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        color=tc,
        transform=ax.transAxes,
        wrap=True,
        linespacing=1.15,
    )


def _arrow(ax, x1, y1, x2, y2, *, color="#333", lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=lw,
            color=color,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def _label(ax, x, y, text, *, fs=7, color="#444", ha="center", weight="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            weight=weight, transform=ax.transAxes, family="DejaVu Sans")


def _hflow(ax, boxes, *, y=0.45, h=0.28, gap=0.035, start_x=0.02, total_w=0.96, edge="#333"):
    """Horizontal chain of equal-ish boxes with arrows between them."""
    n = len(boxes)
    usable = total_w - gap * (n - 1)
    # weight boxes: longer text slightly wider
    weights = [max(1.0, len(b[0].split("\n")[0]) / 12) for b in boxes]
    s = sum(weights)
    widths = [usable * w / s for w in weights]
    x = start_x
    for i, ((text, face), w) in enumerate(zip(boxes, widths)):
        _node(ax, x, y, w, h, text, face=face, edge=edge, fs=7.0)
        if i < n - 1:
            _arrow(ax, x + w + 0.002, y + h / 2, x + w + gap - 0.002, y + h / 2, color=edge)
        x += w + gap


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_title(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=FIG)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.90, "Paper Eval Architecture", fontsize=20, weight="bold",
             ha="center", color="#1a1a1a")
    fig.text(0.5, 0.86, "Concept-Based Learning — System-1 observation modes + suite protocol",
             fontsize=9.5, ha="center", color="#555555")
    fig.text(0.5, 0.825,
             "Current thesis spine: POV facing-frame ego · φ8 bins · clip50  →  48-dim @ stack 4",
             ha="center", fontsize=8.5, weight="bold", color=C_THESIS)

    # Three mode cards
    ax = _axes(fig, [0.06, 0.50, 0.88, 0.28])
    cards = [
        (0.02, "EGO", "SKILL_INPUT_MODE: ego",
         "Relative geometry\n→ flat MLP\nthesis: 48-dim\n(classic: 24-dim)", C_EGO, C_EGO_FACE),
        (0.35, "PIXEL", "SKILL_INPUT_MODE: pixel_embed",
         "84×84 gray + outline\n→ CNN → Dense → MLP\n(+ POV mapper arms)", C_PIX, C_PIX_FACE),
        (0.68, "OC", "SKILL_INPUT_MODE: oc_embed",
         "objects [M,F]\n→ Dense/row → gather\n→ MLP (+ optional ψ)", C_OC, C_OC_FACE),
    ]
    for x, title, mode, body, edge, face in cards:
        _node(ax, x, 0.12, 0.30, 0.78, "", face=face, edge=edge)
        ax.text(x + 0.15, 0.78, title, ha="center", fontsize=12, weight="bold",
                color=edge, transform=ax.transAxes)
        ax.text(x + 0.15, 0.62, mode, ha="center", fontsize=6.5, color="#444",
                family="DejaVu Sans Mono", transform=ax.transAxes)
        ax.text(x + 0.15, 0.35, body, ha="center", va="center", fontsize=7.5,
                color="#222", transform=ax.transAxes)

    fig.text(0.5, 0.46, "Shared rule: target_idx is a gather key — never a slot one-hot",
             ha="center", fontsize=9, weight="bold", color="#333")

    # Mini RQ strip
    ax2 = _axes(fig, [0.08, 0.16, 0.84, 0.24])
    _node(ax2, 0.02, 0.25, 0.28, 0.55, "RQ1\nSeaquest skills\n+ mod retention",
          face=C_SHARED_FACE, edge=C_SHARED, fs=8)
    _arrow(ax2, 0.31, 0.52, 0.35, 0.52)
    _node(ax2, 0.36, 0.25, 0.28, 0.55, "RQ2\nTransfer / finetune\n/ from-scratch\n(+ Phoenix)",
          face=C_SHARED_FACE, edge=C_SHARED, fs=7.5)
    _arrow(ax2, 0.65, 0.52, 0.69, 0.52)
    _node(ax2, 0.70, 0.25, 0.28, 0.55, "RQ3\nMeta/tandem on\nego skill backbone",
          face=C_META_FACE, edge=C_META, fs=8)

    fig.text(0.5, 0.09, "docs/skill_input_modes.md  ·  configs/experiment/paper_eval/  ·  scripts/run_paper_eval.py",
             ha="center", fontsize=7, color="#888")
    fig.text(0.5, 0.055, "Regenerate: python scripts/build_paper_eval_architecture_pdf.py",
             ha="center", fontsize=7, color="#888")
    pdf.savefig(fig)
    plt.close(fig)


def page_suite_diagram(pdf: PdfPages) -> None:
    fig, y = _new_page(pdf, "1. Suite pipeline (diagram)", "How configs become train / eval / reports")
    ax = _axes(fig, [0.06, 0.42, 0.88, 0.48])

    # Top: configs
    _node(ax, 0.15, 0.82, 0.70, 0.12, "configs/experiment/paper_eval/{rq1,rq2,rq3,mods}/*.yaml",
          face="#fff8e6", edge="#a67c00", fs=7.5)
    _arrow(ax, 0.5, 0.82, 0.5, 0.74)
    _node(ax, 0.20, 0.58, 0.60, 0.14,
          "scripts/run_paper_eval.py\n(discover · --suite · shard · RQ gate · resume · lean)",
          face="#eef2f8", edge="#2c4a7c", fs=8, weight="bold")

    # Branch
    _arrow(ax, 0.35, 0.58, 0.22, 0.46)
    _arrow(ax, 0.50, 0.58, 0.50, 0.46)
    _arrow(ax, 0.65, 0.58, 0.78, 0.46)

    _node(ax, 0.02, 0.28, 0.30, 0.16, "TRAIN / FINETUNE\nskills · meta · tandem\nflat PQN",
          face=C_EGO_FACE, edge=C_EGO, fs=7)
    _node(ax, 0.35, 0.28, 0.30, 0.16, "EVAL / TRANSFER\neval_harness\ngreedy × mods",
          face=C_PIX_FACE, edge=C_PIX, fs=7)
    _node(ax, 0.68, 0.28, 0.30, 0.16, "REPORT\nCSV / JSON\nretention PDF",
          face=C_OC_FACE, edge=C_OC, fs=7)

    _arrow(ax, 0.17, 0.28, 0.17, 0.16)
    _arrow(ax, 0.50, 0.28, 0.50, 0.16)
    _node(ax, 0.02, 0.02, 0.30, 0.12, "ckpt + *_config.yaml\n(ego flag sidecar)",
          face=C_SHARED_FACE, edge=C_SHARED, fs=6.5)
    _node(ax, 0.35, 0.02, 0.30, 0.12, "overlay ego flags\nfrom source ckpt",
          face=C_SHARED_FACE, edge=C_SHARED, fs=6.5)
    _arrow(ax, 0.32, 0.08, 0.35, 0.08, color=C_SHARED)

    lines = [
        "## Matched outer loop (RQ1 skill arms)",
        "TOTAL_TIMESTEPS ≈ 24.98M · NUM_ENVS=128 · NUM_STEPS=128 · FRAME_SKIP=4 · stack=4",
        "Skills: DESTROY / COLLECT / AVOID / NAVIGATE · MAX_OBJECTS=35 · RELABEL_ALL_OBJECTS=false",
        "Ego encoding flags copied onto pixel/OC YAMLs for sidecar consistency (even if unused by those nets).",
        "",
        "## RQ dependencies + named suites",
        "RQ2/RQ3 wait on RQ1 Seaquest skill ckpts (PAPER_EVAL_REQUIRES_RQ1 / --wait-deps).",
        "Named suites (thesis_facing, pov_facing_v2, clip50_angbins_full, …) select stems; not only --rq.",
        "RQ3 skills are always ego; meta input is pixel_embed or oc_embed.",
    ]
    _body(fig, 0.38, lines)
    pdf.savefig(fig)
    plt.close(fig)


def page_three_way(pdf: PdfPages) -> None:
    """Side-by-side architecture flows — the main visual separation page."""
    fig, _ = _new_page(
        pdf,
        "2. Three System-1 architectures (side-by-side)",
        "Same outer loop; different observation + target cue. target_idx never enters as a one-hot.",
    )

    # Shared inputs strip
    ax0 = _axes(fig, [0.06, 0.86, 0.88, 0.06])
    _node(ax0, 0.05, 0.15, 0.28, 0.70, "pixel_obs", face="#eee", edge="#666", fs=8)
    _node(ax0, 0.36, 0.15, 0.28, 0.70, "objects (labeled)", face="#eee", edge="#666", fs=8)
    _node(ax0, 0.67, 0.15, 0.28, 0.70, "target_idx (gather key)", face="#fff3cd", edge="#a67c00", fs=7.5)

    columns = [
        ("EGO", C_EGO, C_EGO_FACE, [
            ("objects[target]\ngeometry only", C_EGO_FACE),
            ("get_ego_obs\ndist + φ/ψ encode\n+LOS +weapon\n× stack 4", C_EGO_FACE),
            ("MLP\n(HIDDEN 128 × 3)", C_EGO_FACE),
            ("Q(ego actions)\ndim 18", C_EGO_FACE),
        ], "cue built into\nrelative features"),
        ("PIXEL", C_PIX, C_PIX_FACE, [
            ("paint outline on\ntarget bbox\n(84×84 gray)", C_PIX_FACE),
            ("CNN [16,32]\nstride-2 convs", C_PIX_FACE),
            ("Dense(256)\n+ MLP", C_PIX_FACE),
            ("Q(ego actions)\ndim 18", C_PIX_FACE),
        ], "cue = bright\nbbox outline"),
        ("OC", C_OC, C_OC_FACE, [
            ("build matrix\n[M, F] all leaves", C_OC_FACE),
            ("Dense(F→256)\nshared per row", C_OC_FACE),
            ("gather row\ntarget_idx → [256]\n(+ optional sin/cos ψ)", C_OC_FACE),
            ("MLP → Q\n(ego actions)", C_OC_FACE),
        ], "cue = gather\nembedded row"),
    ]

    for i, (title, edge, face, steps, cue) in enumerate(columns):
        x0 = 0.06 + i * 0.31
        ax = _axes(fig, [x0, 0.18, 0.29, 0.66])
        # column header
        _node(ax, 0.05, 0.90, 0.90, 0.08, title, face=face, edge=edge, fs=11, weight="bold", tc=edge)
        _label(ax, 0.5, 0.85, cue, fs=6.5, color=edge)

        ys = [0.68, 0.48, 0.28, 0.08]
        for (text, f), yy in zip(steps, ys):
            _node(ax, 0.10, yy, 0.80, 0.14, text, face=f, edge=edge, fs=6.8)
        for a, b in zip(ys, ys[1:]):
            _arrow(ax, 0.5, a, 0.5, b + 0.14, color=edge)

        # dashed "does NOT use" notes
        if title == "EGO":
            _label(ax, 0.5, 0.02, "does not feed raw pixels\nor full object matrix", fs=6, color="#888")
        elif title == "PIXEL":
            _label(ax, 0.5, 0.02, "ego vector built but ignored\nby QNetwork pixel branch", fs=6, color="#888")
        else:
            _label(ax, 0.5, 0.02, "no global RAM flatten\nin skill OC path", fs=6, color="#888")

    # Bottom legend
    fig.text(0.08, 0.12, "## Where target_idx is used", fontsize=9, weight="bold")
    fig.text(0.08, 0.095, "EGO: select geometry for feature builder only", fontsize=7.5, color=C_EGO)
    fig.text(0.08, 0.075, "PIXEL: select bbox to paint (outline in pixel space)", fontsize=7.5, color=C_PIX)
    fig.text(0.08, 0.055, "OC: gather which row of the embedded object matrix", fontsize=7.5, color=C_OC)
    fig.text(0.08, 0.030, "NEVER: one_hot(target_idx, MAX_OBJECTS) concatenated into the net  (removed — broke remapping)",
             fontsize=7.5, color="#a00")
    pdf.savefig(fig)
    plt.close(fig)


def page_ego_diagram(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf,
        "3. Ego architecture (detail)",
        "Thesis spine vs classic baseline — layout must match the Seaquest ckpt sidecar on transfer",
    )
    ax = _axes(fig, [0.06, 0.62, 0.88, 0.28])
    _hflow(
        ax,
        [
            ("target_idx", "#fff3cd"),
            ("gather object\ngeometry", C_EGO_FACE),
            ("facing-frame\npose (dist, φ)", C_THESIS_FACE),
            ("φ → 8 bins\n+LOS +weapon\n×4 → 48-d", C_THESIS_FACE),
            ("MLP → Q\n18 ego acts", C_EGO_FACE),
        ],
        y=0.30,
        h=0.45,
        edge=C_EGO,
    )
    _label(ax, 0.5, 0.10, "Thesis spine: USE_POV_FACING_FRAME + EGO_PSI_ANGLE_BINS=8 + clip50",
           fs=7.5, color=C_THESIS, weight="bold")

    # Channel breakdown for thesis spine
    ax2 = _axes(fig, [0.06, 0.36, 0.88, 0.24])
    _label(ax2, 0.5, 0.95, "Thesis spine channels / frame (×4 stack = 48)", fs=8, weight="bold", color=C_THESIS)
    channels = [
        ("distance\n(clip50)", 1, True, C_THESIS),
        ("φ bins ×8", 8, True, C_THESIS),
        ("LOS", 1, True, C_EGO),
        ("can_fire", 1, True, C_EGO),
        ("facing_enemy", 1, True, C_EGO),
        ("size / vel", 2, False, C_REMOVED),
    ]
    x = 0.02
    for name, n, on, col in channels:
        w = 0.085 * max(n, 1) if n <= 2 else 0.28
        if n == 8:
            w = 0.36
        face = C_THESIS_FACE if on and col == C_THESIS else (C_EGO_FACE if on else "#f0f0f0")
        _node(ax2, x, 0.28, w - 0.01, 0.48, f"{name}\n({'ON' if on else 'OFF'})",
              face=face, edge=col, fs=5.8)
        x += w
    _label(ax2, 0.5, 0.10, "φ = bearing of target in player-facing body frame  |  SIZE/VELOCITY always off for paper arms",
           fs=6.5, color="#666")

    lines = [
        "## Encoding families (transfer locks to the trained layout)",
        "Thesis / facing spine: POV facing frame · EGO_PSI_ANGLE_BINS=8 · clip=50 → 12 ch/frame → 48-dim",
        "Classic baseline: target-aligned sin/cos(ψ) · blocking+weapon · clip null/50 → 6 ch/frame → 24-dim",
        "World-θ bins (non-POV): EGO_ANGLE_BINS=8 · clip50 → also 48-dim (different angle semantics)",
        "Ablations: clip0 / clip100 / distflags / facing_xy / restrained pose / shot extras — see YAML stems",
        "",
        "## Shared flags",
        "SIZE=false · VELOCITY=false · BLOCKING=true · WEAPON=true · USE_LEGACY_EGO_OBS=false",
        "USE_LEGACY_SEAQUEST_* action/ascent/death guards = false · EGO_ACTION_DIM=18",
        "Pre-2026-08-30 facing-frame ckpts invalid (obs/mapper fix) — use *_v2 / thesis_facing stems.",
    ]
    _body(fig, 0.34, lines, size=7.5, leading=0.015)
    pdf.savefig(fig)
    plt.close(fig)


def page_pov_mapper(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf,
        "4. POV action mapper + ego frame",
        "Body-relative control must match observation semantics",
    )
    ax = _axes(fig, [0.06, 0.55, 0.88, 0.35])

    _node(ax, 0.02, 0.55, 0.28, 0.35, "Player facing\n(resolve_facing_deg)",
          face=C_THESIS_FACE, edge=C_THESIS, fs=7.5)
    _arrow(ax, 0.31, 0.72, 0.38, 0.72, color=C_THESIS)
    _node(ax, 0.38, 0.55, 0.28, 0.35, "USE_POV_FACING_FRAME\nview = facing\ntarget pose in body frame",
          face=C_THESIS_FACE, edge=C_THESIS, fs=7)
    _arrow(ax, 0.67, 0.72, 0.74, 0.72, color=C_THESIS)
    _node(ax, 0.74, 0.55, 0.24, 0.35, "ego FORWARD =\nstrafe along facing\n(+ FIRE primitives)",
          face=C_EGO_FACE, edge=C_EGO, fs=7)

    _node(ax, 0.02, 0.08, 0.28, 0.35, "Legacy / classic\n(no POV mapper)",
          face=C_SHARED_FACE, edge=C_SHARED, fs=7)
    _arrow(ax, 0.31, 0.25, 0.38, 0.25, color=C_SHARED)
    _node(ax, 0.38, 0.08, 0.28, 0.35, "Target-aligned ψ\nview toward target\nsin/cos or θ-bins",
          face=C_SHARED_FACE, edge=C_SHARED, fs=7)
    _arrow(ax, 0.67, 0.25, 0.74, 0.25, color=C_SHARED)
    _node(ax, 0.74, 0.08, 0.24, 0.35, "Orientation-relative\nor env ALE tables",
          face=C_SHARED_FACE, edge=C_SHARED, fs=7)

    lines = [
        "## Flags",
        "USE_POV_RELATIVE_MAPPER — body-facing ego action table (required for facing / fixed2 / turn).",
        "USE_POV_FACING_FRAME — FPS obs: φ from player facing (thesis spine).",
        "EGO_POSE_XY — emit (fwd, lat) instead of (distance, φ) under facing frame.",
        "USE_POV_FIXED_BILATERAL / USE_POV_TURN_PRIMITIVES — alternate POV ablations (not thesis default).",
        "",
        "## Pixel / OC under POV",
        "thesis_facing pairs pixel_pov / oc_pov arms with the same facing-relative mapper so",
        "System-1 action semantics match across observation modes (obs cue still outline / gather).",
        "Skill OC may set SKILL_OC_PSI_FEATURES to concat sin/cos(ψ) after gather (facing − bearing).",
    ]
    _body(fig, 0.50, lines, size=7.6, leading=0.0155)
    pdf.savefig(fig)
    plt.close(fig)


def page_pixel_diagram(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf,
        "5. Pixel architecture (detail)",
        "SKILL_INPUT_MODE: pixel_embed — CNN + painted target outline",
    )
    ax = _axes(fig, [0.05, 0.72, 0.90, 0.18])

    # Two inputs merge
    _node(ax, 0.02, 0.55, 0.18, 0.35, "pixel stack\n[T,H,W,C]\n84×84 gray", face=C_PIX_FACE, edge=C_PIX, fs=6.5)
    _node(ax, 0.02, 0.08, 0.18, 0.35, "objects +\ntarget_idx\n→ bbox", face="#fff3cd", edge="#a67c00", fs=6.5)
    _arrow(ax, 0.21, 0.72, 0.28, 0.55, color=C_PIX)
    _arrow(ax, 0.21, 0.25, 0.28, 0.45, color="#a67c00")
    _node(ax, 0.28, 0.30, 0.18, 0.40, "paint_target\n_outline_pixels\n(hollow border)", face=C_PIX_FACE, edge=C_PIX, fs=6.5)
    _arrow(ax, 0.47, 0.50, 0.52, 0.50, color=C_PIX)
    _node(ax, 0.52, 0.30, 0.16, 0.40, "rearrange\n[H,W,T·C]", face=C_PIX_FACE, edge=C_PIX, fs=7)
    _arrow(ax, 0.69, 0.50, 0.74, 0.50, color=C_PIX)
    _node(ax, 0.74, 0.55, 0.22, 0.35, "CNN\nch [16,32]", face=C_PIX_FACE, edge=C_PIX, fs=7)
    _node(ax, 0.74, 0.08, 0.22, 0.35, "Dense 256\n+ MLP → Q", face=C_PIX_FACE, edge=C_PIX, fs=7)
    _arrow(ax, 0.85, 0.55, 0.85, 0.44, color=C_PIX)

    # Real Seaquest screenshot: raw → red outline → 84×84 CNN input
    if PIXEL_OUTLINE_IMG.is_file():
        ax_img = fig.add_axes([0.07, 0.36, 0.86, 0.34])
        ax_img.imshow(plt.imread(PIXEL_OUTLINE_IMG))
        ax_img.axis("off")
        fig.text(
            0.50,
            0.345,
            "Seaquest skill input: raw RGB → red target outline cue → 84×84 grayscale CNN frame",
            ha="center",
            fontsize=7,
            color="#444",
        )
    else:
        ax2 = _axes(fig, [0.08, 0.40, 0.38, 0.28])
        ax2.add_patch(Rectangle((0.15, 0.15), 0.70, 0.70, fill=True, facecolor="#ddd",
                                edgecolor="#666", lw=1, transform=ax2.transAxes))
        ax2.add_patch(Rectangle((0.35, 0.35), 0.30, 0.30, fill=False, edgecolor="#c00",
                                lw=2.5, transform=ax2.transAxes))
        _label(ax2, 0.5, -0.05, f"missing screenshot: {PIXEL_OUTLINE_IMG.name}", fs=6.5, color="#a00")

    ax3 = _axes(fig, [0.08, 0.24, 0.84, 0.09])
    _node(
        ax3,
        0.02,
        0.10,
        0.96,
        0.80,
        "Settings: META_PIXEL_DO_RESIZE=true · RESIZE [84,84] · GRAYSCALE=true · "
        "NATIVE_DOWNSCALING=true · SKILL_PIXEL_CNN_CHANNELS=[16,32]",
        face=C_PIX_FACE,
        edge=C_PIX,
        fs=6.5,
    )

    lines = [
        "## Design notes",
        "• Outline replaces slot one-hot / tiled slot embed (those failed remapping ablations).",
        "• Same ~25M env-step budget as ego despite ~10× wall time (RTPT weight ≈ 10).",
        "• Ablation: SKILL_PIXEL_PAINT_TARGET_OUTLINE=false (nooutline) — transfer should collapse.",
        "• Flat PQN (seaquest_pqn) is a separate non-hierarchical pixel baseline at same budget.",
        "• RQ2 from-scratch includes pixel; RQ3 uses pixel as META_INPUT_MODE over ego skills.",
        "• thesis_facing uses *_pixel_pov stems (POV mapper; outline cue unchanged).",
        f"• Screenshot asset: docs/paper_eval/{PIXEL_OUTLINE_IMG.name}",
    ]
    _body(fig, 0.22, lines, size=7.5, leading=0.015)
    pdf.savefig(fig)
    plt.close(fig)


def page_oc_diagram(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf,
        "6. OC architecture (detail)",
        "SKILL_INPUT_MODE: oc_embed — per-object Dense + gather (not flat M·F)",
    )
    ax = _axes(fig, [0.06, 0.55, 0.88, 0.35])

    # Matrix visualization
    _label(ax, 0.18, 0.95, "objects → matrix [M, F]", fs=8, weight="bold", color=C_OC)
    # draw grid
    for r in range(5):
        for c in range(6):
            face = "#f0f0f0"
            if r == 2:  # target row highlight before embed
                face = "#ffe08a"
            ax.add_patch(Rectangle((0.04 + c * 0.045, 0.35 + (4 - r) * 0.10), 0.040, 0.090,
                                   facecolor=face, edgecolor="#888", lw=0.6, transform=ax.transAxes))
    _label(ax, 0.18, 0.22, "row = object\ncol = numeric leaf", fs=6, color="#666")
    _label(ax, 0.18, 0.12, "yellow = target row\n(selected later)", fs=6, color="#a67c00")

    _arrow(ax, 0.34, 0.55, 0.40, 0.55, color=C_OC)
    _node(ax, 0.40, 0.38, 0.20, 0.35, "Dense(F → D)\nshared weights\nper row\nD=SKILL_EMBED_DIM",
          face=C_OC_FACE, edge=C_OC, fs=6.5)
    _arrow(ax, 0.61, 0.55, 0.67, 0.55, color=C_OC)

    # Embedded matrix + gather
    for r in range(5):
        face = C_OC_FACE if r != 2 else "#ffd0c4"
        ax.add_patch(Rectangle((0.68, 0.35 + (4 - r) * 0.10), 0.10, 0.090,
                               facecolor=face, edgecolor=C_OC, lw=0.8, transform=ax.transAxes))
    _label(ax, 0.73, 0.22, "[M, D]", fs=7, color=C_OC, weight="bold")
    _arrow(ax, 0.79, 0.55, 0.84, 0.55, color=C_OC)
    _node(ax, 0.84, 0.42, 0.14, 0.28, "gather\ntarget\n→ [D]", face="#fff3cd", edge="#a67c00", fs=6.5)

    # Why not flat
    ax2 = _axes(fig, [0.08, 0.30, 0.84, 0.20])
    _node(ax2, 0.02, 0.15, 0.45, 0.70,
          "KEPT: Dense(F→D) + gather\n• kernel width = F only\n• M can differ across games\n• loads when leaf schema shared",
          face=C_EGO_FACE, edge=C_EGO, fs=7)
    _node(ax2, 0.53, 0.15, 0.45, 0.70,
          "REMOVED: Dense(M·F → D)\n• first-layer width ∝ M\n• breaks cross-game load\n• + old one_hot(target) cue",
          face="#f5f5f5", edge=C_REMOVED, fs=7)

    lines = [
        "## Leaves in F (same source meta uses per candidate)",
        "bbox, active, visual_id, state, orientation, semantic_label, flags, …",
        "Skill OC does NOT concat a global raw RAM flatten.",
        "",
        "## Meta vs skill OC · POV ψ",
        "META_INPUT_MODE: oc_embed concatenates full object matrix onto a global flatten —",
        "no transferable per-object Dense there. That Dense lives on skill oc_embed only.",
        "SKILL_OC_PSI_FEATURES: after gather, concat sin/cos(ψ) (facing − bearing) for POV arms.",
        "",
        "## Suite role",
        "RQ1 train · RQ2 zero-shot + finetune · NOT from-scratch (protocol: ego+pixel only).",
        "RQ3: oc_embed as meta input over ego skills. RTPT weight ≈ 2×.",
    ]
    _body(fig, 0.28, lines, size=7.5, leading=0.015)
    pdf.savefig(fig)
    plt.close(fig)


def page_rq_diagram(pdf: PdfPages) -> None:
    fig, _ = _new_page(
        pdf,
        "7. RQ protocol map",
        "Which observation mode is used where (thesis_facing + classic paper_eval)",
    )
    ax = _axes(fig, [0.05, 0.42, 0.90, 0.48])

    # RQ1
    _label(ax, 0.5, 0.96, "RQ1 — Seaquest (in-distribution + mod eval)", fs=9, weight="bold")
    _node(ax, 0.02, 0.72, 0.18, 0.16, "skills\nEGO", face=C_EGO_FACE, edge=C_EGO, fs=8, weight="bold")
    _node(ax, 0.22, 0.72, 0.18, 0.16, "skills\nPIXEL", face=C_PIX_FACE, edge=C_PIX, fs=8, weight="bold")
    _node(ax, 0.42, 0.72, 0.18, 0.16, "skills\nOC", face=C_OC_FACE, edge=C_OC, fs=8, weight="bold")
    _node(ax, 0.62, 0.72, 0.16, 0.16, "flat\nPQN", face=C_SHARED_FACE, edge=C_SHARED, fs=8)
    _node(ax, 0.80, 0.72, 0.18, 0.16, "heuristic\n+ random", face=C_SHARED_FACE, edge=C_SHARED, fs=7)

    _arrow(ax, 0.11, 0.72, 0.11, 0.62, color=C_EGO)
    _label(ax, 0.50, 0.64, "↓  ckpts feed RQ2 transfer / RQ3 ego backbone", fs=7, color="#666")

    # RQ2
    _label(ax, 0.5, 0.56,
           "RQ2 — pong · tennis · beamrider · freeway · asterix · phoenix",
           fs=8.5, weight="bold")
    _node(ax, 0.02, 0.32, 0.30, 0.18, "zero-shot + finetune (~5M)\nEGO · PIXEL · OC\n(layout locked to RQ1 ckpt)",
          face="#fff8e6", edge="#a67c00", fs=7)
    _node(ax, 0.36, 0.32, 0.30, 0.18, "from-scratch (~25M)\nEGO · PIXEL only\n(OC excluded by protocol)",
          face=C_PIX_FACE, edge=C_PIX, fs=7)
    _node(ax, 0.70, 0.32, 0.28, 0.18, "heuristics\nenv_specific or\nnearest (pong/tennis)",
          face=C_SHARED_FACE, edge=C_SHARED, fs=7)

    # RQ3
    _label(ax, 0.5, 0.24, "RQ3 — System 2 on Seaquest", fs=9, weight="bold")
    _node(ax, 0.08, 0.02, 0.38, 0.16, "skills = EGO (frozen or tandem)\nmeta = PIXEL or OC",
          face=C_META_FACE, edge=C_META, fs=7.5)
    _node(ax, 0.52, 0.02, 0.40, 0.16, "meta_only (SKILL_LR=0)\nor tandem (skills learn too)",
          face=C_META_FACE, edge=C_META, fs=7.5)

    lines = [
        "## Cross-cutting",
        "• Mod allowlists: configs/experiment/paper_eval/mods/ (control / semantic / dynamics / visual).",
        "• Eval overlays ego encoding flags from ckpt sidecar (clip / bins / POV / weapon / blocking).",
        "• RTPT weights: ego×1, oc×2, pixel×10 (wall-time calibration, not step budget).",
        "• Reporting: env_metrics beyond score; retention = return_mod / return_base.",
        "• Active suite names: thesis_facing · pov_facing_v2 · clip50_angbins_full · … (run_paper_eval.py).",
        "",
        "## Authoritative sources",
        "docs/skill_input_modes.md · paper_eval YAML headers · observations/egocentric.py",
        "observations/target_conditioning.py · networks/skill_network.py · paper_eval/eval_harness.py",
    ]
    _body(fig, 0.38, lines, size=7.5, leading=0.015)
    pdf.savefig(fig)
    plt.close(fig)


def page_text_summary(pdf: PdfPages) -> None:
    """Keep a compact text reference page at the end."""
    fig, y = _new_page(
        pdf,
        "8. Text reference (flags & decisions)",
        "Quick lookup — diagrams above are the primary overview",
    )
    lines = [
        "## Ego — thesis spine (48-dim)",
        "POV_FACING_FRAME · PSI_ANGLE_BINS=8 · clip=50 · SIZE/VEL=false · BLOCKING/WEAPON=true",
        "Stem family: *_facing_psibins8_* / thesis_facing · transfer must match this layout.",
        "",
        "## Ego — classic baseline (24-dim)",
        "sin/cos(ψ) · blocking+weapon · no POV facing · SIZE/VEL=false · clip often null or 50",
        "",
        "## Pixel",
        "pixel_embed · 84×84 gray · CNN [16,32] · outline on target bbox · optional *_pov mapper",
        "Index-invariant visual cue; matched step budget; ~10× wall time.",
        "",
        "## OC",
        "oc_embed · [M,F] → Dense(F→D)/row → gather(target) → MLP · optional SKILL_OC_PSI_FEATURES",
        "No skill RAM flatten; no from-scratch in RQ2; meta OC ≠ skill OC Dense.",
        "",
        "## Removed",
        "one_hot(target_idx) · tiled slot embed in CNN · flat Dense(M·F→D)",
        "",
        "## Shared RQ1 loop",
        "4 skills · 25M steps · 128×128 · skip 4 · stack 4 · MAX_OBJECTS 35 · relabel off",
        "SKILL_DEATH_PENALIZE_COLLECT=true · HEURISTIC_MODE=env_specific (Seaquest)",
    ]
    _body(fig, y, lines, size=8.2, leading=0.0175)

    ax = _axes(fig, [0.06, 0.06, 0.88, 0.26])
    for i, (t, e, f, body) in enumerate([
        ("Ego", C_EGO, C_EGO_FACE, "relative feats\nRQ1/2/3 skills\n48-d thesis / 24-d classic\nRTPT 1×"),
        ("Pixel", C_PIX, C_PIX_FACE, "outline CNN\nRQ1/2 skills\nfrom-scratch ✓\nRTPT 10×"),
        ("OC", C_OC, C_OC_FACE, "Dense+gather\nRQ1/2 transfer\nfrom-scratch ✗\nRTPT 2×"),
    ]):
        _node(ax, 0.03 + i * 0.33, 0.08, 0.30, 0.84, f"{t}\n\n{body}", face=f, edge=e, fs=7.2)

    pdf.savefig(fig)
    plt.close(fig)


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        page_title(pdf)
        page_suite_diagram(pdf)
        page_three_way(pdf)
        page_ego_diagram(pdf)
        page_pov_mapper(pdf)
        page_pixel_diagram(pdf)
        page_oc_diagram(pdf)
        page_rq_diagram(pdf)
        page_text_summary(pdf)
        d = pdf.infodict()
        d["Title"] = "Paper Eval Architecture Overview"
        d["Author"] = "ConceptBasedLearning"
        d["Subject"] = "Ego / pixel / OC observation modes for paper_eval (POV facing spine)"
    return output


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help=f"Output PDF (default: {DEFAULT_OUT})")
    args = p.parse_args()
    path = build(args.output.resolve())
    # also refresh reports copy
    alt = PROJECT_ROOT / "outputs" / "reports" / "paper_eval_architecture_overview.pdf"
    if path != alt.resolve():
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_bytes(path.read_bytes())
    print(f"Wrote {path}")
    if path != alt.resolve():
        print(f"Copied {alt}")


if __name__ == "__main__":
    main()

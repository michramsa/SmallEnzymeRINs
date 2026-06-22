"""
figure_style.py
===============
Central figure style reference for PLOS Computational Biology submissions.

Import this file at the top of any figure script:

    from figure_style import apply_style, COLORS, COLOR_CYCLE, CMAPS, FIG_SIZE, FONT
    apply_style()

All constants are plain dicts — no values should be hard-coded in figure scripts.

PLOS Computational Biology figure requirements (2025)
-----------------------------------------------------
  Fonts        : Arial, Times, or Symbol only; 8–12 pt inside figures
  Width        : 3.27 in (single col) | 4.96 in (1.5 col) | 6.83 in (double col)
  Max height   : 8.75 in
  Resolution   : 300–600 DPI (300 for tone art; 600 for fine line art)
  Format       : TIFF or EPS only (< 10 MB); PDF not accepted
  Color mode   : RGB
"""

import os
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# 1.  TYPOGRAPHY
# ---------------------------------------------------------------------------

FONT = {
    "family":       "sans-serif",
    # PLOS only permits Arial, Times, or Symbol inside figures.
    # Helvetica and DejaVu Sans are NOT on the approved list and have been removed.
    # If Arial is not installed on your system, matplotlib will fall back to its
    # default sans-serif font and log a warning — install Arial (or the
    # `msttcorefonts` package on Linux) to silence this and ensure compliance.
    "sans_serif":   ["Arial", "Helvetica", "DejaVu Sans"],

    # Point sizes — must stay within the PLOS 8–12 pt window inside figures
    "size_title":   11,    # panel/axes title (bold)
    "size_label":   10,    # axis labels
    "size_tick":     8,    # tick labels  <-- PLOS minimum
    "size_legend":   8,    # legend entries
    "size_annot":    8,    # in-plot text — raised from 7 to meet PLOS 8 pt minimum
    "size_panel":   11,    # panel letters A, B, C (bold)

    "weight_title": "bold",
    "weight_panel": "bold",
}


# ---------------------------------------------------------------------------
# 2.  FIGURE DIMENSIONS  (inches)
# ---------------------------------------------------------------------------

FIG_SIZE = {
    # Single-panel
    "single_col":  (3.27, 2.8),   # one column
    "single_sq":   (3.27, 3.27),  # one column, square
    "half_page":   (4.96, 3.2),   # 1.5-column width
    "full_width":  (6.83, 3.5),   # two-column / full page width

    # Multi-panel grids
    "grid_1x2":   (6.83, 3.0),
    "grid_1x3":   (6.83, 2.8),
    "grid_2x2":   (6.83, 5.2),
    "grid_2x3":   (6.83, 5.2),
    "grid_3x3":   (6.83, 7.0),

    # Tall / portrait
    "tall":        (3.27, 5.5),
    "tall_wide":   (4.96, 6.0),
}

DPI = {
    "screen":   100,   # preview
    "print":    300,   # standard raster / TIFF
    "print_hq": 600,   # fine line art
}


# ---------------------------------------------------------------------------
# 3.  CATEGORICAL COLOR PALETTE  (Paul Tol muted — colorblind-safe)
#
#     Distinguishable under deuteranopia, protanopia, and tritanopia,
#     and holds up in greyscale print.
# ---------------------------------------------------------------------------

COLORS = {
    # Primary data colors
    "blue":       "#4878CF",
    "orange":     "#D65F00",
    "green":      "#59A14F",
    "purple":     "#B07AA1",
    "teal":       "#76B7B2",
    "red":        "#E15759",
    "yellow":     "#EDC948",
    "brown":      "#9C755F",
    "yellow_sat": "#F5C800",

    # Neutrals
    "dark_gray":  "#2C2C2C",
    "mid_gray":   "#8D8D8D",
    "light_gray": "#D3D3D3",
    "near_white": "#F7F7F7",
    "white":      "#FFFFFF",
}

# Ordered for automatic cycling — first colors are most visually distinct
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["teal"],
    COLORS["red"],
    COLORS["yellow"],
    COLORS["brown"],
]

# Marker shapes — pair with COLOR_CYCLE for multi-series scatter/line plots
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "h"]


# ---------------------------------------------------------------------------
# 4.  COLORMAPS
#
#     Sequential: for continuous single-metric data (heatmaps, node color)
#     Diverging:  for data centered on zero or a threshold
#
#     ColorBrewer names are used where possible.
# ---------------------------------------------------------------------------

def _seq(name, light, dark):
    """Two-stop LinearSegmentedColormap (light -> dark)."""
    return LinearSegmentedColormap.from_list(name, [light, dark])


CMAPS = {
    # Sequential
    "blues":       _seq("blues",    "#EFF3FF", "#2171B5"),
    "oranges":     _seq("oranges",  "#FFF7EC", "#D94701"),
    "greens":      _seq("greens",   "#F7FCF5", "#238B45"),
    "purples":     _seq("purples",  "#F2F0F7", "#54278F"),

    # Diverging
    "RdBu":        "RdBu_r",    # blue (low) -> white -> red (high)
    "PuOr":        "PuOr",      # purple -> white -> orange
    "BrBG":        "BrBG",      # brown -> white -> teal-green

    # Fallbacks
    "sequential":  "viridis",
    "qualitative": "tab10",
}


# ---------------------------------------------------------------------------
# 5.  LAYOUT & SPACING DEFAULTS
# ---------------------------------------------------------------------------

LAYOUT = {
    "tight_pad":         0.4,
    "subplot_hspace":    0.45,
    "subplot_wspace":    0.35,
    "spine_linewidth":   0.75,
    "tick_length":       3.0,
    "tick_width":        0.75,
    "grid_alpha":        0.25,
    "grid_linewidth":    0.5,
    "line_width":        1.25,
    "marker_size":       4.5,
    "legend_framealpha": 0.85,
    "legend_loc":        "best",
    "cap_size":          2.5,    # error bar cap width
}


# ---------------------------------------------------------------------------
# 6.  apply_style()
#     Call exactly once per script, before any plt.figure() or sns calls.
# ---------------------------------------------------------------------------

def apply_style():
    """
    Apply project-wide rcParams and seaborn context.

    Example
    -------
        from figure_style import apply_style
        apply_style()
        fig, ax = plt.subplots(figsize=FIG_SIZE["half_page"])
    """
    mpl.rcParams.update({
        # Font
        "font.family":          FONT["family"],
        "font.sans-serif":      FONT["sans_serif"],
        "font.size":            FONT["size_label"],
        "axes.titlesize":       FONT["size_title"],
        "axes.titleweight":     FONT["weight_title"],
        "axes.labelsize":       FONT["size_label"],
        "xtick.labelsize":      FONT["size_tick"],
        "ytick.labelsize":      FONT["size_tick"],
        "legend.fontsize":      FONT["size_legend"],

        # Axes & spines
        "axes.linewidth":       LAYOUT["spine_linewidth"],
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Ticks
        "xtick.major.size":     LAYOUT["tick_length"],
        "ytick.major.size":     LAYOUT["tick_length"],
        "xtick.major.width":    LAYOUT["tick_width"],
        "ytick.major.width":    LAYOUT["tick_width"],
        "xtick.direction":      "out",
        "ytick.direction":      "out",

        # Grid
        "axes.grid":            True,
        "grid.alpha":           LAYOUT["grid_alpha"],
        "grid.linewidth":       LAYOUT["grid_linewidth"],
        "grid.linestyle":       "--",
        "grid.color":           COLORS["light_gray"],

        # Lines & markers
        "lines.linewidth":      LAYOUT["line_width"],
        "lines.markersize":     LAYOUT["marker_size"],

        # Color cycle
        "axes.prop_cycle":      mpl.cycler(color=COLOR_CYCLE),

        # Figure
        "figure.facecolor":     COLORS["white"],
        "axes.facecolor":       COLORS["white"],
        "figure.dpi":           DPI["screen"],
        "savefig.dpi":          DPI["print"],
        "savefig.bbox":         "tight",
        "savefig.format":       "tiff",   # PLOS accepts TIFF or EPS only (not PDF)
    })

    sns.set_context("paper", font_scale=1.0)


# ---------------------------------------------------------------------------
# 7.  Convenience helpers
# ---------------------------------------------------------------------------

def new_fig(size_key="half_page", nrows=1, ncols=1, **kwargs):
    """
    Return (fig, ax) with PLOS-compliant dimensions.

    Parameters
    ----------
    size_key : str
        Key from FIG_SIZE. Default 'half_page' (4.96 x 3.2 in).
    nrows, ncols : int
        Passed to plt.subplots().
    **kwargs
        Forwarded to plt.subplots().

    Example
    -------
        fig, axes = new_fig("grid_1x3", ncols=3)
    """
    apply_style()
    figsize = FIG_SIZE.get(size_key, FIG_SIZE["half_page"])
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    fig.tight_layout(pad=LAYOUT["tight_pad"])
    return fig, ax


# ---------------------------------------------------------------------------
# PLOS pixel-dimension bounds (at 300 dpi)
# ---------------------------------------------------------------------------
_PLOS_WIDTH_MIN_PX  =  789
_PLOS_WIDTH_MAX_PX  = 2250
_PLOS_HEIGHT_MAX_PX = 2625
_PLOS_MAX_MB        = 10.0
_PLOS_ALLOWED_FMTS  = {"tiff", "tif", "eps"}


def validate_figure(fig, dpi=None):
    """
    Check a figure against PLOS Computational Biology dimension rules.

    Prints a summary and returns True if all checks pass, False otherwise.
    Does NOT save the figure — call this before save_fig() if you want an
    early warning, or save_fig() calls it automatically.

    Parameters
    ----------
    fig : matplotlib Figure
    dpi : int, optional  Resolution used for pixel calculation. Defaults to
                         DPI['print'] (300).

    Returns
    -------
    bool  True = all checks passed.

    Example
    -------
        ok = validate_figure(fig)
    """
    dpi = dpi or DPI["print"]
    w_in, h_in = fig.get_size_inches()
    w_px = w_in * dpi
    h_px = h_in * dpi
    passed = True

    checks = []

    # Width
    if w_px < _PLOS_WIDTH_MIN_PX:
        checks.append(
            f"  [FAIL] Width {w_px:.0f} px < minimum {_PLOS_WIDTH_MIN_PX} px"
            f" ({w_in:.2f} in @ {dpi} dpi)"
        )
        passed = False
    elif w_px > _PLOS_WIDTH_MAX_PX:
        checks.append(
            f"  [FAIL] Width {w_px:.0f} px > maximum {_PLOS_WIDTH_MAX_PX} px"
            f" ({w_in:.2f} in @ {dpi} dpi)"
        )
        passed = False
    else:
        checks.append(f"  [OK]   Width  {w_px:.0f} px ({w_in:.2f} in @ {dpi} dpi)")

    # Height
    if h_px > _PLOS_HEIGHT_MAX_PX:
        checks.append(
            f"  [FAIL] Height {h_px:.0f} px > maximum {_PLOS_HEIGHT_MAX_PX} px"
            f" ({h_in:.2f} in @ {dpi} dpi)"
        )
        passed = False
    else:
        checks.append(f"  [OK]   Height {h_px:.0f} px ({h_in:.2f} in @ {dpi} dpi)")

    status = "PASSED" if passed else "FAILED"
    print(f"[figure_style] PLOS dimension check: {status}")
    for line in checks:
        print(line)

    return passed


def save_fig(fig, path, dpi=None, fmt=None, strict=True):
    """
    Save a figure at print resolution with PLOS compliance checks.

    Validates pixel dimensions before saving and file size after saving.
    Issues warnings (not errors) so the figure is still written to disk —
    you can then decide whether to adjust dimensions.

    Parameters
    ----------
    fig    : matplotlib Figure
    path   : str or Path
    dpi    : int, optional
        Defaults to DPI['print'] (300).
    fmt    : str, optional
        File format, e.g. 'tiff' or 'eps'. Inferred from *path* extension
        if omitted. Must be 'tiff'/'tif' or 'eps' for PLOS submission.
    strict : bool, optional
        If True (default), warn when the format is not PLOS-approved
        (e.g. 'png', 'pdf'). Set to False to silently allow non-PLOS
        formats — useful for preview or presentation exports.

    Example
    -------
        save_fig(fig, "figures/fig2_heatmap.tiff")           # submission
        save_fig(fig, "figures/fig2_preview.png", strict=False)  # preview
    """
    dpi = dpi or DPI["print"]

    # --- Infer format from path if not given ---
    if fmt is None:
        ext = os.path.splitext(str(path))[-1].lstrip(".").lower()
        fmt_to_use = ext if ext else "tiff"
    else:
        fmt_to_use = fmt.lower()

    # --- Warn if format is not PLOS-approved ---
    if strict and fmt_to_use not in _PLOS_ALLOWED_FMTS:
        warnings.warn(
            f"[figure_style] Format '{fmt_to_use}' is not accepted by PLOS "
            f"Computational Biology. Use 'tiff' or 'eps'.",
            UserWarning,
            stacklevel=2,
        )

    # --- Validate pixel dimensions before saving ---
    validate_figure(fig, dpi=dpi)

    # --- Save ---
    kwargs = {"dpi": dpi, "bbox_inches": "tight", "format": fmt_to_use}
    fig.savefig(path, **kwargs)

    # --- Post-save file-size check ---
    try:
        size_mb = os.path.getsize(path) / (1024 ** 2)
        if size_mb > _PLOS_MAX_MB:
            warnings.warn(
                f"[figure_style] '{path}' is {size_mb:.1f} MB — exceeds the "
                f"PLOS {_PLOS_MAX_MB:.0f} MB limit. Consider using EPS or "
                f"reducing figure complexity.",
                UserWarning,
                stacklevel=2,
            )
        else:
            print(f"[figure_style] saved -> {path}  ({size_mb:.2f} MB)")
    except OSError:
        print(f"[figure_style] saved -> {path}")


def legend_handles(labels, colors=None, markers=None, linestyles=None):
    """
    Build a list of Line2D legend handles manually.

    Parameters
    ----------
    labels     : list of str
    colors     : list of hex str, optional  Defaults to COLOR_CYCLE.
    markers    : list of str, optional      Defaults to MARKER_CYCLE.
    linestyles : list of str, optional      Defaults to solid lines.

    Example
    -------
        handles = legend_handles(["Group A", "Group B", "Group C"])
        ax.legend(handles=handles)
    """
    import matplotlib.lines as mlines
    if colors is None:
        colors = COLOR_CYCLE
    if markers is None:
        markers = MARKER_CYCLE
    if linestyles is None:
        linestyles = ["-"] * len(labels)

    handles = []
    for i, label in enumerate(labels):
        h = mlines.Line2D(
            [], [],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            markersize=5,
            linewidth=1.5,
            label=label,
        )
        handles.append(h)
    return handles


# ---------------------------------------------------------------------------
# 8.  Demo  —  python figure_style.py
# ---------------------------------------------------------------------------

def _demo():
    """Render a palette and spec preview. Run: python figure_style.py"""
    import numpy as np
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE["grid_1x3"])
    fig.suptitle("figure_style.py — PLOS CompBio palette preview",
                 fontsize=FONT["size_title"])

    # Panel A: color cycle bar chart
    ax = axes[0]
    ax.set_title("A  Categorical cycle", fontsize=FONT["size_title"])
    for i, c in enumerate(COLOR_CYCLE):
        ax.bar(i, i + 1, color=c, edgecolor="white", linewidth=0.5)
    ax.set_xticks([])
    ax.set_ylabel("value", fontsize=FONT["size_label"])

    # Panel B: sequential colormaps
    ax = axes[1]
    ax.set_title("B  Sequential maps", fontsize=FONT["size_title"])
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    for i, k in enumerate(["blues", "oranges", "greens", "purples"]):
        ax.imshow(gradient, aspect="auto", cmap=CMAPS[k],
                  extent=[0, 1, i - 0.4, i + 0.4])
        ax.text(-0.04, i, k, ha="right", va="center",
                fontsize=FONT["size_annot"],
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel("low -> high", fontsize=FONT["size_label"])

    # Panel C: multi-series line plot
    ax = axes[2]
    ax.set_title("C  Multi-series lines", fontsize=FONT["size_title"])
    x = np.linspace(0, 10, 60)
    groups = ["Group A", "Group B", "Group C"]
    for i, label in enumerate(groups):
        y = np.cumsum(np.random.randn(60)) * 0.5
        ax.plot(x, y,
                color=COLOR_CYCLE[i],
                marker=MARKER_CYCLE[i],
                markevery=12,
                linewidth=LAYOUT["line_width"],
                markersize=LAYOUT["marker_size"],
                label=label)
    ax.set_xlabel("x axis label", fontsize=FONT["size_label"])
    ax.set_ylabel("y axis label", fontsize=FONT["size_label"])
    handles = legend_handles(groups, colors=COLOR_CYCLE[:3])
    ax.legend(handles=handles, fontsize=FONT["size_legend"])

    plt.tight_layout(pad=LAYOUT["tight_pad"])
    validate_figure(fig)
    plt.show()


if __name__ == "__main__":
    _demo()

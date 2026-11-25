"""
Figures for the small-world efficiency analysis (companion to
small_world_efficiency_colab_ver2.py).

Reads the per-PDB-id efficiency CSVs written by that script and produces:

  Figure set 1 - Latora-Marchiori phase plots, ONE FILE PER DATASET
      "rand" variant : x = E_glob / E_glob_rand   (near 1 -> random-like paths)
                       y = E_loc  / E_loc_rand    (>> 1   -> more local than random)
      "lat"  variant : x = E_glob / E_glob_lat    (> 1    -> shorter paths than lattice)
                       y = E_loc  / E_loc_lat     (near 1 -> lattice-like locality)
      Each point is one graph. A shaded box marks the small-world region and the
      panel reports the fraction of graphs inside it.

  Figure set 2 - Size / density dependence, ONE FILE PER (ratio, x-variable)
      y = a ratio, x = n_nodes (log) by default (density / mean_degree optional),
      all datasets overlaid and coloured by dataset, with a binned-median trend
      line per dataset. Answers: "is small-worldness just a size artifact?"

Also written:
  <output_dir>/all_graphs_efficiency<suffix>.csv   every graph, all datasets
  <output_dir>/size_dependence_stats<suffix>.csv   Spearman rho / p per
                                                   dataset x ratio x x-variable

IMPORTANT: the small-world region thresholds in `thresholds` are conventions,
not laws. Latora & Marchiori define small-world qualitatively (E_glob close to
random, E_loc close to lattice). Choose thresholds you can defend and report
them; the figures draw whatever you set.

Every figure is saved as its own file. Formats listed in `formats` that PLOS
accepts ('tiff', 'eps') go through figure_style.save_fig (with its dimension
checks); other formats (e.g. 'png') are saved directly.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.stats import spearmanr

try:
    from figure_style import (apply_style, COLOR_CYCLE, MARKER_CYCLE, COLORS,
                              FIG_SIZE, FONT, LAYOUT, save_fig)
except ImportError:  # minimal stand-in so the script still runs without it
    COLOR_CYCLE = ["#4878CF", "#D65F00", "#59A14F", "#B07AA1",
                   "#76B7B2", "#E15759", "#EDC948", "#9C755F"]
    MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "h"]
    COLORS = {"mid_gray": "#8D8D8D", "light_gray": "#D3D3D3",
              "dark_gray": "#2C2C2C"}
    FIG_SIZE = {"half_page": (4.96, 3.2), "single_sq": (3.27, 3.27),
                "full_width": (6.83, 3.5)}
    FONT = {"size_title": 11, "size_label": 10, "size_tick": 8,
            "size_legend": 8, "size_annot": 8}
    LAYOUT = {"tight_pad": 0.4}

    def apply_style():
        matplotlib.rcParams.update({
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
            "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 10,
            "legend.fontsize": 8, "savefig.bbox": "tight"})

    def save_fig(fig, path, dpi=300, fmt=None):
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    format=fmt or Path(path).suffix.lstrip("."))


PLOS_FORMATS = {"tiff", "tif", "eps"}
SW_GREEN = "#59A14F"

# axis definitions -----------------------------------------------------------
RATIO_LABELS = {
    "Eg_ratio_rand": r"$E_{glob}\,/\,E_{glob,rand}$",
    "El_ratio_rand": r"$E_{loc}\,/\,E_{loc,rand}$",
    "Eg_ratio_lat":  r"$E_{glob}\,/\,E_{glob,lat}$",
    "El_ratio_lat":  r"$E_{loc}\,/\,E_{loc,lat}$",
}
XVAR_LABELS = {
    "n_nodes": "Number of nodes",
    "density": "Graph density",
    "mean_degree": "Mean degree (2m / n)",
}
XVAR_LOG = {"n_nodes": True, "density": True, "mean_degree": False}

# phase-plot variants: (x column, y column, x log?, y log?, threshold keys)
PHASE_VARIANTS = {
    "rand": dict(x="Eg_ratio_rand", y="El_ratio_rand", xlog=False, ylog=True),
    "lat":  dict(x="Eg_ratio_lat",  y="El_ratio_lat",  xlog=True,  ylog=False),
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_results(results_root, labels, random_null, use_largest_component):
    """Concatenate every per-PDB-id efficiency CSV of every dataset label."""
    suffix = f"_{random_null}" + ("_lcc" if use_largest_component else "")
    frames = []
    for label in labels:
        folder = Path(results_root) / label
        files = sorted(folder.glob(f"*_efficiency{suffix}.csv"))
        if not files:
            print(f"[{label}] WARNING: no '*_efficiency{suffix}.csv' in {folder}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df["dataset"] = label
        frames.append(df)
        print(f"[{label}] {len(df)} graphs from {len(files)} PDB-id files")
    if not frames:
        raise FileNotFoundError(f"No efficiency CSVs found under {results_root}")
    data = pd.concat(frames, ignore_index=True)
    data["mean_degree"] = 2.0 * data["n_edges"] / data["n_nodes"]
    return data, suffix


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _style_for(labels):
    """Fixed colour and marker per dataset so they match across all figures."""
    return {lab: (COLOR_CYCLE[i % len(COLOR_CYCLE)],
                  MARKER_CYCLE[i % len(MARKER_CYCLE)])
            for i, lab in enumerate(labels)}


def _plain_ticks(ax, axis, subs=(1.0, 2.0, 5.0)):
    """Plain-number tick labels (1, 2, 5, 10, ...) on a log axis.
    subs=(1.0,) labels powers of ten only."""
    ax_ = ax.xaxis if axis == "x" else ax.yaxis
    ax_.set_major_locator(LogLocator(base=10, subs=subs))
    ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_.set_minor_formatter(NullFormatter())


def _limits(values, log, pad=0.06):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        lo, hi = np.log10(v.min()), np.log10(v.max())
        d = (hi - lo) * pad or 0.1
        return 10 ** (lo - d), 10 ** (hi + d)
    lo, hi = v.min(), v.max()
    d = (hi - lo) * pad or 0.1
    return lo - d, hi + d


def _save(fig, out_dir, stem, formats, show):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        if fmt.lower() in PLOS_FORMATS:
            save_fig(fig, path, fmt=fmt.lower())
        else:
            fig.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
    if show:
        plt.show()
    plt.close(fig)


def _display(label, display_names):
    return (display_names or {}).get(label, label)


# ----------------------------------------------------------------------------
# Figure set 1: Latora-Marchiori phase plots (one file per dataset)
# ----------------------------------------------------------------------------
def _in_region(df, variant, th):
    if variant == "rand":
        return ((df["Eg_ratio_rand"] >= th["Eg_ratio_rand_min"]) &
                (df["El_ratio_rand"] >= th["El_ratio_rand_min"]))
    return ((df["El_ratio_lat"] >= th["El_ratio_lat_min"]) &
            (df["Eg_ratio_lat"] >= th["Eg_ratio_lat_min"]))


def plot_phase(data, label, variant, th, style, limits, out_dir, formats,
               fig_size, point_size, point_alpha, display_names, show):
    cfg = PHASE_VARIANTS[variant]
    df = data[data["dataset"] == label].dropna(subset=[cfg["x"], cfg["y"]])
    color, marker = style[label]
    xlo, xhi = limits[variant]["x"]
    ylo, yhi = limits[variant]["y"]

    fig, ax = plt.subplots(figsize=FIG_SIZE[fig_size])
    ax.set_xscale("log" if cfg["xlog"] else "linear")
    ax.set_yscale("log" if cfg["ylog"] else "linear")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # small-world region (clipped to the axes)
    if variant == "rand":
        x0, x1 = th["Eg_ratio_rand_min"], xhi
        y0, y1 = th["El_ratio_rand_min"], yhi
    else:
        x0, x1 = th["Eg_ratio_lat_min"], xhi
        y0, y1 = th["El_ratio_lat_min"], yhi
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=SW_GREEN,
                           alpha=0.15, edgecolor="none", zorder=0))
    ax.axvline(1, color=COLORS["mid_gray"], ls="--", lw=0.8, zorder=1)
    ax.axhline(1, color=COLORS["mid_gray"], ls="--", lw=0.8, zorder=1)

    ax.scatter(df[cfg["x"]], df[cfg["y"]], s=point_size, alpha=point_alpha,
               color=color, marker=marker, edgecolor="none", zorder=3)

    frac = _in_region(df, variant, th).mean() if len(df) else np.nan
    ax.text(0.03, 0.97, f"n = {len(df)} graphs\n{frac:.0%} in small-world region",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=FONT["size_annot"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    ax.set_xlabel(RATIO_LABELS[cfg["x"]])
    ax.set_ylabel(RATIO_LABELS[cfg["y"]])
    ax.set_title(f"{_display(label, display_names)} "
                 f"(vs {'random' if variant == 'rand' else 'lattice'} null)",
                 fontsize=FONT["size_title"])
    if cfg["xlog"]:
        _plain_ticks(ax, "x")
    if cfg["ylog"]:
        _plain_ticks(ax, "y")
    fig.tight_layout(pad=LAYOUT["tight_pad"])
    _save(fig, out_dir, f"phase_{variant}_{label}", formats, show)


def phase_limits(data, variants, pad=0.06):
    """Shared axis limits across datasets so panels are directly comparable."""
    lim = {}
    for v in variants:
        cfg = PHASE_VARIANTS[v]
        lim[v] = {"x": _limits(data[cfg["x"]].dropna(), cfg["xlog"], pad),
                  "y": _limits(data[cfg["y"]].dropna(), cfg["ylog"], pad)}
    return lim


# ----------------------------------------------------------------------------
# Figure set 2: size / density dependence (one file per ratio and x-variable)
# ----------------------------------------------------------------------------
def _binned_median(x, y, log, n_bins, min_count):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & ((x > 0) if log else True)
    x, y = x[ok], y[ok]
    if len(x) < min_count:
        return np.array([]), np.array([])
    edges = (np.geomspace(x.min(), x.max(), n_bins + 1) if log
             else np.linspace(x.min(), x.max(), n_bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    cx, cy = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= min_count:
            cx.append(np.exp(np.median(np.log(x[m]))) if log else np.median(x[m]))
            cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def plot_size_dependence(data, ratio, xvar, labels, style, ylog, out_dir,
                         formats, fig_size, point_size, point_alpha, n_bins,
                         min_bin_count, display_names, show):
    xlog = XVAR_LOG[xvar]
    fig, ax = plt.subplots(figsize=FIG_SIZE[fig_size])
    ax.set_xscale("log" if xlog else "linear")
    ax.set_yscale("log" if ylog else "linear")

    handles = []
    for lab in labels:
        df = data[data["dataset"] == lab].dropna(subset=[xvar, ratio])
        if df.empty:
            continue
        color, marker = style[lab]
        ax.scatter(df[xvar], df[ratio], s=point_size, alpha=point_alpha,
                   color=color, marker=marker, edgecolor="none", zorder=2)
        cx, cy = _binned_median(df[xvar], df[ratio], xlog, n_bins, min_bin_count)
        if len(cx):
            ax.plot(cx, cy, color=color, lw=1.5, zorder=4)
        handles.append(Line2D([], [], color=color, marker=marker, ls="-",
                              lw=1.2, markersize=5,
                              label=_display(lab, display_names)))

    ax.axhline(1, color=COLORS["mid_gray"], ls="--", lw=0.8, zorder=1)
    ax.set_xlabel(XVAR_LABELS[xvar])
    ax.set_ylabel(RATIO_LABELS[ratio])
    if xlog:
        _plain_ticks(ax, "x")
    if ylog:
        _plain_ticks(ax, "y")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, title="Dataset (line = binned median)",
              title_fontsize=FONT["size_legend"])
    fig.tight_layout(pad=LAYOUT["tight_pad"])
    _save(fig, out_dir, f"size_{ratio}_vs_{xvar}", formats, show)


def size_dependence_stats(data, labels, ratios, xvars):
    """Spearman correlation of each ratio with each x-variable, per dataset
    and pooled over all datasets."""
    rows = []
    groups = [(lab, data[data["dataset"] == lab]) for lab in labels]
    groups.append(("ALL", data[data["dataset"].isin(labels)]))
    for lab, df in groups:
        for ratio in ratios:
            for xvar in xvars:
                d = df[[xvar, ratio]].dropna()
                if len(d) > 2:
                    rho, p = spearmanr(d[xvar], d[ratio])
                else:
                    rho, p = np.nan, np.nan
                rows.append({"dataset": lab, "ratio": ratio, "x_variable": xvar,
                             "n_graphs": len(d), "spearman_rho": rho,
                             "p_value": p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main(results_root, output_dir, labels,
         random_null="degree_preserving", use_largest_component=False,
         make_phase=True, phase_variants=("rand", "lat"),
         thresholds=None,
         make_size=True,
         size_ratios=("Eg_ratio_rand", "El_ratio_rand",
                      "Eg_ratio_lat", "El_ratio_lat"),
         size_xvars=("n_nodes",),
         size_ylog=("El_ratio_rand", "Eg_ratio_lat"),
         n_bins=8, min_bin_count=5,
         phase_fig_size="half_page", size_fig_size="full_width",
         point_size=10, point_alpha=0.45,
         display_names=None, formats=("png", "tiff"), show=True):
    """
    results_root          : output_dir of small_world_efficiency (holds <label>/ folders)
    output_dir            : where figures and combined CSVs are written
    labels                : dataset labels (the sub-folder names), in plot order
    random_null           : must match the efficiency run ('degree_preserving' or 'gnm')
    use_largest_component : must match the efficiency run (reads the *_lcc.csv files)
    make_phase            : draw the phase plots (one file per dataset per variant)
    phase_variants        : 'rand' and/or 'lat'
    thresholds            : small-world region; see DEFAULT_THRESHOLDS below
    make_size             : draw the size-dependence plots
    size_ratios           : ratio columns to put on y
    size_xvars            : any of 'n_nodes', 'density', 'mean_degree'
    size_ylog             : ratio columns drawn with a log y-axis
    n_bins, min_bin_count : binned-median trend lines (bins are log-spaced when x is log)
    phase_fig_size, size_fig_size : keys of figure_style.FIG_SIZE
    point_size, point_alpha : scatter appearance
    display_names         : optional {label: pretty name} for titles/legends
    formats               : e.g. ('png', 'tiff'); tiff/eps use PLOS checks
    show                  : True = also display each figure (notebooks)
    """
    th = dict(DEFAULT_THRESHOLDS)
    th.update(thresholds or {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    if not show:
        plt.switch_backend("Agg")

    data, suffix = load_results(results_root, labels, random_null,
                                use_largest_component)
    labels = [lab for lab in labels if (data["dataset"] == lab).any()]
    style = _style_for(labels)
    data.to_csv(out_dir / f"all_graphs_efficiency{suffix}.csv", index=False)

    if make_phase:
        limits = phase_limits(data, phase_variants)
        for variant in phase_variants:
            for lab in labels:
                plot_phase(data, lab, variant, th, style, limits,
                           out_dir / "phase", formats, phase_fig_size,
                           point_size, point_alpha, display_names, show)

    if make_size:
        for xvar in size_xvars:
            for ratio in size_ratios:
                plot_size_dependence(
                    data, ratio, xvar, labels, style, ratio in size_ylog,
                    out_dir / "size_dependence", formats, size_fig_size,
                    point_size, point_alpha, n_bins, min_bin_count,
                    display_names, show)
        stats = size_dependence_stats(data, labels, size_ratios, size_xvars)
        stats.to_csv(out_dir / f"size_dependence_stats{suffix}.csv", index=False)
        print("\nSpearman correlations with size / density "
              "(|rho| near 0 = no dependence):")
        print(stats.round(3).to_string(index=False))

    print(f"\nSaved figures and tables to: {out_dir}")


DEFAULT_THRESHOLDS = {
    # 'rand' variant: small-world if paths are random-like AND local efficiency
    # clearly exceeds the random graph's
    "Eg_ratio_rand_min": 0.9,
    "El_ratio_rand_min": 2.0,
    # 'lat' variant: small-world if local efficiency is lattice-like AND paths
    # are at least as short as the lattice's
    "El_ratio_lat_min": 0.8,
    "Eg_ratio_lat_min": 1.0,
}


def mount_drive_if_colab(mount_point="/content/drive"):
    """Mount Google Drive when running inside Colab; do nothing elsewhere."""
    try:
        from google.colab import drive
    except ImportError:
        return False
    if not os.path.ismount(mount_point):
        drive.mount(mount_point)
    return True


if __name__ == "__main__":
    # ---- Edit these values ----
    drive_root = "...0_4_revisions/2_experiments/"

    results_root = f"...0_4_revisions/2_experiments/3_small_world"   # efficiency output
    output_dir = f"...0_4_revisions/2_experiments/3_small_world/figures_ver2"

    labels = ["arp_iac", "arp_no_iac", "ring_iac", "ring_no_iac",
              "pcn5", "pcn6", "pcn7"]
    display_names = {                       # optional pretty names; {} to skip
        "arp_iac": "Arpeggio, IAC", "arp_no_iac": "Arpeggio, no IAC",
        "ring_iac": "RING, IAC", "ring_no_iac": "RING, no IAC",
        "pcn5": "PCN 5 Å", "pcn6": "PCN 6 Å", "pcn7": "PCN 7 Å",
    }

    # must match the settings used for the efficiency run
    random_null = "degree_preserving"
    use_largest_component = False

    # Figure set 1: phase plots (one file per dataset per variant)
    make_phase = True
    phase_variants = ("rand", "lat")        # "rand", "lat", or both
    thresholds = {                          # small-world region (conventions!)
        "Eg_ratio_rand_min": 0.9,
        "El_ratio_rand_min": 2.0,
        "El_ratio_lat_min": 0.8,
        "Eg_ratio_lat_min": 1.0,
    }

    # Figure set 2: size / density dependence
    make_size = True
    size_ratios = ("Eg_ratio_rand", "El_ratio_rand",
                   "Eg_ratio_lat", "El_ratio_lat")
    size_xvars = ("n_nodes",)               # add "density", "mean_degree" if wanted
    size_ylog = ("El_ratio_rand", "Eg_ratio_lat")
    n_bins = 8
    min_bin_count = 5

    # Appearance / output
    phase_fig_size = "half_page"            # keys of figure_style.FIG_SIZE
    size_fig_size = "full_width"
    point_size = 10
    point_alpha = 0.45
    formats = ("png", "tiff")               # PLOS needs tiff/eps
    show = True                             # display figures in the notebook
    mount_drive = False
    # ---------------------------

    if mount_drive:
        mount_drive_if_colab()

    main(
        results_root=results_root, output_dir=output_dir, labels=labels,
        random_null=random_null, use_largest_component=use_largest_component,
        make_phase=make_phase, phase_variants=phase_variants,
        thresholds=thresholds, make_size=make_size, size_ratios=size_ratios,
        size_xvars=size_xvars, size_ylog=size_ylog, n_bins=n_bins,
        min_bin_count=min_bin_count, phase_fig_size=phase_fig_size,
        size_fig_size=size_fig_size, point_size=point_size,
        point_alpha=point_alpha, display_names=display_names,
        formats=formats, show=show,
    )

"""
plot_graph_metric_distributions.py
==================================
Compare the distribution of a per-protein network metric (e.g. characteristic
path length, average clustering) between two sets of graphs built from the
same proteins.

Each figure shows, for both graph sets (choose with plot_kind):
  - "scatter": ONE figure, clustering (x) vs path length (y), one point per
               protein, mean +/- error in both directions, optional marginal
               histograms. (The kinds below make one figure PER metric.)
  - "violin" : smoothed density + jittered points
  - "bar"    : bar of the mean + jittered points
  - "box"    : box plot (median/quartiles) + jittered points
  - "hist"   : overlaid histograms with mean markers along the top
  - plus the mean with an error bar (SD, SEM, or bootstrap 95% CI -- your choice)
Optionally a paired Wilcoxon signed-rank p-value is printed on the plot.

A summary table (n, mean, standard deviation, median, min, max for every metric
and graph set) is printed to the terminal before plotting (print_stats=True).
The exact paired Wilcoxon signed-rank results (W statistic, exact p-value, number
of pairs, median paired difference) are printed as well (print_tests=True).

All settings are arguments of main(); edit them there. No command-line
arguments are used.
"""

import sys
from scipy import stats
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *
from figure_style_ver2 import (apply_style, save_fig, COLORS, FONT, FIG_SIZE, LAYOUT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _normalize_ids(index, id_regex=None):
    """Strip whitespace, lowercase, and optionally delete a regex match (e.g. a
    tool-specific suffix or file extension) so IDs from two tools can match."""
    ids = pd.Index(index.astype(str)).str.strip().str.lower()
    if id_regex:
        ids = ids.str.replace(id_regex, "", regex=True)
    return ids


def load_paired(csv_a, csv_b, id_col="graph", id_regex_a=None, id_regex_b=None):
    """
    Read both CSVs and keep only proteins present in both (inner join on id).

    If the two tools name their graphs differently (e.g. '1abc_arpeggio' vs
    '1abc.json'), pass a regex per file that deletes the differing part.
    Raises an error (with example IDs) if nothing matches.
    """
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)
    raw_a, raw_b = a[id_col].astype(str), b[id_col].astype(str)
    a[id_col] = _normalize_ids(a[id_col], id_regex_a)
    b[id_col] = _normalize_ids(b[id_col], id_regex_b)
    for name, df in (("A", a), ("B", b)):
        if not df[id_col].is_unique:
            dup = df[id_col][df[id_col].duplicated()].iloc[0]
            raise ValueError(f"Set {name}: IDs are not unique after cleaning "
                             f"(e.g. '{dup}'). Adjust id_regex_{name.lower()}.")
    a, b = a.set_index(id_col), b.set_index(id_col)
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        raise ValueError(
            "No graph IDs match between the two CSVs.\n"
            f"  A examples: {list(raw_a[:3])}\n"
            f"  B examples: {list(raw_b[:3])}\n"
            "Set id_regex_a / id_regex_b in main() to delete the differing "
            "parts (suffix, prefix, extension).")
    if len(shared) != len(a) or len(shared) != len(b):
        print(f"[warn] only {len(shared)} graphs shared "
              f"(A has {len(a)}, B has {len(b)}).")
    return a.loc[shared], b.loc[shared]


# ---------------------------------------------------------------------------
# Terminal summary statistics
# ---------------------------------------------------------------------------
def print_summary_stats(a, b, summary_metrics, labels, decimals=4):
    """
    Print n, mean, standard deviation (sample SD, ddof=1), median, min and max
    for each metric and each graph set.

    summary_metrics : sequence of (csv column, display name)
    labels          : (name of set A, name of set B)
    """
    rows = []
    for col, name in summary_metrics:
        for df, lab in zip((a, b), labels):
            v = df[col].to_numpy(float)
            v = v[~np.isnan(v)]
            rows.append({
                "metric": name,
                "set": lab,
                "n": len(v),
                "mean": v.mean(),
                "SD": v.std(ddof=1),
                "median": np.median(v),
                "min": v.min(),
                "max": v.max(),
            })
    table = pd.DataFrame(rows)
    print("\nSummary statistics (SD = sample standard deviation, ddof=1)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.{decimals}f}"))
    print()


def print_wilcoxon_tests(a, b, summary_metrics, labels, decimals=4):
    """
    Print the paired Wilcoxon signed-rank test (set A vs set B, same proteins)
    for each metric, with the exact p-value rather than the 'p < 0.001' shown
    on the figures.

    Columns: n_pairs (pairs used), W (test statistic), p_value, median_diff
    (median of A - B), n_zero_diff (pairs with identical values; scipy drops
    these from the test by default).
    """
    rows = []
    for col, name in summary_metrics:
        va = a[col].to_numpy(float)
        vb = b[col].to_numpy(float)
        ok = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[ok], vb[ok]
        diff = va - vb
        res = stats.wilcoxon(va, vb)
        rows.append({
            "metric": name,
            "n_pairs": len(diff),
            "W": res.statistic,
            "p_value": res.pvalue,
            f"median_diff ({labels[0]} - {labels[1]})": np.median(diff),
            "n_zero_diff": int((diff == 0).sum()),
        })
    table = pd.DataFrame(rows)
    print("Paired Wilcoxon signed-rank tests (two-sided)")
    print(table.to_string(
        index=False,
        formatters={"p_value": lambda x: f"{x:.3e}",
                    "W": lambda x: f"{x:.1f}"},
        float_format=lambda x: f"{x:.{decimals}f}"))
    print()


# ---------------------------------------------------------------------------
# Error bar computation
# ---------------------------------------------------------------------------
def mean_and_error(values, error_type="ci95", n_boot=10000, seed=0):
    """
    Return (mean, lower_err, upper_err) as distances from the mean.

    error_type:
      "sd"    standard deviation  -> spread of the proteins
      "sem"   standard error      -> uncertainty of the mean
      "ci95"  bootstrap 95% CI of the mean (percentile method)
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    m = v.mean()
    if error_type == "sd":
        e = v.std(ddof=1)
        return m, e, e
    if error_type == "sem":
        e = v.std(ddof=1) / np.sqrt(len(v))
        return m, e, e
    if error_type == "ci95":
        rng = np.random.default_rng(seed)
        boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return m, m - lo, hi - m
    raise ValueError("error_type must be 'sd', 'sem', or 'ci95'")


# ---------------------------------------------------------------------------
# Drawing helpers, one per plot kind
# ---------------------------------------------------------------------------
def _clean(v):
    v = np.asarray(v, dtype=float)
    return v[~np.isnan(v)]


def _draw_categorical(ax, vals_list, labels, colors, kind, error_type, rng, seed):
    """violin / bar / box: one slot per graph set on the x axis."""
    for i, (vals, col) in enumerate(zip(vals_list, colors)):
        v = _clean(vals)
        m, lo, hi = mean_and_error(v, error_type, seed=seed)

        if kind == "violin":
            vp = ax.violinplot(v, positions=[i], widths=0.8, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_edgecolor(col)
                body.set_alpha(0.35)
        elif kind == "box":
            bp = ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True,
                            showfliers=False, medianprops={"color": COLORS["dark_gray"]})
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_alpha(0.35)
                patch.set_edgecolor(col)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(col)
        elif kind == "bar":
            ax.bar(i, m, width=0.6, color=col, alpha=0.55, zorder=1)

        # individual proteins
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=col,
                   alpha=0.45, linewidths=0, zorder=2)

        # mean +/- error. On a bar the bar top already marks the mean.
        ax.errorbar(i, m, yerr=[[lo], [hi]],
                    fmt="none" if kind == "bar" else "o",
                    color=COLORS["dark_gray"], markerfacecolor="white",
                    markersize=LAYOUT["marker_size"],
                    capsize=LAYOUT["cap_size"] + 1, elinewidth=1.5,
                    capthick=1.5, zorder=4)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x", visible=False)
    if kind == "bar":
        ax.set_ylim(bottom=0)


def _draw_hist(ax, vals_list, labels, colors, error_type, bins, seed):
    """Overlaid histograms (shared bins) with mean +/- error bars along the top."""
    clean = [_clean(v) for v in vals_list]
    edges = np.histogram_bin_edges(np.concatenate(clean), bins=bins)
    for v, lab, col in zip(clean, labels, colors):
        ax.hist(v, bins=edges, color=col, alpha=0.5, edgecolor="white",
                linewidth=0.4, label=lab)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.25)
    for i, (v, col) in enumerate(zip(clean, colors)):
        m, lo, hi = mean_and_error(v, error_type, seed=seed)
        ax.errorbar(m, ymax * (1.20 - 0.07 * i), xerr=[[lo], [hi]], fmt="o",
                    color=col, markerfacecolor="white",
                    markersize=LAYOUT["marker_size"],
                    capsize=LAYOUT["cap_size"] + 1, elinewidth=1.5,
                    capthick=1.5, zorder=4)
    ax.set_ylabel("Number of proteins")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), frameon=False)


# ---------------------------------------------------------------------------
# One figure for one metric
# ---------------------------------------------------------------------------
def plot_metric(values_a, values_b, ylabel, labels, colors, plot_kind,
                error_type, show_pvalue, out_path, size_key="single_col",
                bins=20, seed=0):
    if plot_kind not in ("violin", "bar", "box", "hist"):
        raise ValueError("plot_kind must be 'violin', 'bar', 'box', or 'hist'")
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE[size_key])
    rng = np.random.default_rng(seed)
    vals_list = [values_a, values_b]

    if plot_kind == "hist":
        _draw_hist(ax, vals_list, labels, colors, error_type, bins, seed)
        ax.set_xlabel(ylabel)          # metric runs along x for histograms
    else:
        _draw_categorical(ax, vals_list, labels, colors, plot_kind,
                          error_type, rng, seed)
        ax.set_ylabel(ylabel)

    # Header line: error-bar definition, n, and (optionally) paired test
    err_name = {"sd": "SD", "sem": "SEM", "ci95": "bootstrap 95% CI"}[error_type]
    header = f"Mean \u00b1 {err_name}, n = {len(_clean(values_a))} proteins"
    if show_pvalue:
        a = np.asarray(values_a, float)
        b = np.asarray(values_b, float)
        ok = ~(np.isnan(a) | np.isnan(b))
        p = stats.wilcoxon(a[ok], b[ok]).pvalue
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        header += f"\nWilcoxon signed-rank (paired), {ptxt}"
    ax.text(0.5, 1.02, header, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=FONT["size_annot"])

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Joint figure: path length (y) vs clustering (x), both graph sets in one panel
# ---------------------------------------------------------------------------
def _paired_p(va, vb):
    va = np.asarray(va, float)
    vb = np.asarray(vb, float)
    ok = ~(np.isnan(va) | np.isnan(vb))
    p = stats.wilcoxon(va[ok], vb[ok]).pvalue
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

def plot_scatter(a, b, x_col, y_col, xlabel, ylabel, labels, colors,
                 error_type, show_pvalue, out_path, size_key="single_sq",
                 marginals=True, bins=20, seed=0, errorbar_color="black"):
    """
    One point per protein (x = x_col, y = y_col) for each graph set, plus the
    set mean with x AND y error bars. If marginals=True, the distribution of
    each metric is drawn as a histogram on the top (x) and right (y) edges.
    """
    apply_style()
    fig = plt.figure(figsize=FIG_SIZE[size_key])
    if marginals:
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              hspace=0.05, wspace=0.05)
        ax = fig.add_subplot(gs[1, 0])
        axx = fig.add_subplot(gs[0, 0], sharex=ax)
        axy = fig.add_subplot(gs[1, 1], sharey=ax)
        axleg = fig.add_subplot(gs[0, 1])
        axleg.axis("off")
    else:
        ax = fig.add_subplot(111)

    data = []
    for df in (a, b):
        x = df[x_col].to_numpy(float)
        y = df[y_col].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        data.append((x[ok], y[ok]))

    xedges = np.histogram_bin_edges(np.concatenate([d[0] for d in data]), bins=bins)
    yedges = np.histogram_bin_edges(np.concatenate([d[1] for d in data]), bins=bins)

    handles = []
    for (x, y), lab, col in zip(data, labels, colors):
        ax.scatter(x, y, s=6, color=col, alpha=0.4, linewidths=0, zorder=2)
        mx, xlo, xhi = mean_and_error(x, error_type, seed=seed)
        my, ylo, yhi = mean_and_error(y, error_type, seed=seed)
        ax.errorbar(mx, my, xerr=[[xlo], [xhi]], yerr=[[ylo], [yhi]], fmt="o",
                    color=col, ecolor=errorbar_color,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=1.5, markersize=LAYOUT["marker_size"] + 1,
                    capsize=LAYOUT["cap_size"] + 1, elinewidth=1.5,
                    capthick=1.5, zorder=4)
        handles.append(ax.scatter([], [], s=14, color=col, label=lab))
        if marginals:
            axx.hist(x, bins=xedges, color=col, alpha=0.5, edgecolor="white",
                     linewidth=0.4)
            axy.hist(y, bins=yedges, color=col, alpha=0.5, edgecolor="white",
                     linewidth=0.4, orientation="horizontal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if marginals:
        for m_ax in (axx, axy):
            m_ax.grid(False)
            m_ax.spines["left" if m_ax is axx else "bottom"].set_visible(False)
        axx.set_yticks([])
        axy.set_xticks([])
        axx.tick_params(labelbottom=False)
        axy.tick_params(labelleft=False)
        axleg.legend(handles=handles, loc="center", frameon=False)
    else:
        ax.legend(handles=handles, loc="best", frameon=False)

    err_name = {"sd": "SD", "sem": "SEM", "ci95": "bootstrap 95% CI"}[error_type]
    header = f"Mean \u00b1 {err_name}, n = {len(data[0][0])} proteins"
    # if show_pvalue:
    #     header += (f"\nWilcoxon (paired): x {_paired_p(a[x_col], b[x_col])}, "
    #                f"y {_paired_p(a[y_col], b[y_col])}")
    title_ax = axx if marginals else ax
    title_ax.set_title(header, fontsize=FONT["size_annot"], fontweight="normal")

    if not marginals:
        fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main -- change everything here
# ---------------------------------------------------------------------------
def main(
    csv_a="....0_4_revisions/network_metrics/Arpeggio/network_metrics.csv",
    csv_b="....0_4_revisions/network_metrics/RING/network_metrics.csv",
    labels=("Arpeggio", "RING"),
    colors=(COLORS["blue"], COLORS["orange"]),
    id_col="graph",
    id_regex_a=r"_clean$",          # regex deleted from set A IDs (Arpeggio: 1AC0_clean -> 1ac0)
    id_regex_b=None,                # regex deleted from set B's IDs, e.g. r"\.json$"
    metrics=(                       # (csv column, y-axis label, output file)
        ("char_path_length", "Characteristic path length", "fig_path_length.png"),
        ("avg_clustering",   "Average clustering coefficient", "fig_clustering.png"),
    ),
    plot_kind="scatter",            # "scatter" | "violin" | "bar" | "box" | "hist"
    # --- scatter only: x = clustering, y = path length, one combined figure ---
    x_metric=("avg_clustering", "Mean clustering coefficient"),
    y_metric=("char_path_length", "Characteristic path length"),
    scatter_out="fig_path_vs_clustering.png",
    marginals=True,                 # histograms of each metric on the top/right edges
    scatter_size_key="single_sq",   # key from FIG_SIZE

    error_type="sd",              # "sd" | "sem" | "ci95"
    bins=20,                        # histogram bins (plot_kind="hist" only)
    show_pvalue=True,
    size_key="single_col",          # key from FIG_SIZE
    print_stats=True,               # print the mean/SD table to the terminal
    print_tests=True,               # print exact paired Wilcoxon results to the terminal
    stats_decimals=4,               # decimal places in that table
    summary_metrics=(               # (csv column, name shown in the table)
        ("char_path_length", "Characteristic path length"),
        ("avg_clustering",   "Average clustering coefficient"),
    ),
):
    a, b = load_paired(csv_a, csv_b, id_col=id_col,
                       id_regex_a=id_regex_a, id_regex_b=id_regex_b)
    if print_stats:
        print_summary_stats(a, b, summary_metrics, labels, decimals=stats_decimals)
    if print_tests:
        print_wilcoxon_tests(a, b, summary_metrics, labels, decimals=stats_decimals)
    if plot_kind == "scatter":
        plot_scatter(a, b, x_metric[0], y_metric[0], x_metric[1], y_metric[1],
                     labels, colors, error_type, show_pvalue, scatter_out,
                     size_key=scatter_size_key, marginals=marginals, bins=bins)
        return
    for col, ylabel, out in metrics:
        plot_metric(a[col], b[col], ylabel, labels, colors, plot_kind,
                    error_type, show_pvalue, out, size_key=size_key, bins=bins)


if __name__ == "__main__":
    main()

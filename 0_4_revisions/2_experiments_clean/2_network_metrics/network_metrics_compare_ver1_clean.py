"""
Compare global network metrics from two CSV files (two sets of graphs).

Each CSV is a metrics table such as the `network_metrics.csv` written by
network_metrics_ver5.py. The script makes the same figures as that script,
but each figure shows BOTH datasets on the same axes (different colour and
marker per dataset, each with its own trend line and bootstrap fits), so the
two sets of graphs can be compared directly.

Metrics plotted against number of nodes (one point per graph):
    - Mean degree
    - Mean clustering coefficient
    - Characteristic path length
    - Graph density
    - Diameter

Required CSV columns:
    nodes, mean_degree, avg_clustering, char_path_length, density, diameter

Outputs (each figure is its own file):
    <output_dir>/mean_degree_vs_nodes.png
    <output_dir>/clustering_vs_nodes.png
    <output_dir>/path_length_vs_nodes.png
    <output_dir>/density_vs_nodes.png
    <output_dir>/diameter_vs_nodes.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imports import *
import matplotlib
matplotlib.use("Agg")  # no display needed; figures are written to disk
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_metrics_csv(path, required_columns):
    """Read a metrics CSV into a dict: column name -> numpy float array.
    Non-numeric or empty cells become NaN (they are dropped when plotting)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
        rows = list(reader)

    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    return {c: np.array([to_float(r[c]) for r in rows]) for c in required_columns}


# ----------------------------------------------------------------------------
# Plotting (one figure per file, two datasets per figure)
# ----------------------------------------------------------------------------
def plain_ticks(axis, is_log):
    """Label ticks as plain numbers (10, 0.05), never scientific notation."""
    if is_log:
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(fmt)
    else:
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        axis.set_major_formatter(fmt)


def draw_series(ax, x, y, color, marker, light_color, log_x, log_y, power_law,
                fit_trend, n_boot, seed):
    """Draw one dataset: scatter points, a least-squares trend line and
    bootstrap fits. The fit is straight in log10 on log axes (or when
    power_law=True, which fits log-log but draws on the axes as they are)."""
    fit_log_x = log_x or power_law
    fit_log_y = log_y or power_law

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if fit_log_x:
        ok &= x > 0
    if fit_log_y:
        ok &= y > 0
    x, y = x[ok], y[ok]

    ax.scatter(x, y, s=30, color=color, marker=marker, edgecolor="black",
               linewidth=0.5, alpha=0.8, zorder=3)

    fx = np.log10(x) if fit_log_x else x
    fy = np.log10(y) if fit_log_y else y
    if fit_trend and len(x) >= 3 and np.ptp(fx) > 0:
        xs = np.linspace(fx.min(), fx.max(), 100)

        def to_data(v, is_log):
            return 10 ** v if is_log else v

        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, len(fx), len(fx))
            if np.ptp(fx[idx]) == 0:
                continue
            a, b = np.polyfit(fx[idx], fy[idx], 1)
            ax.plot(to_data(xs, fit_log_x), to_data(a * xs + b, fit_log_y),
                    color=light_color, alpha=0.08, linewidth=1, zorder=1)
        a, b = np.polyfit(fx, fy, 1)
        ax.plot(to_data(xs, fit_log_x), to_data(a * xs + b, fit_log_y),
                color=color, linewidth=2, zorder=2)


def plot_metric_comparison(
    datasets, key, ylabel, out_path, log_x, log_y, power_law, fit_trend,
    n_boot, seed, dpi, styles, figsize,
):
    """One figure: metric `key` vs. nodes for every dataset in `datasets`
    (list of (label, data_dict))."""
    fig, ax = plt.subplots(figsize=figsize)
    handles = []
    for i, (label, data) in enumerate(datasets):
        color, light_color, marker = styles[i % len(styles)]
        # Offset the seed per dataset so bootstrap draws are independent
        draw_series(ax, data["nodes"], data[key], color, marker, light_color,
                    log_x, log_y, power_law, fit_trend, n_boot, seed + i)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle="-",
                              markeredgecolor="black", markeredgewidth=0.5,
                              markersize=6, linewidth=2, label=label))

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    plain_ticks(ax.xaxis, log_x)
    plain_ticks(ax.yaxis, log_y)
    ax.set_xlabel("Number of nodes, n")
    ax.set_ylabel(ylabel)
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(
    csv_path_1,
    csv_path_2,
    label_1,
    label_2,
    output_dir,
    log_x,
    log_y,
    power_law_fit,
    fit_trend,
    n_boot,
    seed,
    dpi,
    styles,
    figsize,
    lcc_note,
):
    """log_x, log_y: dicts mapping metric name -> True/False for a log-scale axis.
    power_law_fit: metric names whose trend is a power-law (log-log) fit.
    styles: list of (line colour, light bootstrap colour, marker), one per dataset."""
    os.makedirs(output_dir, exist_ok=True)

    specs = [
        ("mean_degree", "Mean degree", "mean_degree_vs_nodes.png"),
        ("avg_clustering", "Mean clustering coefficient", "clustering_vs_nodes.png"),
        ("char_path_length", "Characteristic path length" + lcc_note,
         "path_length_vs_nodes.png"),
        ("density", "Graph density", "density_vs_nodes.png"),
        ("diameter", "Diameter" + lcc_note, "diameter_vs_nodes.png"),
    ]
    required = ["nodes"] + [k for k, _, _ in specs]

    datasets = [
        (label_1, load_metrics_csv(csv_path_1, required)),
        (label_2, load_metrics_csv(csv_path_2, required)),
    ]
    for label, data in datasets:
        print(f"{label}: {len(data['nodes'])} graphs")

    for key, ylabel, out_name in specs:
        plot_metric_comparison(
            datasets, key, ylabel, os.path.join(output_dir, out_name),
            log_x.get(key, False), log_y.get(key, False), key in power_law_fit,
            fit_trend, n_boot, seed, dpi, styles, figsize,
        )

    print(f"\nSaved figures to: {output_dir}")


if __name__ == "__main__":
    # ---- Edit these values ----
    csv_path_1 = "...network_metrics.csv"   # first metrics CSV
    csv_path_2 = "...network_metrics.csv"   # second metrics CSV
    label_1 = "Arpeggio"             # legend label for the first CSV
    label_2 = "RING"             # legend label for the second CSV
    output_dir = "...0_4_revisions/network_metrics/comparison"      # where figures are written
    # Log-scale axes, set per metric (True = log, False = linear).
    log_x = {
        "mean_degree": False,
        "avg_clustering": False,
        "char_path_length": False,
        "density": False,
        "diameter": False,
    }
    log_y = {
        "mean_degree": False,
        "avg_clustering": False,
        "char_path_length": False,
        "density": False,
        "diameter": False,
    }
    # Metrics whose trend line is a power-law fit (straight in log-log space)
    # drawn as a curve on the axes above. Density falls roughly as 1/n.
    power_law_fit = ["density"]
    fit_trend = True                  # trend line + bootstrap fits for each dataset
    n_boot = 100                      # number of bootstrap fits drawn per dataset
    seed = 0                          # makes the bootstrap reproducible
    dpi = 300
    figsize = (5.5, 4)                # inches, per figure
    # (trend line colour, bootstrap colour, marker) for dataset 1, then dataset 2
    styles = [
        ("#4878CF", "#9DB5E6", "o"),  # blue
        ("#D65F00", "#EBA878", "s"),  # orange
    ]
    lcc_note = " (largest component)"  # appended to path length / diameter labels; "" to omit
    # ---------------------------

    main(
        csv_path_1=csv_path_1,
        csv_path_2=csv_path_2,
        label_1=label_1,
        label_2=label_2,
        output_dir=output_dir,
        log_x=log_x,
        log_y=log_y,
        power_law_fit=power_law_fit,
        fit_trend=fit_trend,
        n_boot=n_boot,
        seed=seed,
        dpi=dpi,
        styles=styles,
        figsize=figsize,
        lcc_note=lcc_note,
    )

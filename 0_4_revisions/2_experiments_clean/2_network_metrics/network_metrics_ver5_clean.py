"""
Compute global network metrics for every GraphML graph in a folder and plot
each metric against network size (number of nodes), one point per graph.

Metrics (computed on a simple, undirected, unweighted version of each graph):
    - Average clustering coefficient
    - Characteristic path length (average shortest path length)
    - Graph density
    - Diameter
    - Number of connected components
    - Mean degree (the summary of the degree distribution, one value per graph)
    - Optionally, a full degree-distribution plot per graph

Path length and diameter are only defined for connected graphs, so by default
they are computed on the largest connected component (LCC). For very large
graphs exact computation is slow; above `exact_max_nodes` the script estimates
path length from a random sample of source nodes (and the diameter becomes a
lower bound). Estimated rows are flagged in the CSV.

Outputs (each figure is its own file):
    <output_dir>/network_metrics.csv
    <output_dir>/clustering_vs_nodes.png
    <output_dir>/path_length_vs_nodes.png
    <output_dir>/density_vs_nodes.png
    <output_dir>/diameter_vs_nodes.png
    <output_dir>/components_vs_nodes.png
    <output_dir>/mean_degree_vs_nodes.png
    <output_dir>/degree_distribution_<graph_name>.png   (optional, one per graph)
"""

import csv
import os
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are written to disk
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_graph(path):
    """Load a GraphML file as a simple, undirected graph without self-loops."""
    G = nx.read_graphml(path)
    H = nx.Graph(G.to_undirected()) if G.is_directed() else nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def estimate_paths(H, n_samples, seed):
    """Estimate mean shortest path length and diameter (lower bound) via BFS
    from a random sample of source nodes. H must be connected."""
    rng = random.Random(seed)
    nodes = list(H.nodes())
    sources = rng.sample(nodes, min(n_samples, len(nodes)))
    total, count, max_ecc = 0, 0, 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(H, s)
        total += sum(lengths.values())
        count += len(lengths) - 1
        max_ecc = max(max_ecc, max(lengths.values()))
    return total / count, max_ecc


def compute_metrics(G, use_largest_component, exact_max_nodes, n_samples, seed):
    """Return a dict of global metrics for an undirected simple graph."""
    n = G.number_of_nodes()
    components = list(nx.connected_components(G))
    n_components = len(components)

    if use_largest_component and n_components > 0:
        H = G.subgraph(max(components, key=len))
    else:
        H = G

    estimated = False
    path_length = float("nan")
    diameter = float("nan")
    if H.number_of_nodes() > 1 and nx.is_connected(H):
        if H.number_of_nodes() <= exact_max_nodes:
            path_length = nx.average_shortest_path_length(H)
            diameter = nx.diameter(H)
        else:
            path_length, diameter = estimate_paths(H, n_samples, seed)
            estimated = True

    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "mean_degree": 2 * G.number_of_edges() / n if n else float("nan"),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "char_path_length": path_length,
        "density": nx.density(G),
        "diameter": diameter,
        "n_components": n_components,
        "lcc_size": H.number_of_nodes(),
        "lcc_fraction": H.number_of_nodes() / n if n else float("nan"),
        "paths_estimated": estimated,
    }


def degree_distribution(G):
    """Return (degrees, probabilities) with P(k) = fraction of nodes with degree k."""
    counts = Counter(d for _, d in G.degree())
    n = G.number_of_nodes()
    ks = sorted(counts)
    return ks, [counts[k] / n for k in ks]


# ----------------------------------------------------------------------------
# Plotting (one figure per file)
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


def plot_metric_vs_nodes(
    nodes, values, ylabel, out_path, log_x, log_y, fit_trend, n_boot, seed, dpi,
    power_law=False,
):
    """Scatter of metric vs. nodes (one point per graph) with an optional trend.

    The red line is a least-squares fit, done on whichever scale each axis is
    drawn: straight in log10 on a log axis, straight in the raw values on a
    linear axis. Thin pink lines are fits to bootstrap resamples and show the
    uncertainty of the trend.

    power_law=True fits a straight line in log-log space (a power law,
    y = c * x^slope) but draws the resulting curve on the axes as they are, so
    a metric can keep the same linear x-axis as the other figures.
    """
    fit_log_x = log_x or power_law  # scale used for fitting, not for drawing
    fit_log_y = log_y or power_law

    x = np.asarray(nodes, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if fit_log_x:
        ok &= x > 0
    if fit_log_y:
        ok &= y > 0  # zeros cannot be shown or fitted on a log scale
    x, y = x[ok], y[ok]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(x, y, s=30, color="steelblue", edgecolor="black", linewidth=0.5,
               alpha=0.8, zorder=3)

    fx = np.log10(x) if fit_log_x else x  # fit space for x
    fy = np.log10(y) if fit_log_y else y  # fit space for y
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
                    color="lightcoral", alpha=0.08, linewidth=1, zorder=1)
        a, b = np.polyfit(fx, fy, 1)
        ax.plot(to_data(xs, fit_log_x), to_data(a * xs + b, fit_log_y),
                color="red", linewidth=2, zorder=2)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    plain_ticks(ax.xaxis, log_x)
    plain_ticks(ax.yaxis, log_y)
    ax.set_xlabel("Number of nodes, n")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_degree_distribution(G, name, out_path, log_log, dpi):
    ks, pk = degree_distribution(G)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(ks, pk, s=18, color="steelblue", edgecolor="black", linewidth=0.4)
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("Degree, k")
    ax.set_ylabel("P(k)")
    ax.set_title(f"Degree distribution: {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(
    graphs_dir,
    output_dir,
    use_largest_component,
    exact_max_nodes,
    n_samples,
    seed,
    log_x,
    log_y,
    skip_metrics,
    power_law_fit,
    fit_trend,
    n_boot,
    log_log_degree,
    per_graph_degree_plots,
    dpi,
):
    """log_x, log_y: dicts mapping metric name -> True/False for a log-scale axis.
    skip_metrics: metric names for which no figure is written (still in the CSV)."""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(graphs_dir) if f.lower().endswith(".graphml"))
    if not files:
        print(f"No .graphml files found in {graphs_dir}")
        return

    rows = []
    for fname in files:
        name = os.path.splitext(fname)[0]
        try:
            G = load_graph(os.path.join(graphs_dir, fname))
        except Exception as e:  # keep going if one file is bad
            print(f"Skipping {fname}: {e}")
            continue

        metrics = compute_metrics(
            G, use_largest_component, exact_max_nodes, n_samples, seed
        )
        metrics["graph"] = name
        rows.append(metrics)
        print(f"{name}: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items() if k != "graph"
        ))

        if per_graph_degree_plots:
            plot_degree_distribution(
                G, name,
                os.path.join(output_dir, f"degree_distribution_{name}.png"),
                log_log_degree, dpi,
            )

    if not rows:
        return

    # Table of results
    fieldnames = ["graph"] + [k for k in rows[0] if k != "graph"]
    with open(os.path.join(output_dir, "network_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # One figure per metric: x = number of nodes, y = metric, one point per graph
    nodes = [r["nodes"] for r in rows]
    lcc_note = " (largest component)" if use_largest_component else ""
    specs = [
        ("mean_degree", "Mean degree, k", "mean_degree_vs_nodes.png"),
        ("avg_clustering", "Mean clustering coefficient", "clustering_vs_nodes.png"),
        ("char_path_length", "Characteristic path length" + lcc_note,
         "path_length_vs_nodes.png"),
        ("density", "Graph density", "density_vs_nodes.png"),
        ("diameter", "Diameter" + lcc_note, "diameter_vs_nodes.png"),
        ("n_components", "Number of connected components", "components_vs_nodes.png"),
    ]
    for key, ylabel, out_name in specs:
        if key in skip_metrics:
            continue
        plot_metric_vs_nodes(
            nodes, [r[key] for r in rows], ylabel,
            os.path.join(output_dir, out_name),
            log_x.get(key, False), log_y.get(key, False),
            fit_trend, n_boot, seed, dpi, power_law=(key in power_law_fit),
        )

    print(f"\nSaved table and figures to: {output_dir}")


if __name__ == "__main__":
    # ---- Edit these values ----
    graphs_dir = "...1_data/arpeggio/no_iac/no_iac_test_graphml"             # folder containing your .graphml files
    output_dir = "....1_data/arpeggio/iac/0_network_metrics/no_iac_test_met"    # where the CSV and figures are written
    use_largest_component = True      # path length / diameter on the LCC
    exact_max_nodes = 5000            # above this, estimate paths by sampling
    n_samples = 500                   # BFS source nodes used when estimating
    seed = 0                          # makes the sampling reproducible
    # Log-scale axes, set per metric (True = log, False = linear). All linear by
    # default so every figure shares the same x-axis.
    log_x = {
        "mean_degree": False,
        "avg_clustering": False,
        "char_path_length": False,
        "density": False,
        "diameter": False,
        "n_components": False,
    }
    log_y = {
        "mean_degree": False,
        "avg_clustering": False,
        "char_path_length": False,
        "density": False,
        "diameter": False,
        "n_components": False,
    }
    # Metrics whose trend line is a power-law fit (straight in log-log space)
    # drawn as a curve on the axes above. Density falls roughly as 1/n.
    power_law_fit = ["density"]
    # Metrics that get no figure (they stay in the CSV). With every graph
    # connected, n_components is 1 everywhere and a plot of it is just a flat line.
    skip_metrics = ["n_components"]
    fit_trend = True                  # red trend line + pink bootstrap fits
    n_boot = 100                      # number of bootstrap fits drawn
    log_log_degree = True             # log-log axes for degree distributions
    per_graph_degree_plots = False    # one degree-distribution file per graph
    dpi = 300
    # ---------------------------

    main(
        graphs_dir=graphs_dir,
        output_dir=output_dir,
        use_largest_component=use_largest_component,
        exact_max_nodes=exact_max_nodes,
        n_samples=n_samples,
        seed=seed,
        log_x=log_x,
        log_y=log_y,
        skip_metrics=skip_metrics,
        power_law_fit=power_law_fit,
        fit_trend=fit_trend,
        n_boot=n_boot,
        log_log_degree=log_log_degree,
        per_graph_degree_plots=per_graph_degree_plots,
        dpi=dpi,
    )

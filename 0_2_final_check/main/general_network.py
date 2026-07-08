# general_network.py
#
# Purpose: compute whole-graph (protein-level) topological statistics for
# every RIN in a reference (non-enzyme) set and a test (enzyme) set, then
# plot those statistics against protein size (number of residues/nodes).

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
from scipy import stats

from figure_style_ver2 import (
    apply_style, save_fig, COLORS, COLOR_CYCLE, MARKER_CYCLE,
    FIG_SIZE, FONT, LAYOUT, DPI, legend_handles,
)


# Logging (mirrors parallel_RIN_analysis.py)
def setup_logger(log_dir: Path, log_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_dir / f"{log_name}.log", mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Core per-protein statistics
def _harmonic_path_length(g):
    """
    Harmonic-mean characteristic path length. Works on disconnected graphs
    without discarding nodes outside the largest component: unreachable
    pairs simply contribute 0 to the sum (their 1/d term is omitted).

    Same method used for RCA in parallel_RIN_analysis.py -- kept consistent
    so L values here are comparable to that pipeline if you use harmonic
    CPL there too.

    Returns 0.0 when fewer than 2 nodes remain.
    """
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    n_pairs = n * (n - 1) / 2
    total = 0.0
    for source in g.nodes():
        lengths = nx.single_source_shortest_path_length(g, source)
        for target, d in lengths.items():
            if target != source and d > 0:
                total += 1.0 / d
    return total / (2 * n_pairs)


def _network_stats_single(graph):
    """
    Compute whole-graph topological statistics for one protein RIN.

    Parameters
    ----------
    graph : NetworkX graph (as read from GraphML). May be a MultiGraph if
            the RIN encodes multiple interaction types as parallel edges --
            this is collapsed to a simple Graph for topology metrics
            (diameter / path length / clustering are edge-existence
            measures, not edge-count measures), while N_edges_raw preserves
            the original multi-edge count for reference.

    Returns
    -------
    dict of stats. Diameter_LCC and AvgPathLength_LCC are computed on the
    largest connected component only (undefined for disconnected graphs);
    CharPathLength_harmonic is computed on the full graph and does not
    require connectivity.
    """
    n_edges_raw = graph.number_of_edges()

    # Collapse multigraphs to simple graphs for topology metrics.
    if graph.is_multigraph():
        g = nx.Graph(graph)
    else:
        g = graph

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    stats_dict = {
        "N_nodes":   n_nodes,
        "N_edges":   n_edges,
        "N_edges_raw": n_edges_raw,
        "Density":   nx.density(g) if n_nodes > 1 else 0.0,
    }

    if n_nodes == 0:
        stats_dict.update({
            "N_components": 0, "LCC_size": 0, "LCC_fraction": 0.0,
            "Diameter_LCC": numpy.nan, "AvgPathLength_LCC": numpy.nan,
            "CharPathLength_harmonic": 0.0,
            "AvgClustering": numpy.nan, "Transitivity": numpy.nan,
        })
        return stats_dict

    components = list(nx.connected_components(g))
    lcc_nodes = max(components, key=len)
    lcc = g.subgraph(lcc_nodes).copy()

    stats_dict["N_components"] = len(components)
    stats_dict["LCC_size"]     = lcc.number_of_nodes()
    stats_dict["LCC_fraction"] = lcc.number_of_nodes() / n_nodes

    if lcc.number_of_nodes() >= 2:
        stats_dict["Diameter_LCC"]      = nx.diameter(lcc)
        stats_dict["AvgPathLength_LCC"] = nx.average_shortest_path_length(lcc)
    else:
        stats_dict["Diameter_LCC"]      = 0
        stats_dict["AvgPathLength_LCC"] = 0.0

    stats_dict["CharPathLength_harmonic"] = _harmonic_path_length(g)

    # del Sol et al. (2006) report C as the average local clustering
    # coefficient (Watts-Strogatz style). Transitivity (global C, ratio of
    # triangles to connected triples) is included as a second, related
    # measure -- the two can diverge for degree-heterogeneous networks.
    stats_dict["AvgClustering"] = nx.average_clustering(g)
    stats_dict["Transitivity"]  = nx.transitivity(g)

    return stats_dict


def worker_network_stats(args):
    """
    Top-level worker so multiprocessing can pickle it.

    args: (graphml_path_str,)

    Returns
    -------
    (base_name, result) where result is either a stats dict or an
    "Error: <msg>" string.
    """
    (graphml_path_str,) = args
    graphml_path = Path(graphml_path_str)
    base_name    = graphml_path.stem.split("_")[0]

    try:
        graph = nx.read_graphml(graphml_path)
    except Exception as exc:
        return base_name, f"Error loading graph: {exc}"

    try:
        result = _network_stats_single(graph)
        result["Protein"] = base_name
        return base_name, result
    except Exception as exc:
        return base_name, f"Error computing stats: {exc}"


# Resume + parallel driver for one dataset (ref or test)
def _check_resume(graphml_files, output_csv, overwrite):
    """
    Returns (to_process, existing_df).

    If overwrite=True, everything is (re)processed and existing_df is empty.
    Otherwise, proteins already present in output_csv are skipped.
    """
    output_csv = Path(output_csv)

    if overwrite or not output_csv.exists():
        return list(graphml_files), pd.DataFrame()

    try:
        existing_df = pd.read_csv(output_csv)
    except Exception:
        return list(graphml_files), pd.DataFrame()

    done = set(existing_df["Protein"].astype(str)) if "Protein" in existing_df.columns else set()
    to_process = [f for f in graphml_files if f.stem.split("_")[0] not in done]
    return to_process, existing_df


def compute_network_stats_for_dataset(
    graphs_dir,
    output_csv,
    group_label,
    num_processes=4,
    overwrite=False,
    logger=None,
):
    """
    Compute whole-graph statistics for every *.graphml file in graphs_dir
    and write/append them to output_csv (one row per protein, plus a
    "Group" column set to group_label).

    Resume behaviour: proteins already present in output_csv are skipped
    unless overwrite=True.

    Parameters
    ----------
    graphs_dir    : path-like -- directory of *.graphml RIN files.
    output_csv    : path-like -- combined stats CSV for this dataset.
    group_label   : str       -- e.g. "enzyme" or "non-enzyme"; stored in
                                  a "Group" column for downstream plotting.
    num_processes : int       -- parallel workers.
    overwrite     : bool      -- if True, recompute all proteins from scratch.
    logger        : logging.Logger, optional.

    Returns
    -------
    pd.DataFrame -- the full combined stats table for this dataset
                    (existing rows + newly computed rows).
    """
    graphs_dir = Path(graphs_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    graphml_files = sorted(graphs_dir.glob("*.graphml"))
    if not graphml_files:
        log(f"[{group_label}] No GraphML files found in {graphs_dir}.")
        return pd.DataFrame()

    to_process, existing_df = _check_resume(graphml_files, output_csv, overwrite)
    log(f"[{group_label}] {len(graphml_files)} total, "
        f"{len(graphml_files) - len(to_process)} already done, "
        f"{len(to_process)} to process.")

    new_rows = []
    if to_process:
        work_args = [(str(f),) for f in to_process]
        try:
            with mp.Pool(processes=num_processes) as pool:
                for base_name, result in tqdm(
                    pool.imap_unordered(worker_network_stats, work_args),
                    total=len(work_args),
                    desc=group_label,
                ):
                    if isinstance(result, dict):
                        result["Group"] = group_label
                        new_rows.append(result)
                    else:
                        log(f"  FAILED {base_name}: {result}")
        except KeyboardInterrupt:
            log(f"[{group_label}] KeyboardInterrupt -- pool shut down. "
                f"{len(new_rows)} protein(s) computed this session. Re-run to resume.")

    new_df = pd.DataFrame(new_rows)

    if overwrite:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        if "Protein" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset="Protein", keep="last")

    if not combined_df.empty:
        combined_df = combined_df.sort_values("Protein").reset_index(drop=True)
        combined_df.to_csv(output_csv, index=False)
        log(f"[{group_label}] Wrote {len(combined_df)} rows -> {output_csv}")

    return combined_df


def combine_datasets(test_df, ref_df, combined_csv=None):
    """
    Stack the test (enzyme) and reference (non-enzyme) stats tables into one
    DataFrame for plotting. Optionally writes the combined table to disk.

    Parameters
    ----------
    test_df, ref_df : pd.DataFrame -- outputs of compute_network_stats_for_dataset.
    combined_csv    : path-like, optional -- if given, saves the combined table.

    Returns
    -------
    pd.DataFrame
    """
    combined = pd.concat([test_df, ref_df], ignore_index=True)
    if combined_csv is not None:
        combined_csv = Path(combined_csv)
        combined_csv.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_csv, index=False)
        print(f"Wrote combined stats ({len(combined)} rows) -> {combined_csv}")
    return combined


# Plotting -- one figure per file, per PLOS style conventions
def _scatter_by_group(ax, df, x_col, y_col, group_col="Group", log_x=False):
    """
    Shared scatter-plot helper: plots each group with a distinct color/marker
    from figure_style_ver2, annotates each group's Pearson correlation, and
    returns the legend labels used (for legend_handles()).
    """
    groups = sorted(df[group_col].dropna().unique())
    stats_lines = []

    for i, grp in enumerate(groups):
        sub = df[df[group_col] == grp].dropna(subset=[x_col, y_col])
        if sub.empty:
            continue

        x = sub[x_col].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)

        ax.scatter(
            x, y,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            marker=MARKER_CYCLE[i % len(MARKER_CYCLE)],
            s=LAYOUT["marker_size"] * 4,
            alpha=0.65,
            edgecolor="white",
            linewidth=0.4,
            label=f"{grp} (n={len(sub)})",
        )

        if len(sub) >= 3:
            x_for_corr = numpy.log10(x) if log_x and numpy.all(x > 0) else x
            r, p = stats.pearsonr(x_for_corr, y)
            stats_lines.append(f"{grp}: r={r:.2f}, p={p:.1e}")

    if log_x:
        ax.set_xscale("log")

    if stats_lines:
        ax.text(
            0.02, 0.98, "\n".join(stats_lines),
            transform=ax.transAxes,
            fontsize=FONT["size_annot"],
            va="top", ha="left",
        )

    return groups


def plot_diameter_vs_size(df, out_path, log_x=True):
    """
    Figure: network diameter (largest connected component) vs. protein size.
    Saved as its own file (out_path).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["single_col"])

    groups = _scatter_by_group(ax, df, "N_nodes", "Diameter_LCC", log_x=log_x)

    ax.set_xlabel("Protein size (number of residues)" + (" [log scale]" if log_x else ""),
                  fontsize=FONT["size_label"])
    ax.set_ylabel("Network diameter (LCC)", fontsize=FONT["size_label"])
    ax.set_title("Diameter scales with protein size", fontsize=FONT["size_title"])

    handles = legend_handles([f"{g}" for g in groups])
    ax.legend(handles=handles, fontsize=FONT["size_legend"], loc="lower right")

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, out_path)
    plt.close(fig)


def plot_pathlength_vs_size(df, out_path, log_x=True, method="harmonic"):
    """
    Figure: characteristic path length vs. protein size.

    method : "harmonic" -> CharPathLength_harmonic (whole graph, robust to
                            disconnection; recommended default).
             "lcc"       -> AvgPathLength_LCC (largest component only).
    """
    y_col = "CharPathLength_harmonic" if method == "harmonic" else "AvgPathLength_LCC"

    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["single_col"])

    groups = _scatter_by_group(ax, df, "N_nodes", y_col, log_x=log_x)

    ax.set_xlabel("Protein size (number of residues)" + (" [log scale]" if log_x else ""),
                  fontsize=FONT["size_label"])
    ylabel = "Characteristic path length" + (" (harmonic)" if method == "harmonic" else " (LCC)")
    ax.set_ylabel(ylabel, fontsize=FONT["size_label"])
    ax.set_title("Path length scales with protein size", fontsize=FONT["size_title"])

    handles = legend_handles([f"{g}" for g in groups])
    ax.legend(handles=handles, fontsize=FONT["size_legend"], loc="lower right")

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, out_path)
    plt.close(fig)


def plot_clustering_vs_size(df, out_path, log_x=True, metric="AvgClustering"):
    """
    Figure: clustering coefficient vs. protein size.

    metric : "AvgClustering" (del Sol et al.-style local-average C, default)
             or "Transitivity" (global C).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE["single_col"])

    groups = _scatter_by_group(ax, df, "N_nodes", metric, log_x=log_x)

    ax.set_xlabel("Protein size (number of residues)" + (" [log scale]" if log_x else ""),
                  fontsize=FONT["size_label"])
    ax.set_ylabel(f"Clustering coefficient ({metric})", fontsize=FONT["size_label"])
    ax.set_title("Clustering coefficient vs. protein size", fontsize=FONT["size_title"])

    handles = legend_handles([f"{g}" for g in groups])
    ax.legend(handles=handles, fontsize=FONT["size_legend"], loc="upper right")

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    save_fig(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    num_processes=4,

    # Test set (enzymes)
    test_graphs_dir=DATA_DIR_3 / "test" / "RINs",
    test_stats_csv=DATA_DIR_3 / "general" / "test_network_stats.csv",
    test_group_label="enzyme",

    # Reference set (non-enzymes)
    ref_graphs_dir=DATA_DIR_3 / "ref" / "RINs",
    ref_stats_csv=DATA_DIR_3 / "general" / "ref_network_stats.csv",
    ref_group_label="non-enzyme",

    overwrite=False,

    # Combined table
    combined_csv= DATA_DIR_3 / "general" / "network_stats_combined.csv",

    # Figures
    make_figures=True,
    figs_dir= FIGS_DIR / "network_scaling",
    log_x=True,
    pathlength_method="harmonic",   # "harmonic" or "lcc"
    clustering_metric="AvgClustering",  # "AvgClustering" or "Transitivity"
):
    """
    Full pipeline: compute per-protein whole-graph network statistics for the
    test (enzyme) and reference (non-enzyme) RIN sets, combine them, and
    (optionally) produce the three scaling figures described in the module
    docstring.

    Every argument here is safe to override -- either by calling
    main(...) with keyword arguments, or by editing the values passed in
    the `if __name__ == "__main__":` block below.
    """
    log_dir = DATA_DIR_3 / "logs"
    logger  = setup_logger(log_dir, "general_network")

    logger.info("=" * 60)
    logger.info("GENERAL NETWORK STATS SESSION STARTED")

    logger.info("-" * 40)
    logger.info("Test set (enzymes)")
    test_df = compute_network_stats_for_dataset(
        graphs_dir    = test_graphs_dir,
        output_csv    = test_stats_csv,
        group_label   = test_group_label,
        num_processes = num_processes,
        overwrite     = overwrite,
        logger        = logger,
    )

    logger.info("-" * 40)
    logger.info("Reference set (non-enzymes)")
    ref_df = compute_network_stats_for_dataset(
        graphs_dir    = ref_graphs_dir,
        output_csv    = ref_stats_csv,
        group_label   = ref_group_label,
        num_processes = num_processes,
        overwrite     = overwrite,
        logger        = logger,
    )

    combined_df = combine_datasets(test_df, ref_df, combined_csv=combined_csv)

    if make_figures and not combined_df.empty:
        figs_dir = Path(figs_dir)
        figs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 40)
        logger.info("Generating figures...")

        plot_diameter_vs_size(
            combined_df, figs_dir / "diameter_vs_size.tiff", log_x=log_x,
        )
        plot_pathlength_vs_size(
            combined_df, figs_dir / "pathlength_vs_size.tiff",
            log_x=log_x, method=pathlength_method,
        )
        plot_clustering_vs_size(
            combined_df, figs_dir / "clustering_vs_size.tiff",
            log_x=log_x, metric=clustering_metric,
        )

        logger.info(f"Figures written to {figs_dir}")

    logger.info("GENERAL NETWORK STATS SESSION DONE")
    logger.info("=" * 60)

    return combined_df


if __name__ == "__main__":

    main(
        num_processes = 4,
        overwrite = False,
        make_figures       = True,
        figs_dir            = FIGS_DIR / "network_scaling",
        log_x               = True,
        pathlength_method   = "harmonic",
        clustering_metric   = "AvgClustering",
    )

# parallel_RIN_centrality_ver1.py
#
# Purpose: Compute RAW residue centralities for every protein graph in every
# graph-construction dataset (7 datasets x ref/test). Standardization is done
# in a separate script, after the raw-centrality-vs-size plots have been
# checked.
#
# Changes from parallel_RIN_analysis.py:
#   - Loops over any number of datasets, each with its own ref and test folder.
#   - Reads both graph formats (Arpeggio and RING/PCN) through a node-attribute
#     map, so every output CSV has the same column names.
#   - RCA and PageRank_zscore removed. Only raw values are written:
#       Degree, DegreeCentrality, ClosenessCentrality,
#       BetweennessCentrality, PageRank, PageRank_xN
#   - Adds identity columns (PDBid, dataset, set_name) and a ResidueKey
#     (chain_number, e.g. "A_523") that is the same for every tool, so the
#     same residue can be matched across datasets.
#   - Adds graph-size columns (n_nodes, n_edges, n_components) to every row,
#     ready for the raw-centrality-vs-size plots.
#   - Writes to a NEW folder:  <OUTPUT_ROOT>/<dataset>/<set_name>/<PDBid>_centrality.csv
#     Input graphs are never modified.
#   - All centralities are computed in one pass per protein (one graph load).
#
# Column definitions (one row per residue / node):
#   PDBid, dataset, set_name    identity
#   ResidueKey                  "<chain>_<residue number>", shared across tools
#   Node                        node ID as written by the graph tool
#   Residue, Chain, Residue number
#   IsCanonical                 True for the 20 standard amino acids
#   n_nodes, n_edges, n_components   size of the protein graph (same on every row)
#   Degree                      raw number of contacting residues
#   DegreeCentrality            Degree / (n_nodes - 1)
#   ClosenessCentrality         nx.closeness_centrality (Wasserman-Faust
#                               correction for disconnected graphs)
#   BetweennessCentrality       nx.betweenness_centrality, normalized=True
#   PageRank                    nx.pagerank (sums to 1 within each protein)
#   PageRank_xN                 PageRank * n_nodes (mean is exactly 1 in every
#                               protein; size-invariant version kept in case the
#                               size check shows it is needed)
#
# Graphs are treated as simple, undirected and unweighted, with self-loops
# removed (the same view used by protein_size_distribution.py).
#
# Resume behaviour: a CSV that already exists and contains every column in
# REQUIRED_COLUMNS is skipped unless OVERWRITE = True. Incomplete or unreadable
# CSVs are recomputed. Writes go to a .tmp file that is renamed on success, so an
# interrupted run never leaves a half-written CSV.
#
# All settings live in the `if __name__ == "__main__":` block at the bottom.

import csv
import logging
import multiprocessing as mp
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm


CANONICAL_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

OUTPUT_COLUMNS = [
    "PDBid", "dataset", "set_name", "ResidueKey",
    "Node", "Residue", "Chain", "Residue number", "IsCanonical",
    "n_nodes", "n_edges", "n_components",
    "Degree", "DegreeCentrality", "ClosenessCentrality",
    "BetweennessCentrality", "PageRank", "PageRank_xN",
]

# A CSV is complete only if every one of these columns is present.
REQUIRED_COLUMNS = set(OUTPUT_COLUMNS)


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, log_name: str) -> logging.Logger:
    """Console (INFO+) and appending log file (DEBUG+)."""
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


# ── graph loading ─────────────────────────────────────────────────────────────

def protein_id_from_path(path: Path) -> str:
    """'1AC0_clean.graphml' -> '1AC0';  '1AC0.graphml' -> '1AC0'."""
    return path.stem.split("_")[0]


def load_simple_graph(path: Path) -> nx.Graph:
    """Read GraphML as a simple undirected graph without self-loops, keeping node attributes."""
    G = nx.read_graphml(path)
    H = nx.Graph(G.to_undirected()) if G.is_directed() else nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


# ── single-protein computation (runs inside worker processes) ────────────────

def compute_protein_centralities(
    G: nx.Graph,
    attr_map: dict,
    pdb_id: str,
    dataset: str,
    set_name: str,
    pagerank_alpha: float,
    pagerank_max_iter: int,
) -> tuple[list, list]:
    """
    Compute all raw centralities for one protein graph.

    Returns
    -------
    rows     : list of dicts, one per node, keys = OUTPUT_COLUMNS
    warnings : list of str (e.g. duplicate ResidueKeys, missing attributes)
    """
    warnings = []
    n = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)

    degree = dict(G.degree())
    degree_c = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    pagerank = nx.pagerank(G, alpha=pagerank_alpha, max_iter=pagerank_max_iter)

    rows = []
    seen_keys = {}
    for node, attrs in G.nodes(data=True):
        missing = [std for std, raw in attr_map.items() if raw not in attrs]
        if missing:
            warnings.append(f"node {node} missing attribute(s) {missing}")

        residue = str(attrs.get(attr_map["Residue"], "")).upper()
        chain = attrs.get(attr_map["Chain"], "")
        resnum = attrs.get(attr_map["Residue number"], "")
        key = f"{chain}_{resnum}"
        if key in seen_keys:
            warnings.append(f"duplicate ResidueKey {key} (nodes {seen_keys[key]} and {node}); "
                            "possible insertion code")
        seen_keys[key] = node

        rows.append({
            "PDBid": pdb_id,
            "dataset": dataset,
            "set_name": set_name,
            "ResidueKey": key,
            "Node": node,
            "Residue": residue,
            "Chain": chain,
            "Residue number": resnum,
            "IsCanonical": residue in CANONICAL_AA,
            "n_nodes": n,
            "n_edges": n_edges,
            "n_components": n_components,
            "Degree": degree[node],
            "DegreeCentrality": degree_c.get(node, 0.0),
            "ClosenessCentrality": closeness.get(node, 0.0),
            "BetweennessCentrality": betweenness.get(node, 0.0),
            "PageRank": pagerank[node],
            "PageRank_xN": pagerank[node] * n,
        })

    if n_components > 1:
        warnings.append(f"graph has {n_components} connected components")
    return rows, warnings


def csv_is_complete(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return False
    return REQUIRED_COLUMNS.issubset(cols)


def worker(args: tuple) -> tuple:
    """
    One protein: load graph, compute centralities, write CSV.

    args: (graph_path, out_dir, dataset, set_name, attr_map,
           pagerank_alpha, pagerank_max_iter, overwrite)

    Returns (dataset, set_name, pdb_id, status, warnings) where status is
    "ok", "skipped" or "Error: <msg>".
    """
    (graph_path, out_dir, dataset, set_name, attr_map,
     pagerank_alpha, pagerank_max_iter, overwrite) = args
    graph_path, out_dir = Path(graph_path), Path(out_dir)
    pdb_id = protein_id_from_path(graph_path)
    out_path = out_dir / f"{pdb_id}_centrality.csv"
    tmp_path = out_dir / f"{pdb_id}_centrality.csv.tmp"

    if not overwrite and csv_is_complete(out_path):
        return dataset, set_name, pdb_id, "skipped", []

    try:
        G = load_simple_graph(graph_path)
        rows, warnings = compute_protein_centralities(
            G, attr_map, pdb_id, dataset, set_name, pagerank_alpha, pagerank_max_iter
        )
        with open(tmp_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(out_path)
        return dataset, set_name, pdb_id, "ok", warnings

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return dataset, set_name, pdb_id, f"Error: {exc}", []


# ── job building and running ──────────────────────────────────────────────────

def build_jobs(datasets, node_attr_maps, output_root, graph_glob,
               pagerank_alpha, pagerank_max_iter, overwrite, expected_counts, logger):
    """One job per graph file across all datasets and sets."""
    jobs = []
    for dataset, cfg in datasets.items():
        fmt = cfg["format"]
        if fmt not in node_attr_maps:
            raise ValueError(f"[{dataset}] unknown format '{fmt}'. "
                             f"Known formats: {list(node_attr_maps)}")
        attr_map = node_attr_maps[fmt]

        for set_name in ("ref", "test"):
            graph_dir = Path(cfg[set_name])
            out_dir = Path(output_root) / dataset / set_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for stale in out_dir.glob("*.tmp"):
                stale.unlink()

            files = sorted(graph_dir.glob(graph_glob))
            if not files:
                logger.warning(f"[{dataset}/{set_name}] no '{graph_glob}' files in {graph_dir}")
                continue

            ids = [protein_id_from_path(f) for f in files]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                logger.warning(f"[{dataset}/{set_name}] several graph files map to the same "
                               f"PDB ID (only one CSV each will survive): {sorted(dupes)}")

            expected = expected_counts.get(set_name)
            if expected is not None and len(files) != expected:
                logger.warning(f"[{dataset}/{set_name}] {len(files)} graphs, expected {expected}")
            logger.info(f"[{dataset}/{set_name}] {len(files)} graphs ({fmt}) -> {out_dir}")

            for f in files:
                jobs.append((str(f), str(out_dir), dataset, set_name, attr_map,
                             pagerank_alpha, pagerank_max_iter, overwrite))
    return jobs


def run_jobs(jobs, num_processes, logger):
    counts = {"ok": 0, "skipped": 0, "error": 0}
    n_warned = 0
    try:
        with mp.Pool(processes=num_processes) as pool:
            for dataset, set_name, pdb_id, status, warnings in tqdm(
                pool.imap_unordered(worker, jobs), total=len(jobs), desc="Centralities"
            ):
                tag = f"[{dataset}/{set_name}] {pdb_id}"
                if status.startswith("Error"):
                    counts["error"] += 1
                    logger.warning(f"FAILED {tag}: {status}")
                else:
                    counts[status] += 1
                    logger.debug(f"{status} {tag}")
                if warnings:
                    n_warned += 1
                    for w in warnings:
                        logger.warning(f"{tag}: {w}")
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt: pool shut down. Re-run to resume.")
        return counts

    logger.info(f"Done: computed {counts['ok']}, skipped (already complete) {counts['skipped']}, "
                f"errors {counts['error']}, proteins with warnings {n_warned}")
    return counts


def write_graph_summary(datasets, output_root, logger):
    """
    One row per protein across every dataset/set: PDBid, dataset, set_name,
    n_nodes, n_edges, n_components, mean_degree. Read back from the CSVs so
    proteins skipped on resume are included.
    """
    rows = []
    for dataset in datasets:
        for set_name in ("ref", "test"):
            for csv_path in sorted((Path(output_root) / dataset / set_name).glob("*_centrality.csv")):
                df = pd.read_csv(csv_path, usecols=["PDBid", "n_nodes", "n_edges",
                                                    "n_components", "Degree"])
                if df.empty:
                    continue
                rows.append({
                    "PDBid": df["PDBid"].iloc[0],
                    "dataset": dataset,
                    "set_name": set_name,
                    "n_nodes": df["n_nodes"].iloc[0],
                    "n_edges": df["n_edges"].iloc[0],
                    "n_components": df["n_components"].iloc[0],
                    "mean_degree": df["Degree"].mean(),
                })
    summary = pd.DataFrame(rows)
    out = Path(output_root) / "graph_summary.csv"
    summary.to_csv(out, index=False)
    logger.info(f"Graph summary ({len(summary)} proteins) written to {out}")

    if not summary.empty:
        table = (summary.groupby(["dataset", "set_name"], sort=False)
                 .agg(n_proteins=("PDBid", "count"),
                      median_nodes=("n_nodes", "median"),
                      median_edges=("n_edges", "median"),
                      mean_degree=("mean_degree", "mean"),
                      disconnected=("n_components", lambda x: int((x > 1).sum())))
                 .reset_index())
        logger.info("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    datasets: dict,
    node_attr_maps: dict,
    output_root,
    graph_glob: str,
    num_processes: int,
    pagerank_alpha: float,
    pagerank_max_iter: int,
    overwrite: bool,
    expected_counts: dict,
):
    output_root = Path(output_root)
    logger = setup_logger(output_root / "logs", "rin_centrality")
    logger.info("=" * 60)
    logger.info("RAW CENTRALITY SESSION STARTED")

    jobs = build_jobs(datasets, node_attr_maps, output_root, graph_glob,
                      pagerank_alpha, pagerank_max_iter, overwrite, expected_counts, logger)
    logger.info(f"Total graphs queued: {len(jobs)}")
    run_jobs(jobs, num_processes, logger)
    write_graph_summary(datasets, output_root, logger)

    logger.info("RAW CENTRALITY SESSION DONE")
    logger.info("=" * 60)


if __name__ == "__main__":

    # ── Edit these values ─────────────────────────────────────────────────────

    # Node-attribute names used by each graph tool, mapped to the standard
    # column names written to every CSV.
    NODE_ATTR_MAPS = {
        "arpeggio": {"Residue": "residue_type", "Chain": "chain", "Residue number": "residue_num"},
        "ring":     {"Residue": "residue",      "Chain": "chain", "Residue number": "position"},
    }

    # One entry per graph construction. "format" picks the attribute map above
    # (PCN graphs were made with RING, so they use "ring").
    # PLACEHOLDER paths: replace with your real ref/test graph folders.
    DATASETS = {
        "arp_iac": {"format": "arpeggio", "ref": ".../iac_ref_graphml", "test": ".../iac_test_graphml"}
    }

    # New output folder: <OUTPUT_ROOT>/<dataset>/<ref|test>/<PDBid>_centrality.csv
    OUTPUT_ROOT = Path(".../4_centralities")

    GRAPH_GLOB = "*.graphml"
    NUM_PROCESSES = 4
    PAGERANK_ALPHA = 0.85          # damping factor (Brin & Page 1998)
    PAGERANK_MAX_ITER = 1000       # raise if any graph fails to converge
    OVERWRITE = False              # True = recompute even complete CSVs
    EXPECTED_COUNTS = {"ref": 534, "test": 232}   # warn if a folder differs

    # ──────────────────────────────────────────────────────────────────────────

    main(
        datasets=DATASETS,
        node_attr_maps=NODE_ATTR_MAPS,
        output_root=OUTPUT_ROOT,
        graph_glob=GRAPH_GLOB,
        num_processes=NUM_PROCESSES,
        pagerank_alpha=PAGERANK_ALPHA,
        pagerank_max_iter=PAGERANK_MAX_ITER,
        overwrite=OVERWRITE,
        expected_counts=EXPECTED_COUNTS,
    )

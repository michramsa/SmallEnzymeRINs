# The purpose of this file is to take the RINs (as GraphML files) and compute
# centrality measures for every residue (node) in each protein.
#
# Output: one _centrality.csv per protein, containing:
#   Node, Residue, Chain, Residue number,
#   DegreeCentrality, ClosenessCentrality,   <- computed in the parallel pass
#   Degree,                                  <- appended by append_degree_to_centrality()
#   BetweennessCentrality,                   <- appended by append_betweenness_to_centrality()
#   RCA_deltaL, RCA_zscore, IsCICD,          <- appended by append_rca_to_centrality()
#   PageRank, StdPageRank                <- appended by append_pagerank_to_centrality()
#
# Two datasets are processed:
#   Test set  : BioLiP / single-chain enzymes  (DATA_DIR_3 / "test" / "RINs")
#   Reference : reorganised reference proteins  (DATA_DIR_3 / "ref"  / "RINs")
#
# Parallelized: each worker takes one GraphML file and writes one _centrality.csv.
#
# Resume behaviour (both sets):
#   - A CSV is considered COMPLETE if it exists and contains all expected columns:
#       DegreeCentrality, ClosenessCentrality, Degree, BetweennessCentrality,
#       RCA_deltaL, RCA_zscore, IsCICD, PageRank, StdPageRank
#   - COMPLETE proteins are skipped entirely.
#   - INCOMPLETE CSVs (file exists but columns missing) are deleted before the
#     parallel pass so the protein is reprocessed from scratch.
#   - ABSENT proteins are processed normally.
#
# Ver6 changes from Ver5:
#   - Updated paths to DATA_DIR_3 / "test" and DATA_DIR_3 / "ref".
#   - Added _check_resume() helper implementing the complete/incomplete/absent logic.
#   - Resume support now covers both test and reference sets.
#   - Incomplete CSVs are deleted (with a warning log) before the parallel pass.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# Columns that must all be present for a _centrality.csv to be considered complete.
REQUIRED_COLUMNS = {
    "DegreeCentrality",
    "ClosenessCentrality",
    "Degree",
    "BetweennessCentrality",
    "RCA_deltaL",
    "RCA_zscore",
    "IsCICD",
    "PageRank",
    "StdPageRank",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(log_dir: Path, log_name: str) -> logging.Logger:
    """
    Creates a logger writing to both console (INFO+) and a log file (DEBUG+).
    Appends across runs so the full history of every session is preserved.
    """
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


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------
def _check_resume(graphml_files, centrality_dir, logger, dataset_label):
    """
    Inspect existing _centrality.csv files and classify each protein as:
        - complete   : CSV exists and contains all REQUIRED_COLUMNS -> skip entirely.
        - incomplete : CSV exists but is missing one or more columns -> delete and reprocess.
        - absent     : no CSV found -> process normally.

    Incomplete CSVs are deleted here (before the parallel pass) so that
    _compute_centrality sees them as absent and writes fresh output.

    Parameters
    ----------
    graphml_files  : list of Path -- all GraphML files for this dataset.
    centrality_dir : Path         -- directory where _centrality.csv files live.
    logger         : logging.Logger
    dataset_label  : str

    Returns
    -------
    to_process : list of Path -- GraphML files that need (re)processing.
    n_complete : int          -- number of already-complete proteins skipped.
    """
    centrality_dir = Path(centrality_dir)
    centrality_dir.mkdir(parents=True, exist_ok=True)

    to_process  = []
    n_complete  = 0
    n_incomplete = 0

    for gml in graphml_files:
        base_name = gml.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"

        if not csv_path.exists():
            to_process.append(gml)
            continue

        try:
            existing_cols = set(pd.read_csv(csv_path, nrows=0).columns)
        except Exception as exc:
            logger.warning(f"[{dataset_label}] Could not read {csv_path.name} "
                           f"({exc}) — will reprocess.")
            csv_path.unlink(missing_ok=True)
            to_process.append(gml)
            continue

        missing = REQUIRED_COLUMNS - existing_cols
        if missing:
            logger.warning(
                f"[{dataset_label}] INCOMPLETE {base_name} "
                f"(missing: {', '.join(sorted(missing))}) — deleting and reprocessing."
            )
            csv_path.unlink()
            to_process.append(gml)
            n_incomplete += 1
        else:
            n_complete += 1

    logger.info(
        f"[{dataset_label}] Resume check — "
        f"complete: {n_complete}, incomplete (deleted): {n_incomplete}, "
        f"absent: {len(to_process) - n_incomplete} | "
        f"to process: {len(to_process)}"
    )
    return to_process, n_complete


# ---------------------------------------------------------------------------
# Core single-protein centrality function (called inside worker processes)
# ---------------------------------------------------------------------------
def _compute_centrality(graph, base_name, centrality_dir):
    """
    Compute degree + closeness centrality for every node in `graph` and write
    the results to <centrality_dir>/<base_name>_centrality.csv.

    Uses a write-to-temp-then-rename pattern to prevent corrupt outputs if the
    process is interrupted mid-write.

    Returns
    -------
    "skipped"       -- output CSV already exists; nothing done.
    "ok"            -- CSV written successfully.
    "Error: <msg>"  -- an exception was raised; temp file cleaned up.
    """
    centrality_dir = Path(centrality_dir)
    centrality_dir.mkdir(parents=True, exist_ok=True)

    out_path = centrality_dir / f"{base_name}_centrality.csv"
    tmp_path = centrality_dir / f"{base_name}_centrality.csv.tmp"

    if out_path.exists():
        return "skipped"

    try:
        degree_centrality    = nx.degree_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)

        combined_data = []
        for node, attrs in graph.nodes(data=True):
            combined_data.append({
                "Node":                node,
                "Residue":             attrs["residue_type"],
                "Chain":               attrs["chain"],
                "Residue number":      attrs["residue_num"],
                "DegreeCentrality":    degree_centrality.get(node, 0),
                "ClosenessCentrality": closeness_centrality.get(node, 0),
            })

        with open(tmp_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["Node", "Residue", "Chain", "Residue number",
                            "DegreeCentrality", "ClosenessCentrality"],
            )
            writer.writeheader()
            writer.writerows(combined_data)

        tmp_path.rename(out_path)
        return "ok"

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Worker function (top-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------
def worker_centrality(args):
    """
    Worker for centrality computation -- shared by both the test set and the
    reference set.

    args: (graphml_path_str, centrality_dir_str)

    Returns
    -------
    (base_name, centrality_result) where centrality_result is one of:
        "skipped"       -- CSV already exists.
        "ok"            -- CSV written successfully.
        "Error: <msg>"  -- exception during graph load or centrality computation.
    """
    graphml_path_str, centrality_dir_str = args
    graphml_path = Path(graphml_path_str)
    base_name    = graphml_path.stem.split("_")[0]

    try:
        graph = nx.read_graphml(graphml_path)
    except Exception as exc:
        return base_name, f"Error loading graph: {exc}"

    cent_result = _compute_centrality(graph, base_name, centrality_dir_str)
    return base_name, cent_result


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
def run_parallel(work_args, logger, num_processes, dataset_label):
    """
    Runs worker_centrality over work_args using a multiprocessing Pool.
    Logs per-file outcomes and prints a session summary on completion.
    Handles KeyboardInterrupt gracefully.

    Parameters
    ----------
    work_args     : list of (graphml_path_str, centrality_dir_str) tuples.
    logger        : logging.Logger instance.
    num_processes : int -- number of parallel worker processes.
    dataset_label : str -- label shown in log messages and the tqdm bar.
    """
    if not work_args:
        logger.info(f"[{dataset_label}] Nothing to process.")
        return

    logger.info(f"[{dataset_label}] Submitting {len(work_args)} proteins to "
                f"{num_processes} worker(s).")

    results = []
    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker_centrality, work_args),
                total=len(work_args),
                desc=dataset_label,
            ):
                results.append(result)
                base_name, cent = result

                if str(cent).startswith("Error"):
                    logger.warning(f"FAILED  {base_name}: {cent}")
                else:
                    logger.debug(f"OK  {base_name}: {cent}")

    except KeyboardInterrupt:
        logger.warning(
            f"[{dataset_label}] KeyboardInterrupt — pool shut down. "
            f"Completed {len(results)} protein(s) this session. Re-run to resume."
        )
        return

    errors = [(b, c) for b, c in results if str(c).startswith("Error")]
    ok     = len(results) - len(errors)
    logger.info(f"[{dataset_label}] Done — {ok}/{len(results)} successful.")
    if errors:
        logger.warning(f"[{dataset_label}] {len(errors)} protein(s) with errors:")
        for b, c in errors:
            logger.warning(f"  {b}: {c}")

def append_degree_to_centrality(graphml_dir, centrality_dir):
    """
    Appends a Degree column (raw integer edge count) to each _centrality.csv
    using node degrees read from the corresponding GraphML.

    Runs sequentially. Skips any CSV that already has a Degree column.

    Parameters
    ----------
    graphml_dir    : path-like -- directory containing *.graphml files.
    centrality_dir : path-like -- directory containing *_centrality.csv files.
    """
    graphml_dir    = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)

    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return

    n_ok = n_skipped = n_missing = n_errors = 0

    for graphml_file in graphml_dir.glob("*.graphml"):
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"

        if not csv_path.exists():
            print(f"  No centrality CSV for {base_name}, skipping.")
            n_missing += 1
            continue

        df = pd.read_csv(csv_path)

        if "Degree" in df.columns:
            n_skipped += 1
            continue

        try:
            graph        = nx.read_graphml(graphml_file)
            degree_dict  = dict(graph.degree())
            df["Degree"] = df["Node"].astype(str).map(degree_dict)

            unmatched = df["Degree"].isna().sum()
            if unmatched > 0:
                print(f"  Warning: {unmatched} unmatched node(s) in {csv_path.name}.")

            df.to_csv(csv_path, index=False)
            print(f"  Appended Degree to {csv_path.name}")
            n_ok += 1

        except Exception as exc:
            print(f"  Error processing {base_name}: {exc}")
            n_errors += 1

    print(
        f"\nappend_degree_to_centrality complete — "
        f"updated: {n_ok}, skipped: {n_skipped}, "
        f"missing CSV: {n_missing}, errors: {n_errors}"
    )

def append_betweenness_to_centrality(graphml_dir, centrality_dir):
    """
    Appends a BetweennessCentrality column to each existing _centrality.csv
    using the corresponding GraphML file in graphml_dir.

    Betweenness is computed with nx.betweenness_centrality(graph, normalized=True).
    Runs sequentially. Skips any CSV that already has a BetweennessCentrality column.

    Parameters
    ----------
    graphml_dir    : path-like -- directory containing *.graphml files.
    centrality_dir : path-like -- directory containing *_centrality.csv files.
    """
    graphml_dir    = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)

    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return

    graphml_files = sorted(graphml_dir.glob("*.graphml"))
    if not graphml_files:
        print(f"No GraphML files found in {graphml_dir}.")
        return

    n_ok = n_skipped = n_missing = n_errors = 0

    for graphml_file in graphml_files:
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"

        if not csv_path.exists():
            print(f"  No centrality CSV for {base_name}, skipping.")
            n_missing += 1
            continue

        df = pd.read_csv(csv_path)

        if "BetweennessCentrality" in df.columns:
            n_skipped += 1
            continue

        try:
            graph = nx.read_graphml(graphml_file)
            betweenness_dict = nx.betweenness_centrality(graph, normalized=True)

            df["BetweennessCentrality"] = df["Node"].astype(str).map(betweenness_dict)

            unmatched = df["BetweennessCentrality"].isna().sum()
            if unmatched > 0:
                print(f"  Warning: {unmatched} unmatched node(s) in {csv_path.name}.")

            df.to_csv(csv_path, index=False)
            print(f"  Appended BetweennessCentrality to {csv_path.name}")
            n_ok += 1

        except Exception as exc:
            print(f"  Error processing {base_name}: {exc}")
            n_errors += 1

    print(
        f"\nappend_betweenness_to_centrality complete — "
        f"updated: {n_ok}, skipped: {n_skipped}, "
        f"missing CSV: {n_missing}, errors: {n_errors}"
    )

def append_pagerank_to_centrality(
    graphml_dir,
    centrality_dir,
    alpha=0.85,
    overwrite=False,
):
    """
    Appends PageRank and StdPageRank columns to each existing
    _centrality.csv using the corresponding GraphML file in graphml_dir.

    PageRank is computed with nx.pagerank(graph, alpha=alpha) following
    Brin & Page (1998). RINs are undirected; nx.pagerank() treats each edge
    as bidirectional, consistent with Brysbaert & Lensink (2021) PRA.

    Z-scores are computed per protein (population std, ddof=0):
        StdPageRank_k = (PageRank_k - mean(PageRank)) / std(PageRank)

    Runs sequentially. Skips any CSV that already has a PageRank column
    unless overwrite=True.

    Parameters
    ----------
    graphml_dir    : path-like -- directory containing *.graphml files.
    centrality_dir : path-like -- directory containing *_centrality.csv files.
    alpha          : float     -- PageRank damping factor (default 0.85).
    overwrite      : bool      -- if True, recomputes even if column exists.
    """
    graphml_dir    = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)

    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return

    graphml_files = sorted(graphml_dir.glob("*.graphml"))
    if not graphml_files:
        print(f"No GraphML files found in {graphml_dir}.")
        return

    n_ok = n_skipped = n_missing = n_errors = 0

    for graphml_file in graphml_files:
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"

        if not csv_path.exists():
            print(f"  No centrality CSV for {base_name}, skipping.")
            n_missing += 1
            continue

        df = pd.read_csv(csv_path)

        if "PageRank" in df.columns and not overwrite:
            n_skipped += 1
            continue

        try:
            graph   = nx.read_graphml(graphml_file)
            pr_dict = nx.pagerank(graph, alpha=alpha)

            values  = numpy.array(list(pr_dict.values()))
            mean_pr = numpy.mean(values)
            std_pr  = numpy.std(values, ddof=0)

            if std_pr == 0:
                zscore_dict = {node: 0.0 for node in pr_dict}
            else:
                zscore_dict = {
                    node: float((v - mean_pr) / std_pr)
                    for node, v in pr_dict.items()
                }

            df["PageRank"]        = df["Node"].astype(str).map(pr_dict)
            df["StdPageRank"] = df["Node"].astype(str).map(zscore_dict)

            unmatched = df["PageRank"].isna().sum()
            if unmatched > 0:
                print(f"  Warning: {unmatched} unmatched node(s) in {csv_path.name}.")

            df.to_csv(csv_path, index=False)
            print(f"  Appended PageRank to {csv_path.name}")
            n_ok += 1

        except Exception as exc:
            print(f"  Error processing {base_name}: {exc}")
            n_errors += 1

    print(
        f"\nappend_pagerank_to_centrality complete — "
        f"updated: {n_ok}, skipped: {n_skipped}, "
        f"missing CSV: {n_missing}, errors: {n_errors}"
    )


# ---------------------------------------------------------------------------
# RCA helpers -- del Sol et al. (2006), Mol. Syst. Biol.
# ---------------------------------------------------------------------------
def _characteristic_path_length(g, method="lcc"):
    """
    Compute the characteristic path length (CPL) of a graph.

    Parameters
    ----------
    g      : NetworkX graph (may be disconnected after node removal).
    method : "lcc"      -> CPL of the largest connected component only.
             "harmonic" -> harmonic-mean CPL; unreachable pairs contribute 0.
                          Recommended for sparse non-covalent interaction graphs.

    Returns
    -------
    CPL as a float. Returns 0.0 when fewer than 2 nodes remain.
    """
    if g.number_of_nodes() < 2:
        return 0.0

    if method == "harmonic":
        n       = g.number_of_nodes()
        n_pairs = n * (n - 1) / 2
        if n_pairs == 0:
            return 0.0
        total = 0.0
        for source in g.nodes():
            lengths = nx.single_source_shortest_path_length(g, source)
            for target, d in lengths.items():
                if target != source and d > 0:
                    total += 1.0 / d
        return total / (2 * n_pairs)

    else:  # method == "lcc"
        if not nx.is_connected(g):
            lcc_nodes = max(nx.connected_components(g), key=len)
            g = g.subgraph(lcc_nodes).copy()
            if g.number_of_nodes() < 2:
                return 0.0
        return nx.average_shortest_path_length(g)


def _compute_rca(graph, cpl_method="lcc"):
    """
    Compute Residue Centrality Analysis (RCA) and its z-scores.

    For each node k:
        delta_L_k = |L - L_rem_k|
    where L is the CPL of the full graph and L_rem_k is the CPL after
    removing node k and all its edges.

    Z-score (del Sol et al. 2006, Eq. 2):
        z_k = (delta_L_k - mean(delta_L)) / std(delta_L)

    Residues with z_k >= 2.0 are classified as CICD residues.

    Parameters
    ----------
    graph      : NetworkX graph representing the protein RIN.
    cpl_method : "lcc" or "harmonic" -- passed to _characteristic_path_length.

    Returns
    -------
    (delta_L_dict, zscore_dict) -- both keyed by node ID.
    """
    L = _characteristic_path_length(graph, method=cpl_method)

    delta_L = {}
    for node in graph.nodes():
        g_removed     = graph.copy()
        g_removed.remove_node(node)
        L_rem         = _characteristic_path_length(g_removed, method=cpl_method)
        delta_L[node] = abs(L - L_rem)

    values  = numpy.array(list(delta_L.values()))
    mean_dL = numpy.mean(values)
    std_dL  = numpy.std(values, ddof=0)

    if std_dL == 0:
        zscore_dict = {node: 0.0 for node in delta_L}
    else:
        zscore_dict = {
            node: float((delta_L[node] - mean_dL) / std_dL)
            for node in delta_L
        }

    return delta_L, zscore_dict


def _worker_rca(args: tuple) -> tuple[str, str]:
    """
    Top-level worker for parallel RCA computation.

    args: (graphml_path_str, csv_path_str, cpl_method_str, overwrite_bool)

    Returns
    -------
    (base_name, outcome_str) where outcome_str is one of:
        "skipped"       -- RCA_deltaL column already present and overwrite=False.
        "ok"            -- columns appended/overwritten successfully.
        "no_csv"        -- no _centrality.csv found for this protein.
        "Error: <msg>"  -- exception during computation or CSV write.
    """
    graphml_path_str, csv_path_str, cpl_method, overwrite = args
    graphml_path = Path(graphml_path_str)
    csv_path     = Path(csv_path_str)
    base_name    = graphml_path.stem.split("_")[0]

    if not csv_path.exists():
        return base_name, "no_csv"

    try:
        df = pd.read_csv(csv_path)

        if "RCA_deltaL" in df.columns and not overwrite:
            return base_name, "skipped"

        graph = nx.read_graphml(graphml_path)
        delta_L_dict, zscore_dict = _compute_rca(graph, cpl_method=cpl_method)

        df["RCA_deltaL"] = df["Node"].astype(str).map(delta_L_dict)
        df["RCA_zscore"] = df["Node"].astype(str).map(zscore_dict)
        df["IsCICD"]     = df["RCA_zscore"] >= 2.0

        df.to_csv(csv_path, index=False)
        return base_name, "ok"

    except Exception as exc:
        return base_name, f"Error: {exc}"


def _run_parallel_rca(work_args, num_processes, dataset_label):
    """
    Parallel runner for RCA append tasks.

    Each item in work_args is a tuple:
        (graphml_path_str, csv_path_str, cpl_method_str, overwrite_bool)
    """
    if not work_args:
        print(f"[{dataset_label}] Nothing to process.")
        return

    print(f"[{dataset_label}] Submitting {len(work_args)} proteins to "
          f"{num_processes} worker(s).")

    n_ok = n_skipped = n_no_csv = n_errors = 0
    results = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            for base_name, outcome in tqdm(
                pool.imap_unordered(_worker_rca, work_args),
                total=len(work_args),
                desc=dataset_label,
            ):
                results.append((base_name, outcome))
                if outcome == "ok":
                    n_ok += 1
                elif outcome == "skipped":
                    n_skipped += 1
                elif outcome == "no_csv":
                    print(f"  NO CSV   {base_name}")
                    n_no_csv += 1
                elif outcome.startswith("Error"):
                    print(f"  FAILED   {base_name}: {outcome}")
                    n_errors += 1

    except KeyboardInterrupt:
        print(f"\n[{dataset_label}] KeyboardInterrupt -- pool shut down. "
              f"Completed {len(results)} protein(s) this session. Re-run to resume.")
        return

    print(
        f"\n[{dataset_label}] append_rca_to_centrality complete -- "
        f"updated: {n_ok}, skipped: {n_skipped}, "
        f"no CSV: {n_no_csv}, errors: {n_errors}"
    )


def append_rca_to_centrality(
    graphml_dir,
    centrality_dir,
    cpl_method="lcc",
    num_processes=4,
    overwrite=False,
):
    """
    Appends RCA_deltaL, RCA_zscore, and IsCICD columns to each existing
    _centrality.csv using the corresponding GraphML file in graphml_dir.

    RCA (Residue Centrality Analysis) is computed per del Sol et al. (2006):
        delta_L_k = |L - L_rem_k|
    where L is the characteristic path length of the full graph and L_rem_k
    is the CPL after removing node k and all its edges. Z-scores are computed
    over all residues in each protein; residues with z >= 2.0 are flagged as
    CICD (Centrally Interconnected Conserved Determinant) residues.

    Runs in parallel across proteins. By default skips any CSV that already
    has an RCA_deltaL column -- set overwrite=True to recompute all proteins
    (e.g. when switching between cpl_method="lcc" and cpl_method="harmonic").

    Parameters
    ----------
    graphml_dir    : path-like -- directory containing *.graphml files.
    centrality_dir : path-like -- directory containing *_centrality.csv files.
    cpl_method     : "lcc" (default) or "harmonic".
    num_processes  : int -- number of parallel worker processes (default 4).
    overwrite      : bool -- if True, recomputes and overwrites RCA columns.
    """
    graphml_dir    = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)

    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return

    graphml_files = sorted(graphml_dir.glob("*.graphml"))
    if not graphml_files:
        print(f"No GraphML files found in {graphml_dir}.")
        return

    work_args = []
    for graphml_file in graphml_files:
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"
        work_args.append((str(graphml_file), str(csv_path), cpl_method, overwrite))

    dataset_label = centrality_dir.parent.name
    _run_parallel_rca(work_args, num_processes, dataset_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    num_processes=4,

    # Test set (BioLiP / single-chain enzymes)
    test_graphs_dir=DATA_DIR_3 / "test" / "RINs",
    test_centrality_dir=DATA_DIR_3 / "test" / "Centralities",

    # Reference set
    ref_graphs_dir=DATA_DIR_3 / "ref" / "RINs",
    ref_centrality_dir=DATA_DIR_3 / "ref" / "Centralities",

    # RCA parameters (shared across both sets)
    rca_cpl_method="harmonic",
    rca_overwrite=False,

    # PageRank parameters (shared across both sets)
    pagerank_alpha=0.85,
    pagerank_overwrite=False,
):
    """
    Run the full centrality pipeline for both the test and reference sets.

    For each dataset the pipeline runs in order:
      1. Resume check  -- classifies each protein as complete / incomplete / absent.
                          Incomplete CSVs are deleted so step 2 rewrites them fresh.
      2. Parallel pass -- degree + closeness centrality written to _centrality.csv.
      3. Degree        -- raw integer degree appended sequentially.
      4. Betweenness   -- betweenness centrality appended sequentially.
      5. RCA           -- residue centrality (del Sol 2006) appended in parallel.
      6. PageRank      -- PageRank + z-score appended sequentially.

    Steps 3-6 each have their own per-column skip guard, so they naturally
    handle proteins that were reprocessed from scratch in step 2.

    Parameters
    ----------
    num_processes       : int       -- parallel workers for steps 2 and 5.
    test_graphs_dir     : path-like -- GraphML directory for the test set.
    test_centrality_dir : path-like -- output directory for test centrality CSVs.
    ref_graphs_dir      : path-like -- GraphML directory for the reference set.
    ref_centrality_dir  : path-like -- output directory for reference centrality CSVs.
    rca_cpl_method      : str       -- "lcc" or "harmonic" for RCA computation.
    rca_overwrite       : bool      -- if True, recomputes RCA even if columns exist.
    pagerank_alpha      : float     -- PageRank damping factor (default 0.85).
    pagerank_overwrite  : bool      -- if True, recomputes PageRank even if columns exist.
    """
    log_dir = DATA_DIR_3 / "logs"
    logger  = setup_logger(log_dir, "rin_analysis")

    logger.info("=" * 60)
    logger.info("RIN ANALYSIS SESSION STARTED")

    # Remove any stale .tmp files left by a previous interrupted run
    for search_dir in [test_centrality_dir, ref_centrality_dir]:
        search_dir = Path(search_dir)
        if search_dir.exists():
            stale = list(search_dir.glob("*.tmp"))
            if stale:
                logger.info(f"Removing {len(stale)} stale .tmp file(s) in {search_dir}")
                for f in stale:
                    f.unlink()

    # ------------------------------------------------------------------
    # 1. Test set (BioLiP / single-chain enzymes)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Test set: BioLiP / single-chain enzymes")

    test_graphmls = sorted(Path(test_graphs_dir).glob("*.graphml"))
    logger.info(f"GraphML files found: {len(test_graphmls)}")

    test_to_process, _ = _check_resume(
        test_graphmls, test_centrality_dir, logger, "Test-enzymes"
    )

    test_args = [
        (str(f), str(test_centrality_dir))
        for f in test_to_process
    ]
    run_parallel(test_args, logger, num_processes, "Test-enzymes")

    logger.info("Appending Degree to test centrality CSVs...")
    append_degree_to_centrality(test_graphs_dir, test_centrality_dir)

    logger.info("Appending BetweennessCentrality to test centrality CSVs...")
    append_betweenness_to_centrality(test_graphs_dir, test_centrality_dir)

    logger.info("Appending RCA to test centrality CSVs...")
    append_rca_to_centrality(
        graphml_dir    = test_graphs_dir,
        centrality_dir = test_centrality_dir,
        cpl_method     = rca_cpl_method,
        num_processes  = num_processes,
        overwrite      = rca_overwrite,
    )

    logger.info("Appending PageRank to test centrality CSVs...")
    append_pagerank_to_centrality(
        graphml_dir    = test_graphs_dir,
        centrality_dir = test_centrality_dir,
        alpha          = pagerank_alpha,
        overwrite      = pagerank_overwrite,
    )

    # ------------------------------------------------------------------
    # 2. Reference set
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Reference set")

    ref_graphmls = sorted(Path(ref_graphs_dir).glob("*.graphml"))
    logger.info(f"GraphML files found: {len(ref_graphmls)}")

    ref_to_process, _ = _check_resume(
        ref_graphmls, ref_centrality_dir, logger, "Reference"
    )

    ref_args = [
        (str(f), str(ref_centrality_dir))
        for f in ref_to_process
    ]
    run_parallel(ref_args, logger, num_processes, "Reference")

    logger.info("Appending Degree to reference centrality CSVs...")
    append_degree_to_centrality(ref_graphs_dir, ref_centrality_dir)

    logger.info("Appending BetweennessCentrality to reference centrality CSVs...")
    append_betweenness_to_centrality(ref_graphs_dir, ref_centrality_dir)

    logger.info("Appending RCA to reference centrality CSVs...")
    append_rca_to_centrality(
        graphml_dir    = ref_graphs_dir,
        centrality_dir = ref_centrality_dir,
        cpl_method     = rca_cpl_method,
        num_processes  = num_processes,
        overwrite      = rca_overwrite,
    )

    logger.info("Appending PageRank to reference centrality CSVs...")
    append_pagerank_to_centrality(
        graphml_dir    = ref_graphs_dir,
        centrality_dir = ref_centrality_dir,
        alpha          = pagerank_alpha,
        overwrite      = pagerank_overwrite,
    )

    logger.info("RIN ANALYSIS SESSION DONE")
    logger.info("=" * 60)


if __name__ == "__main__":

    main(
        num_processes       = 4,

        # Test set paths
        test_graphs_dir     = DATA_DIR_3 / "test" / "RINs",
        test_centrality_dir = DATA_DIR_3 / "test" / "Centralities",

        # Reference set paths
        ref_graphs_dir      = DATA_DIR_3 / "ref" / "RINs",
        ref_centrality_dir  = DATA_DIR_3 / "ref" / "Centralities",

        # RCA settings
        rca_cpl_method      = "harmonic",
        rca_overwrite       = False,

        # PageRank settings
        pagerank_alpha      = 0.85,
        pagerank_overwrite  = False,
    )

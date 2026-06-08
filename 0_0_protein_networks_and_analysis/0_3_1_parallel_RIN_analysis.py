# The purpose of this file is to take the RIGs and perform some network analysis on them.
# The network analysis includes:
# - counts + degree and closeness centrality measures for the residues within each protein
# - taking these measurements to then use for comparison to the Acr protein rigs themselves
# In this case, we're going from GraphML files -> CSV files with our data of interest
#
# Parallelized version: each protein (graphml -> centrality + SASA) is processed by a
# single worker so the pool handles one protein end-to-end. This avoids passing large
# graph objects across process boundaries.
#
# Resume behaviour:
#   - BioLiP/enzyme set: skips proteins whose centrality AND sasa CSVs already exist.
#   - PISCES set:        processed fresh (no prior run), so no skip logic applied.
#   - append_degree_to_centrality() runs sequentially after each pool finishes;
#     it is fast and already has its own per-file skip guard.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# ---------------------------------------------------------------------------
# SASA reference tables
# Tien MZ et al. (2013) PLoS ONE 8(11): e80635. doi:10.1371/journal.pone.0080635
# ---------------------------------------------------------------------------
max_sasa_theo_dict = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}

max_sasa_emp_dict = {
    "ALA": 121, "ARG": 265, "ASN": 187, "ASP": 187, "CYS": 148,
    "GLN": 214, "GLU": 214, "GLY":  97, "HIS": 216, "ILE": 195,
    "LEU": 191, "LYS": 230, "MET": 203, "PHE": 228, "PRO": 154,
    "SER": 143, "THR": 163, "TRP": 264, "TYR": 255, "VAL": 165,
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
# Core single-protein functions (called inside worker processes)
# ---------------------------------------------------------------------------
def _compute_centrality(graph, base_name, centrality_dir):
    """
    Compute degree + closeness centrality and write to CSV using a
    write-to-temp-then-rename pattern to prevent corrupt outputs on crash.
    Returns "skipped", "ok", or "Error: <msg>".
    """
    centrality_dir = Path(centrality_dir)
    centrality_dir.mkdir(parents=True, exist_ok=True)

    out_path = centrality_dir / f"{base_name}_centrality.csv"
    tmp_path = centrality_dir / f"{base_name}_centrality.csv.tmp"

    if out_path.exists():
        return "skipped"

    try:
        degree_centrality   = nx.degree_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)

        combined_data = []
        for node, attrs in graph.nodes(data=True):
            combined_data.append({
                "Node":               node,
                "Residue":            attrs["residue_type"],
                "Chain":              attrs["chain"],
                "Residue number":     attrs["residue_num"],
                "DegreeCentrality":   degree_centrality.get(node, 0),
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


def _compute_sasa_cif(base_name, cif_dir, sasa_dir):
    """
    Compute SASA/RSA from a mmCIF file and write to CSV.
    Returns "skipped", "ok", "no_file", or "Error: <msg>".
    """
    cif_dir  = Path(cif_dir)
    sasa_dir = Path(sasa_dir)
    sasa_dir.mkdir(parents=True, exist_ok=True)

    out_path = sasa_dir / f"{base_name}_sasa.csv"
    tmp_path = sasa_dir / f"{base_name}_sasa.csv.tmp"

    if out_path.exists():
        return "skipped"

    # locate the CIF file
    cif_file = next(
        (cif_dir / f for f in os.listdir(cif_dir)
         if f.startswith(base_name) and f.endswith(".cif")),
        None,
    )
    if cif_file is None:
        return "no_file"

    try:
        p      = MMCIFParser(QUIET=True)
        struct = p.get_structure(base_name, cif_file)
        sr     = ShrakeRupley()
        sr.compute(struct, level="R")

        sasa_data = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if "CA" not in residue:
                        continue
                    res_name  = residue.resname
                    sasa_val  = round(residue.sasa, 2)
                    theo_max  = max_sasa_theo_dict.get(res_name)
                    emp_max   = max_sasa_emp_dict.get(res_name)
                    rsa_theo  = (sasa_val / theo_max) if theo_max else -1
                    rsa_emp   = (sasa_val / emp_max)  if emp_max  else -1
                    sasa_data.append({
                        "Model":           model.id,
                        "Chain":           chain.id,
                        "Residue":         res_name,
                        "Residue ID":      residue.id[1],
                        "SASA (Å²)":       sasa_val,
                        "RSA-theoretical": rsa_theo,
                        "RSA-empirical":   rsa_emp,
                    })

        pd.DataFrame(sasa_data).to_csv(tmp_path, index=False)
        tmp_path.rename(out_path)
        return "ok"

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"Error: {exc}"


def _compute_sasa_pdb(base_name, pdb_dir, sasa_dir):
    """
    Compute SASA/RSA from a PDB file and write to CSV.
    Identical logic to _compute_sasa_cif — only the parser and extension differ.
    Returns "skipped", "ok", "no_file", or "Error: <msg>".
    """
    pdb_dir  = Path(pdb_dir)
    sasa_dir = Path(sasa_dir)
    sasa_dir.mkdir(parents=True, exist_ok=True)

    out_path = sasa_dir / f"{base_name}_sasa.csv"
    tmp_path = sasa_dir / f"{base_name}_sasa.csv.tmp"

    if out_path.exists():
        return "skipped"

    # locate the PDB file
    pdb_file = next(
        (pdb_dir / f for f in os.listdir(pdb_dir)
         if f.startswith(base_name) and f.endswith(".pdb")),
        None,
    )
    if pdb_file is None:
        return "no_file"

    try:
        p      = PDBParser(QUIET=True)
        struct = p.get_structure(base_name, pdb_file)
        sr     = ShrakeRupley()
        sr.compute(struct, level="R")

        sasa_data = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if "CA" not in residue:
                        continue
                    res_name  = residue.resname
                    sasa_val  = round(residue.sasa, 2)
                    theo_max  = max_sasa_theo_dict.get(res_name)
                    emp_max   = max_sasa_emp_dict.get(res_name)
                    rsa_theo  = (sasa_val / theo_max) if theo_max else -1
                    rsa_emp   = (sasa_val / emp_max)  if emp_max  else -1
                    sasa_data.append({
                        "Model":           model.id,
                        "Chain":           chain.id,
                        "Residue":         res_name,
                        "Residue ID":      residue.id[1],
                        "SASA (Å²)":       sasa_val,
                        "RSA-theoretical": rsa_theo,
                        "RSA-empirical":   rsa_emp,
                    })

        pd.DataFrame(sasa_data).to_csv(tmp_path, index=False)
        tmp_path.rename(out_path)
        return "ok"

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Worker functions (top-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------
def worker_enzyme(args):
    """
    Worker for the BioLiP/enzyme set (CIF files).
    args: (graphml_path_str, cif_dir_str, centrality_dir_str, sasa_dir_str)
    Returns (stem, centrality_result, sasa_result).
    """
    graphml_path, cif_dir, centrality_dir, sasa_dir = args
    graphml_path = Path(graphml_path)
    stem         = graphml_path.stem
    base_name    = stem.split("_")[0]

    try:
        graph = nx.read_graphml(graphml_path)
    except Exception as exc:
        return base_name, f"Error loading graph: {exc}", "skipped (graph load failed)"

    cent_result = _compute_centrality(graph, base_name, centrality_dir)
    sasa_result = _compute_sasa_cif(base_name, cif_dir, sasa_dir)
    return base_name, cent_result, sasa_result


def worker_pisces(args):
    """
    Worker for the old PISCES set (PDB files).
    args: (graphml_path_str, pdb_dir_str, centrality_dir_str, sasa_dir_str)
    Returns (stem, centrality_result, sasa_result).
    """
    graphml_path, pdb_dir, centrality_dir, sasa_dir = args
    graphml_path = Path(graphml_path)
    stem         = graphml_path.stem
    base_name    = stem.split("_")[0]

    try:
        graph = nx.read_graphml(graphml_path)
    except Exception as exc:
        return base_name, f"Error loading graph: {exc}", "skipped (graph load failed)"

    cent_result = _compute_centrality(graph, base_name, centrality_dir)
    sasa_result = _compute_sasa_pdb(base_name, pdb_dir, sasa_dir)
    return base_name, cent_result, sasa_result


# ---------------------------------------------------------------------------
# Sequential post-processing
# ---------------------------------------------------------------------------
def append_degree_to_centrality(graphml_dir, centrality_dir):
    """
    Appends a Degree column to each centrality CSV using node degrees from
    the corresponding GraphML. Runs sequentially — fast, and already has its
    own per-file skip guard.
    """
    graphml_dir    = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)

    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return

    for graphml_file in graphml_dir.glob("*.graphml"):
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"

        if not csv_path.exists():
            print(f"No centrality CSV found for {base_name}, skipping.")
            continue

        df = pd.read_csv(csv_path)

        if "Degree" in df.columns:
            print(f"{csv_path.name} already has a Degree column, skipping.")
            continue

        graph       = nx.read_graphml(graphml_file)
        degree_dict = dict(graph.degree())
        df["Degree"] = df["Node"].astype(str).map(degree_dict)

        unmatched = df["Degree"].isna().sum()
        if unmatched > 0:
            print(f"Warning: {unmatched} unmatched nodes in {csv_path.name}.")

        df.to_csv(csv_path, index=False)
        print(f"Appended Degree column to {csv_path.name}")

def append_betweenness_to_centrality(graphml_dir, centrality_dir):
    """
    Appends a BetweennessCentrality column to each existing _centrality.csv
    using the corresponding GraphML file in graphml_dir.
 
    Betweenness is computed with nx.betweenness_centrality(graph, normalized=True).
    Runs sequentially. Skips any CSV that already has a BetweennessCentrality column.
 
    Parameters
    ----------
    graphml_dir    : path-like   Directory containing *.graphml files.
                                 Typically DATA_DIR_2 / "RINs".
    centrality_dir : path-like   Directory containing *_centrality.csv files.
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
 
    n_ok      = 0
    n_skipped = 0
    n_missing = 0
    n_errors  = 0
 
    for graphml_file in graphml_files:
        base_name = graphml_file.stem.split("_")[0]
        csv_path  = centrality_dir / f"{base_name}_centrality.csv"
 
        if not csv_path.exists():
            print(f"  No centrality CSV found for {base_name}, skipping.")
            n_missing += 1
            continue
 
        df = pd.read_csv(csv_path)
 
        if "BetweennessCentrality" in df.columns:
            print(f"  {csv_path.name} already has BetweennessCentrality, skipping.")
            n_skipped += 1
            continue
 
        try:
            graph = nx.read_graphml(graphml_file)
            betweenness_dict = nx.betweenness_centrality(graph, normalized=True)
 
            df["BetweennessCentrality"] = df["Node"].astype(str).map(betweenness_dict)
 
            unmatched = df["BetweennessCentrality"].isna().sum()
            if unmatched > 0:
                print(f"  Warning: {unmatched} unmatched nodes in {csv_path.name}.")
 
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

# ---------------------------------------------------------------------------
# FASTA helper
# ---------------------------------------------------------------------------
def parse_rep_seq_fasta(fasta_path):
    """
    Returns a set of uppercase 4-character PDB IDs from a representative
    sequence FASTA file (e.g. {"2A11", "4A1G", "6A3Z"}).
    """
    rep_ids = set()
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                rep_ids.add(line[1:5].upper())
    return rep_ids


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
def run_parallel(work_args, worker_fn, logger, num_processes, dataset_label):
    """
    Runs worker_fn over work_args using a Pool, with tqdm progress and clean
    Ctrl+C handling. Logs per-file outcomes and prints a session summary.
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
                pool.imap_unordered(worker_fn, work_args),
                total=len(work_args),
                desc=dataset_label,
            ):
                results.append(result)
                base_name, cent, sasa = result

                # log errors at WARNING so they surface on the console
                for label, outcome in [("centrality", cent), ("sasa", sasa)]:
                    if str(outcome).startswith("Error"):
                        logger.warning(f"FAILED {label}  {base_name}: {outcome}")
                    elif outcome == "no_file":
                        logger.warning(f"NO FILE {label}  {base_name}")
                    else:
                        logger.debug(f"OK {label}  {base_name}: {outcome}")

    except KeyboardInterrupt:
        logger.warning(f"[{dataset_label}] KeyboardInterrupt — pool shut down. "
                       f"Completed {len(results)} protein(s) this session. Re-run to resume.")
        return

    # session summary
    errors = [(b, c, s) for b, c, s in results
              if str(c).startswith("Error") or str(s).startswith("Error")
              or c == "no_file" or s == "no_file"]
    logger.info(f"[{dataset_label}] Done — {len(results) - len(errors)}/{len(results)} "
                f"fully successful.")
    if errors:
        logger.warning(f"[{dataset_label}] {len(errors)} protein(s) with issues:")
        for b, c, s in errors:
            logger.warning(f"  {b}  centrality={c}  sasa={s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- configurable parameters ----
    num_processes = 4   # number of parallel worker processes

    # BioLiP / enzyme set paths
    enzyme_graphs_dir    = DATA_DIR_2 / "RINs"
    enzyme_cif_dir       = DATA_DIR_2 / "enzymes_CIF_unflagged" / "CIFs"
    enzyme_centrality_dir = DATA_DIR_2 / "Centralities"
    enzyme_sasa_dir      = DATA_DIR_2 / "SASAs"
    enzyme_fasta_path    = ENZYMES_DIR / "mmseq" / "runs" / "050626_enzymes_nr30_rep_seq.fasta"

    # old PISCES set paths
    pisces_graphs_dir    = DATA_DIR_2 / "old_PISCES" / "GraphMLs_filtered_v2"
    pisces_pdb_dir       = DATA_DIR_2 / "old_PISCES" / "PDBs"
    pisces_centrality_dir = DATA_DIR_2 / "old_PISCES" / "Centralities"
    pisces_sasa_dir      = DATA_DIR_2 / "old_PISCES" / "SASAs"
    # ---- end configurable parameters ----

    log_dir = DATA_DIR_2 / "logs"
    logger  = setup_logger(log_dir, "rin_analysis")

    logger.info("=" * 60)
    logger.info("RIN ANALYSIS SESSION STARTED")

    # remove any stale .tmp files left by a previous interrupted run
    for search_dir in [enzyme_centrality_dir, enzyme_sasa_dir,
                       pisces_centrality_dir, pisces_sasa_dir]:
        search_dir = Path(search_dir)
        if search_dir.exists():
            stale = list(search_dir.glob("*.tmp"))
            if stale:
                logger.info(f"Removing {len(stale)} stale .tmp file(s) in {search_dir}")
                for f in stale:
                    f.unlink()

    # ------------------------------------------------------------------
    # 1. BioLiP / enzyme set  (CIF files, FASTA filter, resume support)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("BioLiP / enzyme set")

    rep_ids = parse_rep_seq_fasta(enzyme_fasta_path)
    logger.info(f"Found {len(rep_ids)} representative sequences in FASTA.")

    # collect all graphml files that pass the FASTA filter
    all_enzyme_graphmls = [
        f for f in Path(enzyme_graphs_dir).glob("*.graphml")
        if f.stem.split("_")[0].upper() in rep_ids
    ]
    logger.info(f"GraphML files matching FASTA: {len(all_enzyme_graphmls)}")

    # resume: skip proteins where BOTH outputs already exist
    already_done_centrality = {f.stem.replace("_centrality", "")
                               for f in Path(enzyme_centrality_dir).glob("*_centrality.csv")} \
                              if Path(enzyme_centrality_dir).exists() else set()
    already_done_sasa       = {f.stem.replace("_sasa", "")
                               for f in Path(enzyme_sasa_dir).glob("*_sasa.csv")} \
                              if Path(enzyme_sasa_dir).exists() else set()
    already_complete        = already_done_centrality & already_done_sasa

    enzyme_to_process = [
        f for f in all_enzyme_graphmls
        if f.stem.split("_")[0].upper() not in already_complete
           and f.stem.split("_")[0] not in already_complete
    ]
    logger.info(f"Already complete: {len(already_complete)} | "
                f"To process: {len(enzyme_to_process)}")

    enzyme_args = [
        (str(f), str(enzyme_cif_dir), str(enzyme_centrality_dir), str(enzyme_sasa_dir))
        for f in enzyme_to_process
    ]
    # run_parallel(enzyme_args, worker_enzyme, logger, num_processes, "BioLiP-enzymes")

    logger.info("Appending degree values to enzyme centrality CSVs...")
    append_degree_to_centrality(enzyme_graphs_dir, enzyme_centrality_dir)

    # ------------------------------------------------------------------
    # 2. Old PISCES set  (PDB files, no FASTA filter, no prior progress)
    # ------------------------------------------------------------------
    logger.info("-" * 40)
    logger.info("Old PISCES set")

    pisces_graphmls = sorted(Path(pisces_graphs_dir).glob("*.graphml"))
    logger.info(f"GraphML files found: {len(pisces_graphmls)}")

    pisces_args = [
        (str(f), str(pisces_pdb_dir), str(pisces_centrality_dir), str(pisces_sasa_dir))
        for f in pisces_graphmls
    ]
    run_parallel(pisces_args, worker_pisces, logger, num_processes, "PISCES")

    logger.info("Appending degree values to PISCES centrality CSVs...")
    append_degree_to_centrality(pisces_graphs_dir, pisces_centrality_dir)

    logger.info("RIN ANALYSIS SESSION DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # main()

    append_betweenness_to_centrality(
        graphml_dir    = DATA_DIR_2 / "RINs",
        centrality_dir = DATA_DIR_2 / "Centralities",
    )

    # For the PISCES reference set, call it again with the PISCES paths:
    append_betweenness_to_centrality(
        graphml_dir    = DATA_DIR_2 / "RINs",
        centrality_dir = DATA_DIR_2 / "old_PISCES" / "Centralities",
    )

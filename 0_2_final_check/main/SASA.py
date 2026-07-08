# SASA.py
#
# Computes per-residue SASA and RSA for a set of proteins.
#
# Inputs:
#   - names_dir   : directory whose filenames are parsed for PDB IDs
#   - struct_dir  : directory containing the corresponding .pdb or .cif files
#   - sasa_dir    : output directory for per-protein *_sasa.csv files
#
# Output columns per CSV:
#   Model, Chain, Residue, Residue ID, SASA (Å²), RSA-theoretical, RSA-empirical
#
# Resume behaviour: proteins whose *_sasa.csv already exists are skipped.
# Stale *.tmp files from a previous interrupted run are removed on startup.

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
# Core SASA computation (called inside worker processes)
# ---------------------------------------------------------------------------
def _compute_sasa_cif(base_name, cif_dir, sasa_dir):
    """
    Compute SASA/RSA from a mmCIF file and write to CSV.
    Uses a write-to-temp-then-rename pattern to prevent corrupt outputs on crash.
    Returns "skipped", "ok", "no_file", or "Error: <msg>".
    """
    cif_dir  = Path(cif_dir)
    sasa_dir = Path(sasa_dir)
    sasa_dir.mkdir(parents=True, exist_ok=True)

    out_path = sasa_dir / f"{base_name}_sasa.csv"
    tmp_path = sasa_dir / f"{base_name}_sasa.csv.tmp"

    if out_path.exists():
        return "skipped"

    cif_file = next(
        (cif_dir / f for f in os.listdir(cif_dir)
         if f.lower().startswith(base_name.lower()) and f.endswith(".cif")),
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
                    res_name = residue.resname
                    sasa_val = round(residue.sasa, 2)
                    theo_max = max_sasa_theo_dict.get(res_name)
                    emp_max  = max_sasa_emp_dict.get(res_name)
                    rsa_theo = (sasa_val / theo_max) if theo_max else -1
                    rsa_emp  = (sasa_val / emp_max)  if emp_max  else -1
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

    pdb_file = next(
        (pdb_dir / f for f in os.listdir(pdb_dir)
         if f.lower().startswith(base_name.lower()) and f.endswith(".pdb")),
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
                    res_name = residue.resname
                    sasa_val = round(residue.sasa, 2)
                    theo_max = max_sasa_theo_dict.get(res_name)
                    emp_max  = max_sasa_emp_dict.get(res_name)
                    rsa_theo = (sasa_val / theo_max) if theo_max else -1
                    rsa_emp  = (sasa_val / emp_max)  if emp_max  else -1
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
def _worker_cif(args):
    """
    Worker for CIF-backed structure directories.
    args: (base_name_str, struct_dir_str, sasa_dir_str)
    Returns (base_name, sasa_result).
    """
    base_name, struct_dir, sasa_dir = args
    result = _compute_sasa_cif(base_name, struct_dir, sasa_dir)
    return base_name, result


def _worker_pdb(args):
    """
    Worker for PDB-backed structure directories.
    args: (base_name_str, struct_dir_str, sasa_dir_str)
    Returns (base_name, sasa_result).
    """
    base_name, struct_dir, sasa_dir = args
    result = _compute_sasa_pdb(base_name, struct_dir, sasa_dir)
    return base_name, result


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
def run_parallel_sasa(work_args, worker_fn, logger, num_processes, dataset_label):
    """
    Runs worker_fn over work_args using a multiprocessing Pool, with tqdm
    progress and clean Ctrl+C handling. Logs per-file outcomes and prints a
    session summary.
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
                base_name, sasa = result
                if str(sasa).startswith("Error"):
                    logger.warning(f"FAILED  {base_name}: {sasa}")
                elif sasa == "no_file":
                    logger.warning(f"NO FILE {base_name}")
                else:
                    logger.debug(f"OK      {base_name}: {sasa}")

    except KeyboardInterrupt:
        logger.warning(
            f"[{dataset_label}] KeyboardInterrupt — pool shut down. "
            f"Completed {len(results)} protein(s) this session. Re-run to resume."
        )
        return

    issues = [(b, s) for b, s in results
              if str(s).startswith("Error") or s == "no_file"]
    logger.info(
        f"[{dataset_label}] Done — "
        f"{len(results) - len(issues)}/{len(results)} fully successful."
    )
    if issues:
        logger.warning(f"[{dataset_label}] {len(issues)} protein(s) with issues:")
        for b, s in issues:
            logger.warning(f"  {b}  sasa={s}")


# ---------------------------------------------------------------------------
# PDB ID parsing helper
# ---------------------------------------------------------------------------
def parse_pdb_ids_from_dir(names_dir: Path) -> list[str]:
    """
    Returns a sorted list of unique PDB IDs (first 4 characters of each
    filename, uppercased) found in names_dir. Subdirectories are ignored.
    """
    ids = set()
    for f in names_dir.iterdir():
        if f.is_file():
            stem = f.stem  # filename without extension
            if len(stem) >= 4:
                ids.add(stem[:4].upper())
    return sorted(ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    names_dir,
    struct_dir,
    sasa_dir,
    struct_fmt,
    num_processes,
    log_dir,
    dataset_label,
):
    """
    Parameters
    ----------
    names_dir     : Path — directory whose filenames supply the PDB IDs to process.
    struct_dir    : Path — directory containing the PDB or CIF structure files.
    sasa_dir      : Path — directory where *_sasa.csv outputs are written.
    struct_fmt    : str  — "pdb" or "cif" (controls which parser is used).
    num_processes : int  — number of parallel worker processes.
    log_dir       : Path — directory for the .log file.
    dataset_label : str  — label used in log messages and the tqdm progress bar.
    """
    names_dir  = Path(names_dir)
    struct_dir = Path(struct_dir)
    sasa_dir   = Path(sasa_dir)
    log_dir    = Path(log_dir)

    logger = setup_logger(log_dir, "sasa_computation")
    logger.info("=" * 60)
    logger.info("SASA COMPUTATION SESSION STARTED")
    logger.info(f"  names_dir  : {names_dir}")
    logger.info(f"  struct_dir : {struct_dir}")
    logger.info(f"  sasa_dir   : {sasa_dir}")
    logger.info(f"  struct_fmt : {struct_fmt}")
    logger.info(f"  processes  : {num_processes}")

    # Remove stale .tmp files from any previous interrupted run
    sasa_dir.mkdir(parents=True, exist_ok=True)
    stale = list(sasa_dir.glob("*.tmp"))
    if stale:
        logger.info(f"Removing {len(stale)} stale .tmp file(s) in {sasa_dir}")
        for f in stale:
            f.unlink()

    # Parse PDB IDs from the filenames directory
    pdb_ids = parse_pdb_ids_from_dir(names_dir)
    logger.info(f"PDB IDs parsed from names_dir: {len(pdb_ids)}")

    # Resume: skip proteins whose SASA CSV already exists
    already_done = {
        f.stem.replace("_sasa", "").upper()
        for f in sasa_dir.glob("*_sasa.csv")
    }
    to_process = [pid for pid in pdb_ids if pid not in already_done]
    logger.info(f"Already complete: {len(already_done)} | To process: {len(to_process)}")

    if not to_process:
        logger.info("All proteins already processed. Nothing to do.")
        logger.info("=" * 60)
        return

    # Build work args and select worker
    worker_fn = _worker_cif if struct_fmt.lower() == "cif" else _worker_pdb
    work_args = [
        (pid, str(struct_dir), str(sasa_dir))
        for pid in to_process
    ]

    run_parallel_sasa(work_args, worker_fn, logger, num_processes, dataset_label)

    logger.info("SASA COMPUTATION SESSION DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Configure all run parameters here
    # -----------------------------------------------------------------------
    names_dir     = DATA_DIR_3 / "ref" / "RINs"                 # directory whose filenames supply PDB IDs
    struct_dir    = DATA_DIR_2 / "old_PISCES" / "PDBs"  # directory containing PDB or CIF files
    sasa_dir      = DATA_DIR_3 / "ref" / "SASAs"                 # output directory for *_sasa.csv files
    struct_fmt    = "pdb"                                 # "cif" or "pdb"
    num_processes = 4                                     # parallel worker count
    log_dir       = DATA_DIR_3 / "logs"                  # log file directory
    dataset_label = "reference_proteins"                  # label for progress bar and logs

    main(
        names_dir     = names_dir,
        struct_dir    = struct_dir,
        sasa_dir      = sasa_dir,
        struct_fmt    = struct_fmt,
        num_processes = num_processes,
        log_dir       = log_dir,
        dataset_label = dataset_label,
    )

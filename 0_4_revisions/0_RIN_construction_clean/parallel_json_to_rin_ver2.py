import sys
import json
import hashlib
import inspect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# Single source of truth for the RIN-building logic. The function definitions
# and STANDARD_AA live ONLY in arpeggio_to_rin_ver2.py -- this script no longer
# carries its own copy, so every dataset built here uses identical logic.
from arpeggio_to_rin_ver2 import (
    arpeggio_to_rin,
    arpeggio_to_rin_hbond_exp,
    arpeggio_to_rin_weak_hbond_exp,
    STANDARD_AA,
)

# name -> builder function. The name is used for the output folder
# (RINs_<name>). Builders must be module-level functions so multiprocessing
# can pickle them.
EXPERIMENTS = {
    "baseline":   arpeggio_to_rin,
    "hbond":      arpeggio_to_rin_hbond_exp,       # 'weak_hbond' removed
    "weak_hbond": arpeggio_to_rin_weak_hbond_exp,  # 'hbond' removed
}


def setup_logger(out_dir: Path) -> logging.Logger:
    """
    Creates a logger writing to both console and <out_dir>/rin_build.log.
    Appends across runs so the full history of every session is preserved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rin_builder")
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handlers if called more than once in a session.
    # NOTE: the log file is attached on the first call only, so when several
    # out_dirs are built in one session, log lines for later out_dirs still go
    # to the first out_dir's log. Fine for a few runs; swap handlers here if
    # you want one log per out_dir.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(out_dir / "rin_build.log", mode='a')
    fh.setLevel(logging.DEBUG)   # DEBUG and above → file
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)    # INFO and above → console
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_biolip_labels(csv_path: Path) -> dict:
    """
    Load biolip_residue_labels.csv into a lookup dict.

    Key   : (pdb_id: str, chain: str, seq_num: int)
              pdb_id is always lowercase to match the CSV.
    Value : label str — one of "binding", "catalytic", or "dual"
              "dual" is assigned when the same residue appears with both
              "binding" and "catalytic" labels in the CSV.

    Residues absent from this dict should be treated as "none" at call sites.
    """
    df = pd.read_csv(csv_path)

    # Normalise types up front
    df['pdb_id']  = df['pdb_id'].str.lower().str.strip()
    df['chain']   = df['chain'].str.strip()
    df['seq_num'] = df['seq_num'].astype(int)
    df['label']   = df['label'].str.strip().str.lower()

    label_lookup = {}
    for _, row in df.iterrows():
        key      = (row['pdb_id'], row['chain'], row['seq_num'])
        new_lbl  = row['label']
        existing = label_lookup.get(key)

        if existing is None:
            label_lookup[key] = new_lbl
        elif existing != new_lbl:
            # residue appears with two different labels → dual
            label_lookup[key] = "dual"
        # if existing == new_lbl, no change needed

    return label_lookup


def logic_fingerprint(rin_builder) -> dict:
    """
    Identify exactly which logic built a dataset: the builder's name plus a
    SHA-256 of the arpeggio_to_rin_ver2.py source file. Any edit to that file
    (hierarchy, excluded types, filters) changes the hash.
    """
    src = Path(inspect.getsourcefile(arpeggio_to_rin)).read_bytes()
    return {
        "builder":      rin_builder.__name__,
        "logic_sha256": hashlib.sha256(src).hexdigest(),
    }


def check_or_write_settings(out_dir: Path, rin_builder) -> None:
    """
    Guard against mixing logic inside one output folder.

    First run in a folder : writes <out_dir>/rin_settings.json.
    Later runs (resume)   : the stored fingerprint must match the current
                            builder and source file, otherwise raise. A folder
                            that already has .graphml files but no settings
                            file (built by an older version) is also refused.

    This matters because the resume logic skips any stem that already has a
    .graphml file: without this check, a re-run after changing the logic would
    silently keep the old results.
    """
    settings_path = out_dir / "rin_settings.json"
    current       = logic_fingerprint(rin_builder)

    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if saved != current:
            raise RuntimeError(
                f"{out_dir} was built with different RIN logic.\n"
                f"  saved   : {saved}\n"
                f"  current : {current}\n"
                "Use a new out_dir (or delete this one) rather than mixing "
                "networks built with different settings."
            )
    else:
        if any(out_dir.glob("*.graphml")):
            raise RuntimeError(
                f"{out_dir} already contains .graphml files but no "
                "rin_settings.json, so their logic is unknown. Use a fresh "
                "out_dir."
            )
        settings_path.write_text(json.dumps(current, indent=2))


def process_single_json(json_file, out_dir, label_lookup, rin_builder=arpeggio_to_rin):
    """
    Build a RIN with *rin_builder* and write it as GraphML using a
    write-to-temp-then-rename pattern. This means a crash or interrupt
    mid-write never leaves a corrupt .graphml on disk — only complete files
    are ever present.

    PDB ID is extracted from the JSON filename stem:
        e.g. "1AC0_clean.json" → stem "1AC0_clean" → pdb_id "1ac0"

    Returns (stem, node_count, n_total, n_filtered) on success
         or (stem, "Error: ...", 0, 0)              on failure.
    """
    json_file = Path(json_file)
    stem      = json_file.stem                        # e.g. "1AC0_clean"
    pdb_id    = stem.split("_")[0].lower()            # e.g. "1ac0"
    out_path  = Path(out_dir) / f"{stem}.graphml"
    tmp_path  = Path(out_dir) / f"{stem}.graphml.tmp"

    try:
        rin, n_total, n_filtered = rin_builder(json_file, pdb_id, label_lookup)
        nx.write_graphml(rin, tmp_path)
        tmp_path.rename(out_path)   # atomic on most local filesystems
        return stem, len(rin.nodes()), n_total, n_filtered
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return stem, f"Error: {exc}", 0, 0


def parallel_jsons_to_rins(json_dir, out_dir, biolip_csv, num_processes=None,
                           rin_builder=arpeggio_to_rin):
    """
    Convert all Arpeggio JSONs in *json_dir* to GraphML RINs in *out_dir*,
    annotating each node with a 'biolip_label' attribute drawn from
    *biolip_csv*. Networks are built by *rin_builder* (default: the baseline
    arpeggio_to_rin).

    Interrupt behaviour
    -------------------
    Ctrl+C is caught cleanly: worker processes are terminated, the log records
    how many files were completed in the interrupted session, and re-running the
    script resumes automatically from where it left off (already-complete
    .graphml files are skipped; stale .tmp files from the interrupted run are
    deleted first).

    Logging
    -------
    Appends to <out_dir>/rin_build.log across all runs. Each session is clearly
    delimited. Per-file results are logged at DEBUG level (file only) so the
    console stays clean; errors are logged at WARNING on both outputs.
    """
    json_dir = Path(json_dir)
    out_dir  = Path(out_dir)
    logger   = setup_logger(out_dir)

    # refuse to resume into a folder built with different logic
    check_or_write_settings(out_dir, rin_builder)

    logger.info("=" * 60)
    logger.info("RIN BUILD SESSION STARTED")
    logger.info(f"  Builder     : {rin_builder.__name__}")
    logger.info(f"  JSON source : {json_dir}")
    logger.info(f"  Output dir  : {out_dir}")
    logger.info(f"  BioLiP CSV  : {biolip_csv}")

    # Load BioLiP labels once — shared (read-only) across all worker calls
    label_lookup = load_biolip_labels(Path(biolip_csv))
    logger.info(f"  BioLiP entries loaded: {len(label_lookup)}")

    # remove .tmp files left by any previous interrupted run
    stale = list(out_dir.glob("*.graphml.tmp"))
    if stale:
        logger.info(f"Removing {len(stale)} stale .tmp file(s) from a previous interrupted run")
        for f in stale:
            f.unlink()

    all_jsons    = sorted(json_dir.glob("*.json"))
    already_done = {f.stem for f in out_dir.glob("*.graphml")}
    to_process   = [f for f in all_jsons if f.stem not in already_done]

    logger.info(
        f"Total JSONs: {len(all_jsons)} | "
        f"Already done: {len(already_done)} | "
        f"To process: {len(to_process)}"
    )

    if not to_process:
        logger.info("Nothing to do. Exiting.")
        logger.info("=" * 60)
        return

    if num_processes is None:
        num_processes = mp.cpu_count()
    logger.info(f"Using {num_processes} worker process(es)")

    worker  = partial(process_single_json, out_dir=str(out_dir),
                      label_lookup=label_lookup, rin_builder=rin_builder)
    results = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, [str(f) for f in to_process]),
                total=len(to_process),
                desc=f"Building RINs ({rin_builder.__name__})",
            ):
                results.append(result)
                stem, outcome, n_total, n_filtered = result
                if str(outcome).startswith("Error"):
                    logger.warning(f"FAILED  {stem}: {outcome}")
                else:
                    pct = (n_filtered / n_total * 100) if n_total > 0 else 0
                    logger.debug(
                        f"OK      {stem} ({outcome} nodes | "
                        f"{n_filtered}/{n_total} interactions filtered [{pct:.1f}%])"
                    )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — shutting down worker pool")
        logger.warning(
            f"Completed this session before interrupt: {len(results)} file(s). "
            "Re-run to resume."
        )
        logger.info("=" * 60)
        return

    # session summary
    successes = [r for r in results if not str(r[1]).startswith("Error")]
    errors    = [r for r in results if     str(r[1]).startswith("Error")]

    logger.info(f"Session complete — {len(successes)}/{len(to_process)} succeeded")
    if errors:
        logger.warning(f"{len(errors)} error(s) this session:")
        for stem, msg, _, __ in errors:
            logger.warning(f"  {stem}: {msg}")

    logger.info("RIN BUILD SESSION DONE")
    logger.info("=" * 60)


def validate_rins(rin_dir):
    """
    Check all GraphML files in rin_dir to confirm:
      1. Every node residue_type is one of the 20 standard amino acids.
      2. Every node has a valid biolip_label ("binding", "catalytic", "dual", "none").

    Prints a summary and returns a dict of any offending files for inspection.
    """
    rin_dir   = Path(rin_dir)
    graphmls  = sorted(rin_dir.glob("*.graphml"))
    offenders = {}

    valid_labels = {"binding", "catalytic", "dual", "none"}

    print(f"Validating {len(graphmls)} GraphML files in {rin_dir}...")

    label_counts = defaultdict(int)

    for gml in graphmls:
        graph = nx.read_graphml(gml)
        non_standard  = set()
        invalid_labels = set()

        for node, attrs in graph.nodes(data=True):
            res = attrs.get("residue_type", "")
            lbl = attrs.get("biolip_label", "MISSING")

            if res not in STANDARD_AA:
                non_standard.add(res)
            if lbl not in valid_labels:
                invalid_labels.add(lbl)

            label_counts[lbl] += 1

        issues = {}
        if non_standard:
            issues['non_standard_residues'] = sorted(non_standard)
        if invalid_labels:
            issues['invalid_biolip_labels'] = sorted(invalid_labels)
        if issues:
            offenders[gml.name] = issues

    if offenders:
        print(f"\nFound {len(offenders)} file(s) with issues:")
        for fname, issues in sorted(offenders.items()):
            print(f"  {fname}: {issues}")
    else:
        print("All GraphML files pass validation.")

    print("\nBioLiP label distribution across all nodes:")
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl:12s}: {count:>8,}")

    return offenders


def tester(experiment="hbond"):
    # Update paths as needed for one-off testing
    test_file    = DATA_DIR_3 / "test" / "JSONs" / "1ID8_clean.json"
    biolip_csv   = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_residue_labels.csv"
    out_dir      = DATA_DIR_3 / "test" / "RIN_create_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_lookup = load_biolip_labels(Path(biolip_csv))
    rin_builder  = EXPERIMENTS[experiment]

    stem     = Path(test_file).stem
    pdb_id   = stem.split("_")[0].lower()
    tmp_path = out_dir / f"{stem}.graphml.tmp"

    rin, n_total, n_filtered = rin_builder(test_file, pdb_id, label_lookup)
    nx.write_graphml(rin, tmp_path)

    # Spot-check: print label distribution for this one RIN
    label_dist = defaultdict(int)
    for node, attrs in rin.nodes(data=True):
        label_dist[attrs.get("biolip_label", "MISSING")] += 1
    print(f"[{experiment}] Label distribution for {stem}:")
    for lbl, count in sorted(label_dist.items()):
        print(f"  {lbl}: {count}")


def main():

    #### EDIT HERE FOR YOUR LOCATION
    biolip_csv    = DATA_DIR_4 / "biolip_residue_labels.csv"
    num_processes = 4   # set to None to use all available CPUs

    # Which networks to build. Each one gets its own folder, RINs_<name>, so
    # nothing built with different logic is ever mixed together.
    # Add "baseline" to also build the all-contacts network for comparison.
    # experiments = ["hbond", "weak_hbond"]
    experiments = ["hbond"]

    # (json_dir, parent folder for the RIN output folders)
    datasets = {
        # Enzyme set — residues will receive BioLiP labels where available
        "test": (DATA_DIR_3 / "test" / "JSONs",
                 ARP_DATA_DIR / "no_iac" / "no_iac_test_graphml"),
        # Reference set — no BioLiP annotations; all nodes will receive label "none"
        "ref":  (DATA_DIR_3 / "ref" / "JSONs" / "structures",
                 ARP_DATA_DIR / "no_iac" / "no_iac_ref_graphml"),
    }

    # Same two lists of settings for every dataset: for each experiment, the
    # enzyme and reference sets are built by the identical function.
    for exp_name in experiments:
        for ds_name, (json_dir, out_parent) in datasets.items():
            parallel_jsons_to_rins(
                json_dir      = json_dir,
                ## CHANGE THIS
                out_dir       = out_parent,
                biolip_csv    = biolip_csv,
                num_processes = num_processes,
                rin_builder   = EXPERIMENTS[exp_name],
            )


if __name__ == "__main__":
    # 11/11/26
    main()
    for exp_name in ["hbond"]:
        validate_rins(ARP_DATA_DIR / "no_iac" / "no_iac_test_graphml")
        validate_rins(ARP_DATA_DIR / "no_iac" / "no_iac_ref_graphml")

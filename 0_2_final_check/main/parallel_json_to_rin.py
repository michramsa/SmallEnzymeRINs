import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

def setup_logger(out_dir: Path) -> logging.Logger:
    """
    Creates a logger writing to both console and <out_dir>/rin_build.log.
    Appends across runs so the full history of every session is preserved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rin_builder")
    logger.setLevel(logging.DEBUG)

    # avoid duplicate handlers if called more than once in a session
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

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

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


def arpeggio_to_rin(json_file, pdb_id: str, label_lookup: dict):
    """Convert a single Arpeggio JSON to a NetworkX residue interaction graph.

    Only protein-protein interactions between the 20 standard amino acids are
    included. Waters, ligands, ions, and cofactors are excluded so that
    centrality measures reflect the protein interaction network only,
    consistent with Amitai et al. (2004).

    Each node receives a 'biolip_label' attribute with one of four values:
        "binding"   — residue contacts a ligand in BioLiP
        "catalytic" — residue is annotated as catalytic in BioLiP
        "dual"      — residue carries both binding and catalytic annotations
        "none"      — residue is absent from BioLiP for this structure

    Parameters
    ----------
    json_file    : path to the Arpeggio JSON
    pdb_id       : lowercase PDB ID (e.g. "1ac0"), used for label lookup
    label_lookup : dict returned by load_biolip_labels()
    """
    with open(json_file, 'r') as f:
        interactions = json.load(f)

    rin = nx.Graph()
    residue_interactions = {}
    n_total    = 0
    n_filtered = 0

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']
        n_total += 1

        # ignore waters
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            n_filtered += 1
            continue

        # ignore non-standard residues (ligands, ions, cofactors, modified AAs)
        if bgn['label_comp_id'] not in STANDARD_AA or end['label_comp_id'] not in STANDARD_AA:
            n_filtered += 1
            continue

        # ignore interactions with the same residue (self loop)
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
                bgn['auth_seq_id'] == end['auth_seq_id'] and
                bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        bgn_node = f"{bgn['auth_asym_id']}_{bgn['auth_seq_id']}"
        end_node = f"{end['auth_asym_id']}_{end['auth_seq_id']}"
        res_pair = tuple(sorted([bgn_node, end_node]))

        if res_pair in residue_interactions:
            continue

        # BioLiP label lookup — keyed by (pdb_id, chain, auth_seq_id as int)
        bgn_key   = (pdb_id, bgn['auth_asym_id'], int(bgn['auth_seq_id']))
        end_key   = (pdb_id, end['auth_asym_id'], int(end['auth_seq_id']))
        bgn_label = label_lookup.get(bgn_key, "none")
        end_label = label_lookup.get(end_key, "none")

        residue_interactions[res_pair] = {
            'interaction': {
                'distance':     interaction['distance'],
                'contact_type': interaction['contact'],
                'bgn_atom':     bgn['auth_atom_id'],
                'end_atom':     end['auth_atom_id'],
                'bgn_res':      bgn['label_comp_id'],
                'end_res':      end['label_comp_id'],
            },
            'nodes': {
                bgn_node: {'residue_type': bgn['label_comp_id'],
                           'chain':        bgn['auth_asym_id'],
                           'residue_num':  bgn['auth_seq_id'],
                           'biolip_label': bgn_label},
                end_node: {'residue_type': end['label_comp_id'],
                           'chain':        end['auth_asym_id'],
                           'residue_num':  end['auth_seq_id'],
                           'biolip_label': end_label},
            }
        }

    for (node1, node2), data in residue_interactions.items():
        for node in (node1, node2):
            if not rin.has_node(node):
                rin.add_node(node, **data['nodes'][node])

        idet    = data['interaction']
        contact = idet['contact_type']
        if isinstance(contact, list):
            contact = ','.join(contact)

        rin.add_edge(
            node1, node2,
            distance=idet['distance'],
            contact_types=contact,
            bgn_atom=idet['bgn_atom'],
            end_atom=idet['end_atom'],
            bgn_res=idet['bgn_res'],
            end_res=idet['end_res'],
        )

    return rin, n_total, n_filtered


def process_single_json(json_file, out_dir, label_lookup):
    """
    Build a RIN and write it as GraphML using a write-to-temp-then-rename
    pattern. This means a crash or interrupt mid-write never leaves a corrupt
    .graphml on disk — only complete files are ever present.

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
        rin, n_total, n_filtered = arpeggio_to_rin(json_file, pdb_id, label_lookup)
        nx.write_graphml(rin, tmp_path)
        tmp_path.rename(out_path)   # atomic on most local filesystems
        return stem, len(rin.nodes()), n_total, n_filtered
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return stem, f"Error: {exc}", 0, 0


def parallel_jsons_to_rins(json_dir, out_dir, biolip_csv, num_processes=None):
    """
    Convert all Arpeggio JSONs in *json_dir* to GraphML RINs in *out_dir*,
    annotating each node with a 'biolip_label' attribute drawn from
    *biolip_csv*.

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

    logger.info("=" * 60)
    logger.info("RIN BUILD SESSION STARTED")
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

    worker  = partial(process_single_json, out_dir=str(out_dir), label_lookup=label_lookup)
    results = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, [str(f) for f in to_process]),
                total=len(to_process),
                desc="Building RINs",
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


def tester():
    # Update paths as needed for one-off testing
    test_file    = DATA_DIR_3 / "test" / "JSONs" / "1ID8_clean.json"
    biolip_csv   = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_residue_labels.csv"
    out_dir      = DATA_DIR_3 / "test" / "RIN_create_test"

    label_lookup = load_biolip_labels(Path(biolip_csv))

    stem     = Path(test_file).stem
    pdb_id   = stem.split("_")[0].lower()
    tmp_path = out_dir / f"{stem}.graphml.tmp"

    rin, n_total, n_filtered = arpeggio_to_rin(test_file, pdb_id, label_lookup)
    nx.write_graphml(rin, tmp_path)

    # Spot-check: print label distribution for this one RIN
    label_dist = defaultdict(int)
    for node, attrs in rin.nodes(data=True):
        label_dist[attrs.get("biolip_label", "MISSING")] += 1
    print(f"Label distribution for {stem}:")
    for lbl, count in sorted(label_dist.items()):
        print(f"  {lbl}: {count}")


def main():
    biolip_csv    = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_residue_labels.csv"
    num_processes = 4   # set to None to use all available CPUs

    # Enzyme set — residues will receive BioLiP labels where available
    parallel_jsons_to_rins(
        json_dir      = DATA_DIR_3 / "test" / "JSONs",
        out_dir       = DATA_DIR_3 / "test" / "RINs",
        biolip_csv    = biolip_csv,
        num_processes = num_processes,
    )

    # Reference set — no BioLiP annotations; all nodes will receive label "none"
    parallel_jsons_to_rins(
        json_dir      = DATA_DIR_3 / "ref" / "JSONs" / "structures",
        out_dir       = DATA_DIR_3 / "ref" / "RINs", 
        biolip_csv    = biolip_csv,
        num_processes = num_processes,
    )


if __name__ == "__main__":
    # 6/28/26 - test
    # tester()

    # main()
    validate_rins(DATA_DIR_3 / "test" / "RINs")
    validate_rins(DATA_DIR_3 / "ref" / "RINs")    # update to actual path

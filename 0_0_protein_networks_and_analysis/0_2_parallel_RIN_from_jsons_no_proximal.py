# RIN construction pipeline — non-specific contact filtering variants
#
# Produces three RIN sets from the same Arpeggio JSONs (no reruns needed):
#
#   Variant 1 — all contacts (ver2 behaviour, reference baseline)
#       Keeps: proximal, vdw, vdw_clash, and all specific types
#       Output: RINs_all  (equivalent to 0_2_parallel_RIN_from_jsons_ver2.py)
#
#   Variant 2 — no proximal-only
#       Drops:  pairs whose sole contact label is "proximal"
#       Keeps:  vdw, vdw_clash, and all specific types
#       Output: RINs_no_proximal
#
#   Variant 3 — specific contacts only
#       Drops:  pairs whose contact set contains only labels from
#               {proximal, vdw, vdw_clash}
#       Keeps:  pairs with at least one chemically specific label
#               (hbond, weak_hbond, polar, weak_polar, hydrophobic,
#                ionic, aromatic, carbonyl, covalent, AMIDEAMIDE, etc.)
#       Output: RINs_specific_only
#
# From the audit (0_0_audit_JSONs.py) on the test set:
#   proximal-only pairs : ~45% of all residue pairs
#   vdw / vdw_clash     : ~38% of all residue pairs
#   specific types      : ~15% of all residue pairs (after dedup)
#
# Robustness interpretation: if centrality-based functional residue
# predictions are stable across all three variants, the conclusions do
# not depend on the choice of contact definition.  Variant 3 is the
# closest analog to CSU (Sobolev et al. 1999), the tool used by
# Amitai et al. (2004), which retains only chemically typed contacts.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# =============================================================================
# CONSTANTS
# =============================================================================
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}

# Labels treated as non-specific (distance-based, no chemical typing)
NON_SPECIFIC = {"proximal", "vdw", "vdw_clash"}


# =============================================================================
# LOGGER
# =============================================================================
def setup_logger(out_dir: Path, name: str) -> logging.Logger:
    """
    Creates a logger writing to both console and <out_dir>/rin_build.log.
    Appends across runs. The name argument keeps loggers for different
    variants from sharing handlers if called in the same session.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(out_dir / "rin_build.log", mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# CORE CONVERSION
# =============================================================================
def arpeggio_to_rin(json_file, drop_if_only: set = None):
    """Convert a single Arpeggio JSON to a NetworkX residue interaction graph.

    Parameters
    ----------
    json_file     : path to Arpeggio JSON
    drop_if_only  : set of contact labels — a residue pair is dropped when
                    its first-seen contact set is a subset of this set
                    (i.e. every label on that pair is in drop_if_only).
                    Pass None or an empty set to keep all pairs (variant 1).

    Filters applied (in order):
      1. Waters excluded.
      2. Non-standard residues excluded.
      3. Self-loops excluded.
      4. Contact-type filter (controlled by drop_if_only).
      5. First-contact-wins deduplication.

    Returns
    -------
    rin            : nx.Graph
    n_total        : total atom-level records in the JSON
    n_filtered     : records removed by filters 1-3
    n_type_dropped : unique residue pairs dropped by filter 4
    """
    if drop_if_only is None:
        drop_if_only = set()

    with open(json_file, 'r') as f:
        interactions = json.load(f)

    rin                  = nx.Graph()
    residue_interactions = {}
    n_total              = 0
    n_filtered           = 0
    n_type_dropped       = 0

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']
        n_total += 1

        # -- filter 1: waters --
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            n_filtered += 1
            continue

        # -- filter 2: non-standard residues --
        if (bgn['label_comp_id'] not in STANDARD_AA or
                end['label_comp_id'] not in STANDARD_AA):
            n_filtered += 1
            continue

        # -- filter 3: self-loops --
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
                bgn['auth_seq_id'] == end['auth_seq_id'] and
                bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        bgn_node = f"{bgn['auth_asym_id']}_{bgn['auth_seq_id']}"
        end_node = f"{end['auth_asym_id']}_{end['auth_seq_id']}"
        res_pair = tuple(sorted([bgn_node, end_node]))

        # first-contact-wins deduplication
        if res_pair in residue_interactions:
            continue

        # -- filter 4: contact-type filter --
        contact = interaction['contact']
        if not isinstance(contact, list):
            contact = [contact]
        contact_set = set(contact)

        if drop_if_only and contact_set.issubset(drop_if_only):
            n_type_dropped += 1
            # mark seen so later atom records for this pair don't re-evaluate
            residue_interactions[res_pair] = None
            continue

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
                           'residue_num':  bgn['auth_seq_id']},
                end_node: {'residue_type': end['label_comp_id'],
                           'chain':        end['auth_asym_id'],
                           'residue_num':  end['auth_seq_id']},
            }
        }

    for (node1, node2), data in residue_interactions.items():
        if data is None:
            continue

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

    return rin, n_total, n_filtered, n_type_dropped


# =============================================================================
# SINGLE-FILE WORKER
# =============================================================================
def process_single_json(json_file, out_dir, drop_if_only):
    """
    Build a RIN and write it as GraphML using write-to-temp-then-rename
    so interrupted runs never leave corrupt files on disk.

    Returns (stem, node_count, n_total, n_filtered, n_type_dropped)
    or      (stem, "Error: ...", 0, 0, 0) on failure.
    """
    stem     = Path(json_file).stem
    out_path = Path(out_dir) / f"{stem}.graphml"
    tmp_path = Path(out_dir) / f"{stem}.graphml.tmp"

    try:
        rin, n_total, n_filtered, n_type_dropped = arpeggio_to_rin(
            json_file, drop_if_only=drop_if_only
        )
        nx.write_graphml(rin, tmp_path)
        tmp_path.rename(out_path)
        return stem, len(rin.nodes()), n_total, n_filtered, n_type_dropped
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return stem, f"Error: {exc}", 0, 0, 0


# =============================================================================
# PARALLEL RUNNER
# =============================================================================
def parallel_jsons_to_rins(json_dir, out_dir, drop_if_only,
                            variant_label, num_processes=None):
    """
    Convert all Arpeggio JSONs in json_dir to GraphML RINs in out_dir.

    Parameters
    ----------
    json_dir      : directory of Arpeggio JSON files
    out_dir       : output directory for GraphML files
    drop_if_only  : set of contact labels to filter on (see arpeggio_to_rin)
    variant_label : short string identifying the variant for log messages
    num_processes : worker count (None = all CPUs)
    """
    json_dir = Path(json_dir)
    out_dir  = Path(out_dir)
    logger   = setup_logger(out_dir, name=f"rin_builder_{variant_label}")

    logger.info("=" * 60)
    logger.info(f"RIN BUILD SESSION STARTED  [{variant_label}]")
    logger.info(f"  JSON source   : {json_dir}")
    logger.info(f"  Output dir    : {out_dir}")
    logger.info(f"  Drop if only  : {drop_if_only if drop_if_only else 'none (keep all)'}")

    stale = list(out_dir.glob("*.graphml.tmp"))
    if stale:
        logger.info(f"Removing {len(stale)} stale .tmp file(s) from previous interrupted run")
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

    worker  = partial(process_single_json,
                      out_dir=str(out_dir),
                      drop_if_only=drop_if_only)
    results = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, [str(f) for f in to_process]),
                total=len(to_process),
                desc=f"Building RINs [{variant_label}]",
            ):
                results.append(result)
                stem, outcome, n_total, n_filtered, n_type_dropped = result
                if str(outcome).startswith("Error"):
                    logger.warning(f"FAILED  {stem}: {outcome}")
                else:
                    pct = (n_filtered / n_total * 100) if n_total > 0 else 0
                    logger.debug(
                        f"OK      {stem} ({outcome} nodes | "
                        f"{n_filtered}/{n_total} residue-filtered [{pct:.1f}%] | "
                        f"{n_type_dropped} pairs dropped by contact filter)"
                    )

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — shutting down worker pool")
        logger.warning(
            f"Completed this session before interrupt: {len(results)} file(s). "
            "Re-run to resume."
        )
        logger.info("=" * 60)
        return

    successes          = [r for r in results if not str(r[1]).startswith("Error")]
    errors             = [r for r in results if     str(r[1]).startswith("Error")]
    total_type_dropped = sum(r[4] for r in successes)

    logger.info(f"Session complete — {len(successes)}/{len(to_process)} succeeded")
    logger.info(f"Total pairs dropped by contact filter: {total_type_dropped:,}")
    if errors:
        logger.warning(f"{len(errors)} error(s) this session:")
        for stem, msg, _, __, ___ in errors:
            logger.warning(f"  {stem}: {msg}")

    logger.info(f"RIN BUILD SESSION DONE  [{variant_label}]")
    logger.info("=" * 60)


# =============================================================================
# VALIDATION
# =============================================================================
def validate_rins(rin_dir):
    """
    Check all GraphML files in rin_dir to confirm every node is one of the
    20 standard amino acids.
    """
    rin_dir   = Path(rin_dir)
    graphmls  = sorted(rin_dir.glob("*.graphml"))
    offenders = {}

    print(f"Validating {len(graphmls)} GraphML files in {rin_dir}...")

    for gml in graphmls:
        graph = nx.read_graphml(gml)
        non_standard = set()
        for node, attrs in graph.nodes(data=True):
            res = attrs.get("residue_type", "")
            if res not in STANDARD_AA:
                non_standard.add(res)
        if non_standard:
            offenders[gml.name] = non_standard

    if offenders:
        print(f"\nFound {len(offenders)} file(s) with non-standard residues:")
        for fname, residues in sorted(offenders.items()):
            print(f"  {fname}: {sorted(residues)}")
    else:
        print("All GraphML files contain only standard amino acids.")

    return offenders


# =============================================================================
# MAIN
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # EDIT THESE
    # -------------------------------------------------------------------------
    json_dir      = DATA_DIR_2 / "enzymes_CIF_unflagged" / "JSONs"
    num_processes = 4    # set to None to use all available CPUs

    # Output directories — one per variant
    out_dir_all      = DATA_DIR_2 / "RINs_all"            # variant 1
    out_dir_no_prox  = DATA_DIR_2 / "RINs_no_proximal"    # variant 2
    out_dir_specific = DATA_DIR_2 / "RINs_specific_only"  # variant 3

    # Which variants to run — set to False to skip one
    run_variant_1 = False
    run_variant_2 = False
    run_variant_3 = True
    # -------------------------------------------------------------------------

    if run_variant_1:
        print("\n--- Variant 1: all contacts ---")
        parallel_jsons_to_rins(
            json_dir, out_dir_all,
            drop_if_only=set(),
            variant_label="all_contacts",
            num_processes=num_processes
        )
        validate_rins(out_dir_all)

    if run_variant_2:
        print("\n--- Variant 2: no proximal-only ---")
        parallel_jsons_to_rins(
            json_dir, out_dir_no_prox,
            drop_if_only={"proximal"},
            variant_label="no_proximal",
            num_processes=num_processes
        )
        validate_rins(out_dir_no_prox)

    if run_variant_3:
        print("\n--- Variant 3: specific contacts only ---")
        parallel_jsons_to_rins(
            json_dir, out_dir_specific,
            drop_if_only=NON_SPECIFIC,   # {"proximal", "vdw", "vdw_clash"}
            variant_label="specific_only",
            num_processes=num_processes
        )
        validate_rins(out_dir_specific)


if __name__ == "__main__":
    main()

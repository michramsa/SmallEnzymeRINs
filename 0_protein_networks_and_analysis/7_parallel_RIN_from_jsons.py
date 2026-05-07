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

def arpeggio_to_rin(json_file):
    """Convert a single Arpeggio JSON to a NetworkX residue interaction graph."""
    with open(json_file, 'r') as f:
        interactions = json.load(f)

    rin = nx.Graph()
    residue_interactions = {}

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']

        # ignore waters
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
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

    return rin

def process_single_json(json_file, out_dir):
    """
    Build a RIN and write it as GraphML using a write-to-temp-then-rename
    pattern. This means a crash or interrupt mid-write never leaves a corrupt
    .graphml on disk — only complete files are ever present.

    Returns (stem, node_count) on success or (stem, "Error: ...") on failure.
    """
    stem     = Path(json_file).stem
    out_path = Path(out_dir) / f"{stem}.graphml"
    tmp_path = Path(out_dir) / f"{stem}.graphml.tmp"

    try:
        rin = arpeggio_to_rin(json_file)
        nx.write_graphml(rin, tmp_path)
        tmp_path.rename(out_path)   # atomic on most local filesystems
        return stem, len(rin.nodes())
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return stem, f"Error: {exc}"

def parallel_jsons_to_rins(json_dir, out_dir, num_processes=None):
    """
    Convert all Arpeggio JSONs in *json_dir* to GraphML RINs in *out_dir*.

    Interrupt behaviour
    -------------------
    Ctrl+C is caught cleanly: worker processes are terminated, the log records
    how many files were completed in the interrupted session, and re-running the
    script resumes automatically from where it left off (already-complete .graphml
    files are skipped; stale .tmp files from the interrupted run are deleted first).

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

    worker  = partial(process_single_json, out_dir=str(out_dir))
    results = []

    try:
        with mp.Pool(processes=num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(worker, [str(f) for f in to_process]),
                total=len(to_process),
                desc="Building RINs",
            ):
                results.append(result)
                stem, outcome = result
                if str(outcome).startswith("Error"):
                    logger.warning(f"FAILED  {stem}: {outcome}")
                else:
                    logger.debug(f"OK      {stem} ({outcome} nodes)")

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
        for stem, msg in errors:
            logger.warning(f"  {stem}: {msg}")

    logger.info("RIN BUILD SESSION DONE")
    logger.info("=" * 60)

def tester():
    # 5/4/2026 - tester run to check output with 2025 code
    test_file = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/1CWX_clean.json"
    stem     = Path(test_file).stem
    out_dir = Path("/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot")
    rin_file = arpeggio_to_rin(test_file)
    tmp_path = Path(out_dir) / f"{stem}.graphml.tmp"
    nx.write_graphml(rin_file, tmp_path)

    # 5/4/2026 - tester run 2
    test_file = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/6AVD_clean.json"
    stem     = Path(test_file).stem
    out_dir = Path("/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot")
    rin_file = arpeggio_to_rin(test_file)
    tmp_path = Path(out_dir) / f"new_{stem}.graphml"
    nx.write_graphml(rin_file, tmp_path)

def main():
    # Edit these two paths
    json_dir = UF_ENZYMES_DIR / "JSONs"  # folder of .json files
    out_dir  = UF_ENZYMES_DIR / "RINs"            # GraphML files go here

    num_processes = 4   # set to None to use all available CPUs

    parallel_jsons_to_rins(json_dir, out_dir, num_processes)


if __name__ == "__main__":
    main()

    # 5/5/2026
    # tester()
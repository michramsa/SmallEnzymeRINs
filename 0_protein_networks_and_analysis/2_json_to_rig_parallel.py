# Parallel JSON-to-RIG Pipeline
# Converts Arpeggio JSON output to Residue Interaction Graphs (RIGs) in parallel.
# Run this AFTER 1_parallel_arpeggio.py has produced JSONs.
#
# This does NOT require the Arpeggio conda environment - just networkx and matplotlib.
#
# Steps:
# 1. Find all Arpeggio JSON files in the output directory
# 2. Convert each JSON to a NetworkX graph (RIG) in parallel
# 3. Save as GraphML

# for subdirectories to find the imports.py file
import sys
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

MAX_WORKERS = 8  # safe for 14 cores / 20 logical processors


def arpeggio_to_rig(json_file):
    """
    Convert Arpeggio JSON output to a Residue Interaction Graph (RIG).

    Parameters:
    - json_file: Path to the Arpeggio JSON output file

    Returns:
    - rig: NetworkX Graph representing residue interactions
    """
    with open(json_file, 'r') as f:
        interactions = json.load(f)

    rig = nx.Graph()

    # Keep track of all interactions between residue pairs
    residue_interactions = {}

    # process each atom-atom interaction
    for interaction in interactions:
        # get residue info from beginning and end atoms
        bgn = interaction['bgn']
        end = interaction['end']

        # skip water
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            continue

        # skip interactions involving the same residue
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
            bgn['auth_seq_id'] == end['auth_seq_id'] and
            bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        # create residue identifiers (chain_residueNum)
        bgn_chain_id = bgn['auth_asym_id']
        bgn_res_num_id = bgn['auth_seq_id']
        end_chain_id = end['auth_asym_id']
        end_res_num_id = end['auth_seq_id']

        bgn_node_name = str(bgn_chain_id) + "_" + str(bgn_res_num_id)
        end_node_name = str(end_chain_id) + "_" + str(end_res_num_id)

        res_pair = tuple(sorted([bgn_node_name, end_node_name]))

        if res_pair in residue_interactions:
            continue

        interaction_detail = {
            'distance': interaction['distance'],
            'contact_type': interaction['contact'],
            'bgn_atom': bgn['auth_atom_id'],
            'end_atom': end['auth_atom_id'],
            'bgn_res': bgn['label_comp_id'],
            'end_res': end['label_comp_id'],
            'bgn_node': bgn_node_name,
            'end_node': end_node_name
        }

        residue_interactions[res_pair] = {
            'interaction': interaction_detail,
            'residue_info': {
                bgn_node_name: {
                    'residue_type': bgn['label_comp_id'],
                    'chain': bgn['auth_asym_id'],
                    'residue_num': bgn['auth_seq_id']
                },
                end_node_name: {
                    'residue_type': end['label_comp_id'],
                    'chain': end['auth_asym_id'],
                    'residue_num': end['auth_seq_id']
                }
            }
        }

    # process the collected data to build the graph
    for res_pair, data in residue_interactions.items():
        node1, node2 = res_pair

        # add nodes with details
        if not rig.has_node(node1):
            rig.add_node(
                node1,
                residue_type=data['residue_info'][node1]['residue_type'],
                chain=data['residue_info'][node1]['chain'],
                residue_num=data['residue_info'][node1]['residue_num']
            )

        if not rig.has_node(node2):
            rig.add_node(
                node2,
                residue_type=data['residue_info'][node2]['residue_type'],
                chain=data['residue_info'][node2]['chain'],
                residue_num=data['residue_info'][node2]['residue_num']
            )

        # get the single interaction for this residue pair
        interaction = data['interaction']

        # add edge with attributes from the single interaction
        rig.add_edge(
            node1, node2,
            distance=interaction['distance'],
            contact_types=",".join(interaction['contact_type']),
            bgn_atom=interaction['bgn_atom'],
            end_atom=interaction['end_atom'],
            bgn_res=interaction['bgn_res'],
            end_res=interaction['end_res']
        )

    return rig


def visualize_rig(rig, output_file=None):
    """
    Visualize a Residue Interaction Graph.

    Parameters:
    - rig: NetworkX Graph
    - output_file: Optional path to save the figure
    """
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(rig, seed=42)

    # Get node colors based on chain
    chains = set(data['chain'] for _, data in rig.nodes(data=True))
    chain_colors = {chain: plt.cm.tab10(i) for i, chain in enumerate(chains)}
    node_colors = [chain_colors[data['chain']] for _, data in rig.nodes(data=True)]

    # Draw the graph
    nx.draw_networkx_nodes(rig, pos, node_size=100, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(rig, pos, width=0.5, alpha=0.5)
    nx.draw_networkx_labels(rig, pos, font_size=8)

    plt.title("Residue Interaction Graph")
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()


def get_all_json_files(jsons_dir):
    """
    Find all Arpeggio JSON files in the output directory structure.

    Parameters:
    - jsons_dir: Base directory containing per-acr_type JSON subdirectories

    Returns:
    - List of tuples: (json_file_path, acr_type, structure_name)
    """
    json_files = []

    if not jsons_dir.exists():
        print(f"Warning: {jsons_dir} does not exist")
        return json_files

    for acr_type_dir in jsons_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            for json_file in acr_type_dir.glob("*.json"):
                # structure name from the JSON filename (strip _clean suffix if present)
                structure_name = json_file.stem.replace("_clean", "")
                json_files.append((json_file, acr_type, structure_name))

    return json_files


def process_single_rig(json_file, acr_type, structure_name, rigs_dir):
    """
    Convert a single Arpeggio JSON to a RIG GraphML file.

    This is the worker function called by each parallel process.

    Parameters:
    - json_file: Path to the Arpeggio JSON file
    - acr_type: The Acr type (used for subdirectory naming)
    - structure_name: Name of the structure
    - rigs_dir: Base output directory for RIG files

    Returns:
    - dict with status info for logging/summary
    """
    result_info = {
        'acr_type': acr_type,
        'structure': structure_name,
        'status': 'success',
        'error': None,
        'elapsed': 0,
        'skipped_existing': False,
        'nodes': 0,
        'edges': 0
    }

    start_time = time.time()

    rig_out_dir = rigs_dir / acr_type
    rig_out_dir.mkdir(parents=True, exist_ok=True)

    rig_path = rig_out_dir / f"{structure_name}.graphml"

    # Skip if already exists
    if rig_path.exists():
        result_info['skipped_existing'] = True
        result_info['elapsed'] = time.time() - start_time
        return result_info

    try:
        rig = arpeggio_to_rig(json_file)
        nx.write_graphml(rig, rig_path)
        result_info['nodes'] = rig.number_of_nodes()
        result_info['edges'] = rig.number_of_edges()
    except Exception as e:
        result_info['status'] = 'failed'
        result_info['error'] = str(e)

    result_info['elapsed'] = time.time() - start_time
    return result_info


def write_log_entry(log_fh, result):
    """Write a single result to the log file and flush immediately."""
    status = result['status'].upper()
    if result.get('skipped_existing'):
        status = "EXISTED"

    line = (
        f"[{status:<7}] {result['acr_type']}/{result['structure']}"
        f"  ({result['elapsed']:.1f}s)"
    )
    if result['status'] == 'success' and not result.get('skipped_existing'):
        line += f"  nodes={result['nodes']}  edges={result['edges']}"
    if result.get('error'):
        line += f"  error={result['error']}"

    log_fh.write(line + "\n")
    log_fh.flush()


if __name__ == "__main__":

    ## Input/output directories
    jsons_dir = DATA_DIR / "arp_results" / "JSONs"
    rigs_dir  = DATA_DIR / "arp_results" / "RIGs"
    log_dir   = DATA_DIR / "arp_results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ## Gather all JSON files produced by 1_parallel_arpeggio.py
    json_files = get_all_json_files(jsons_dir)
    print(f"Found {len(json_files)} JSON files across all Acr types")

    ## Sort by file size (smallest first for better load balancing)
    json_files = sorted(json_files, key=lambda x: os.path.getsize(x[0]))

    ## mini test - uncomment to test with a small batch
    # json_files = json_files[:5]
    # print(f"test: run on {len(json_files)} files: {[s for _, _, s in json_files]}")

    ## Open log file BEFORE the executor so every result is recorded incrementally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rig_run_{timestamp}.log"

    with open(log_file, "w") as log_fh:
        # Write header
        log_fh.write(f"RIG conversion run: {datetime.now()}\n")
        log_fh.write(f"Workers: {MAX_WORKERS}\n")
        log_fh.write(f"Total files: {len(json_files)}\n")
        log_fh.write(f"{'='*60}\n\n")
        log_fh.flush()

        ## Run in parallel
        results = []
        completed = 0
        print(f"Starting parallel RIG conversion with {MAX_WORKERS} workers...")
        print(f"Log file: {log_file}\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(
                    process_single_rig,
                    json_file, acr_type, structure_name, rigs_dir
                ): structure_name
                for json_file, acr_type, structure_name in json_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                name = future_to_file[future]
                completed += 1
                progress = f"[{completed}/{len(json_files)}]"

                try:
                    result = future.result()
                    results.append(result)

                    if result.get('skipped_existing'):
                        tag = "EXISTED"
                    elif result['status'] == 'success':
                        tag = "OK"
                    else:
                        tag = "FAIL"

                    detail = ""
                    if result['status'] == 'success' and not result.get('skipped_existing'):
                        detail = f"  nodes={result['nodes']}  edges={result['edges']}"

                    print(f"  {progress} [{tag:<7}] {result['acr_type']}/{result['structure']} "
                          f"({result['elapsed']:.1f}s){detail}")

                    # Write to log immediately
                    write_log_entry(log_fh, result)

                except Exception as exc:
                    print(f"  {progress} [ERR    ] {name} raised exception: {exc}")
                    err_result = {
                        'acr_type': '?', 'structure': name,
                        'status': 'failed', 'error': str(exc),
                        'elapsed': 0, 'nodes': 0, 'edges': 0
                    }
                    results.append(err_result)
                    write_log_entry(log_fh, err_result)

        ## Summary (written to both console and log)
        successes = [r for r in results if r['status'] == 'success' and not r.get('skipped_existing')]
        failures  = [r for r in results if r['status'] == 'failed']
        existed   = [r for r in results if r.get('skipped_existing')]

        summary_lines = [
            f"\n{'='*60}",
            "RIG CONVERSION COMPLETE",
            f"  Newly converted:    {len(successes)}",
            f"  Already existed:    {len(existed)}",
            f"  Failed:             {len(failures)}",
        ]
        if successes:
            avg = sum(r['elapsed'] for r in successes) / len(successes)
            summary_lines.append(f"  Avg time (new):     {avg:.1f}s")
        if failures:
            summary_lines.append(f"\n  Failed structures:")
            for r in failures:
                summary_lines.append(f"    {r['acr_type']}/{r['structure']}: {r['error']}")
        summary_lines.append(f"{'='*60}")

        for line in summary_lines:
            print(line)
            log_fh.write(line + "\n")
        log_fh.flush()

    print(f"\nLog written to: {log_file}")

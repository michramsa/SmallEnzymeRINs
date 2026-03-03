# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# This file converts Arpeggio JSON output to Residue Interaction Graphs (RIGs).
# Run this AFTER 1_parallel_arpeggio.py has produced JSONs.
#
# This does NOT require the Arpeggio conda environment - just networkx and matplotlib.
#
# Steps:
# 1. Find all Arpeggio JSON files in the output directory
# 2. Convert each JSON to a NetworkX graph (RIG)
# 3. Save as GraphML


def setup_logger(log_dir):
    """
    Sets up a logger that writes to both terminal and a timestamped log file.
    
    Parameters:
    - log_dir: Directory to save the log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rig_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Log file: {log_file}")
    return logging.getLogger()


def log_step(logger, step, acr_type, structure_name, status, error=None):
    """
    Logs a single pipeline step with consistent formatting.
    """
    msg = f"[{step:<10}] acr_type={acr_type} | file={structure_name} | status={status}"
    if error:
        msg += f" | error={error}"
    
    if status == "FAILED":
        logger.error(msg)
    elif status == "SKIP":
        logger.debug(msg)
    else:
        logger.info(msg)


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


if __name__ == "__main__":

    ## Input/output directories
    jsons_dir = DATA_DIR / "arp_results" / "JSONs"
    rigs_dir  = DATA_DIR / "arp_results" / "RIGs"

    ## Gather all JSON files produced by 1_parallel_arpeggio.py
    json_files = get_all_json_files(jsons_dir)
    print(f"Found {len(json_files)} JSON files across all Acr types")

    logger = setup_logger(DATA_DIR / "arp_results" / "logs")
    logger.info(f"Found {len(json_files)} JSON files across all Acr types")

    failed_rig = []

    for json_file, acr_type, structure_name in json_files:

        rig_out_dir = rigs_dir / acr_type
        rig_out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # convert Arpeggio JSON to RIG and save as GraphML
        # ----------------------------------------------------------------
        rig_path = rig_out_dir / f"{structure_name}.graphml"

        if rig_path.exists():
            log_step(logger, "RIG", acr_type, structure_name, "SKIP")
        else:
            log_step(logger, "RIG", acr_type, structure_name, "START")
            try:
                rig = arpeggio_to_rig(json_file)
                nx.write_graphml(rig, rig_path)
                log_step(logger, "RIG", acr_type, structure_name, "SUCCESS")
            except Exception as e:
                log_step(logger, "RIG", acr_type, structure_name, "FAILED", error=str(e))
                failed_rig.append((acr_type, structure_name))

    logger.info("=== RIG CONVERSION COMPLETE ===")
    logger.info(f"Failed RIG conversion ({len(failed_rig)}): {failed_rig}")

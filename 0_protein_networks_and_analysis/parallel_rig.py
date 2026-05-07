# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

### convert JSON to RIG, input: json, output: networkX file
def arpeggio_to_rig(json_file):
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
        
        # create residue identifiers (chain, residue_num)
        bgn_chain_id = bgn['auth_asym_id']
        bgn_res_num_id = bgn['auth_seq_id']
        end_chain_id = end['auth_asym_id'] 
        end_res_num_id = end['auth_seq_id']

        bgn_node_name = str(bgn_chain_id)+ "_" + str(bgn_res_num_id)
        end_node_name = str(end_chain_id)+ "_" + str(end_res_num_id)
        
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
            'bgn_node' : bgn_node_name,
            'end_node' : end_node_name
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
            contact_types=interaction['contact_type'],
            bgn_atom=interaction['bgn_atom'],
            end_atom=interaction['end_atom'],
            bgn_res=interaction['bgn_res'],
            end_res=interaction['end_res']
        )
    
    return rig

### see created graph, input: networkX graph, output: optional, path to save it 
def visualize_rig(rig, output_file=None):
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

# inputs: cifs directory, output: list containing Paths to jsons of interest for creating RIGs
def size_sort(cifs_dir, out_dir, json_dir, max_size_kb):
    # check existence of an output directory, create if not there
    os.makedirs(out_dir, exist_ok=True)

    # sort all CIFs by size
    cif_files = list(Path(cifs_dir).glob("*.cif"))
    # print(cif_files)
    cif_files = sorted(cif_files, key=lambda p: os.path.getsize(p))
    # print(cif_files)

    # check existing files already turned
    processed_files = set(f.stem for f in Path(out_dir).glob("*.json"))
    to_process = [f for f in cif_files if f.stem not in processed_files]
    print(to_process)
    # filter big list by file size
    filtered_files = []
    oversized_files = []
    for f in to_process:
        # out of all the files, if the file is of the correct size, we need to see if the json exists
        if os.path.getsize(f) / 1024 <= max_size_kb:
            # get the base of the existing cif file
            # this matches with the corresponding JSON file
            base_name = f.stem
            print(base_name)
            base_name = base_name.split(".")[0]
            # path to jsons
            json_path = Path(json_dir)
            # hypothetical existing file
            total_path = json_path / f"{base_name}.json"
            # if it exists, then it can be included for processing
            if total_path.exists():
                filtered_files.append(total_path)
        else:
            print("Oversized")
            oversized_files.append(f)
    print(filtered_files)
    return filtered_files, oversized_files

# Process a single JSON file and save as GraphML
def process_single_json(json_file, out_dir):
    try:
        rig = arpeggio_to_rig(json_file)
        base_name = Path(json_file).stem
        
        # Convert contact_types from list to string if needed
        for u, v, attr in rig.edges(data=True):
            if 'contact_types' in attr and isinstance(attr['contact_types'], list):
                attr['contact_types'] = ','.join(attr['contact_types'])
        
        output_path = os.path.join(out_dir, f"{base_name}.graphml")
        nx.write_graphml(rig, output_path)
        return base_name, len(rig.nodes())
    except Exception as e:
        return Path(json_file).stem, f"Error: {str(e)}"

# This is to turn a list of potential jsons into RIGs using parallelization
def parallel_arpeggio_to_rig(json_list, out_dir, num_processes=None):
    # check existence of an output directory, create if not there
    os.makedirs(out_dir, exist_ok=True)

    # check existing files already turned
    processed_files = set(f.stem for f in Path(out_dir).glob("*.graphml"))
    to_process = [str(f) for f in json_list if f.stem not in processed_files]

    total_files = len(to_process)
    print(f"Found {total_files} structures to process")
    
    if total_files == 0:
        print("No files to process.")
        return
    
    # If num_processes is not specified, use CPU count
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Create a partial function with fixed out_dir
    process_func = partial(process_single_json, out_dir=out_dir)
    
    # Create a pool of worker processes
    print(f"Using {num_processes} processes for parallelization")
    
    results = []
    with mp.Pool(processes=num_processes) as pool:
        # Process files with progress bar
        for result in tqdm(pool.imap_unordered(process_func, to_process), total=total_files):
            results.append(result)
    
    # Print summary
    success_count = sum(1 for r in results if not str(r[1]).startswith("Error"))
    print(f"Successfully processed {success_count} out of {total_files} files")
    
    # Print any errors
    errors = [r for r in results if str(r[1]).startswith("Error")]
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for base_name, error in errors[:10]:  # Show first 10 errors
            print(f"  - {base_name}: {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

def tester():
    # 5/4/2026 - tester run with new code 1
    # test_file = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/1CWX_clean.json"
    # out_dir = Path("/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot")
    # process_single_json(test_file, out_dir)

    # 5/4/2026 - tester run with new code 2
    test_file = "/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot/6AVD_clean.json"
    out_dir = Path("/home/epi2melabs/AcrNetworks-clean-v2/7_RIG_creation_troubleshoot")
    process_single_json(test_file, out_dir)

def main():
    for i in range(0,10):
        print("Hold for tester")
    # first run params
    # place for completed RIGs
    # out_dir = PISCES_DIR / "RIGs"
    # # place for all your JSONs
    # json_dir = PISCES_DIR / "JSONs"
    # # place for all your CIFs
    # cifs_dir = PISCES_DIR / "CIFs"
    # # place for all your graphs
    # graphs_dir = PISCES_DIR / "GraphMLs"

    # second run params - 5/19/2025
    # place for completed RIGs
    # out_dir = IMP_DIR / "Cleaned RIGs"
    # # place for all your JSONs
    # json_dir = IMP_DIR / "Cleaned JSONs"
    # # place for all your CIFs
    # cifs_dir = IMP_DIR / "mmCIF files"


    # Get all CIF files and sort by size - also creates out_dir if it doesn't already exist
    # filter_list, oversize_list = size_sort(cifs_dir, out_dir, json_dir, 350)
    
    # print(f"Found {len(filter_list)} structures to process")
    # if filter_list:
    #     print(f"First file to process: {filter_list[0]}")
    # if oversize_list:
    #     print(f"First oversized file: {oversize_list[0]}")

    # # Set the number of processes (optional, defaults to CPU count if not specified)
    # num_processes = 4
    
    # # Run parallelized conversion
    # parallel_arpeggio_to_rig(filter_list, out_dir, num_processes)
    
if __name__ == "__main__":
    # main()
    tester()

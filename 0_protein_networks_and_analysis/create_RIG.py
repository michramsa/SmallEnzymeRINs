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
    
    #### debugging stuff
    #counter = 0

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
        
        # print((end_chain_id, end_res_num_id))
        # print(residue_interactions.values())

        res_pair = tuple(sorted([bgn_node_name, end_node_name]))

        if res_pair in residue_interactions:
             continue

        ## new

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
    
    #### text to print simplified view of the graph for debugging
    print("Graph Nodes:")
    for node, attrs in rig.nodes(data=True):
        print(f"  {node}: {attrs}")
    
    print("\nGraph Edges:")
    for u, v, attrs in rig.edges(data=True):
        print(f"  {u} -- {v}:")
        print(f"    Distance: {attrs['distance']}")
        print(f"    Contact Types: {', '.join(attrs['contact_types'])}")
        print(f"    Atoms: {attrs['bgn_atom']} ({attrs['bgn_res']}) -- {attrs['end_atom']} ({attrs['end_res']})")
    
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
    cif_files = sorted(cif_files, key=lambda p: os.path.getsize(p))

    # check existing files already turned
    processed_files = set(f.stem for f in Path(out_dir).glob("*.json"))
    to_process = [f for f in cif_files if f.stem not in processed_files]

    # filter big list by file size
    filtered_files = []
    oversized_files = []
    for f in to_process:
        # out of all the files, if the file is of the correct size, we need to see if the json exists
        if os.path.getsize(f) / 1024 <= max_size_kb:
            # get the base of the existing cif file
            # this matches with the corresponding JSON file
            base_name = f.stem
            # path to jsons
            json_path = Path(json_dir)
            # hypothetical existing file
            total_path = json_path / f"{base_name}.json"
            # if it exists, then it can be included for processing
            if total_path.exists():
                filtered_files.append(total_path)
        else:
            oversized_files.append(f)
    
    return filtered_files, oversized_files

# this is to turn a list of potential jsons into RIGs
def batch_arpeggio_to_rig(json_list, out_dir):
    # check existence of an output directory, create if not there
    os.makedirs(out_dir, exist_ok=True)

    # check existing files already turned
    processed_files = set(f.stem for f in Path(out_dir).glob("*.graphml"))
    to_process = [f for f in json_list if f.stem not in processed_files]

    print(f"Found {len(to_process)} structures to process")

    # batch run
    for f in to_process:
        rig = arpeggio_to_rig(f)
        print(len(rig.nodes()))
        base_name = f.stem
        for u, v, attr in rig.edges(data=True):
            if 'contact_types' in attr and isinstance(attr['contact_types'], list):
                attr['contact_types'] = ','.join(attr['contact_types'])
        nx.write_graphml(rig, os.path.join(out_dir, base_name+".graphml"))
        break


def main():
    # place for completed RIGs
    out_dir = PISCES_DIR / "RIGs"
    # place for all your JSONs
    json_dir = IMP_DIR / "Cleaned JSONs" ## Cleaned JSONs
    # place for all your CIFs
    cifs_dir = PISCES_DIR / "CIFs"
    # place for all your graphs
    graphs_dir = PISCES_DIR / "GraphMLs"

    # # Replace with your actual Arpeggio JSON file
    # json_file = "/home/epi2melabs/projects/arpeggio/out/7C78.json"
    
    # Get all CIF files and sort by size
    filter_list, oversize_list = size_sort(cifs_dir, out_dir, json_dir, 150)
    
    print(f"Found {len(filter_list)} structures to process")
    print(filter_list[0])
    print(oversize_list[0])

    # # Convert to RIG
    batch_arpeggio_to_rig(filter_list, graphs_dir)
    
    # # Export the RIG to various formats
    # nx.write_gexf(rig, "residue_interaction_graph.gexf")  # For Gephi
    # nx.write_graphml(rig, "residue_interaction_graph.graphml")  # General format
    
    # # Export as edge list with key attributes
    # with open("rig_edges.csv", "w") as f:
    #     f.write("source,target,min_distance,num_atom_interactions\n")
    #     for u, v, data in rig.edges(data=True):
    #         f.write(f"{u},{v},{data['min_distance']},{data['num_atom_interactions']}\n")

if __name__ == "__main__":
    # main()

    save_dir = PISCES_DIR / "known acr data"

    file1 = PISCES_DIR / "known acr data/2LW5_clean.json"

    rig1 = arpeggio_to_rig(file1)
    print(len(rig1.nodes()))
    for u, v, attr in rig1.edges(data=True):
        if 'contact_types' in attr and isinstance(attr['contact_types'], list):
            attr['contact_types'] = ','.join(attr['contact_types'])
    nx.write_graphml(rig1, os.path.join(save_dir, "2LW5_clean"+".graphml"))

    file2 = PISCES_DIR / "known acr data/5Y6A_clean.json"
    rig2 = arpeggio_to_rig(file2)
    print(len(rig2.nodes()))
    for u, v, attr in rig2.edges(data=True):
        if 'contact_types' in attr and isinstance(attr['contact_types'], list):
            attr['contact_types'] = ','.join(attr['contact_types'])
    nx.write_graphml(rig2, os.path.join(save_dir, "5Y6A_clean"+".graphml"))

    file3 = PISCES_DIR / "known acr data/6vrb_clean.json"
    rig3 = arpeggio_to_rig(file3)
    print(len(rig3.nodes()))
    for u, v, attr in rig3.edges(data=True):
        if 'contact_types' in attr and isinstance(attr['contact_types'], list):
            attr['contact_types'] = ','.join(attr['contact_types'])
    nx.write_graphml(rig3, os.path.join(save_dir, "6vrb_clean"+".graphml"))

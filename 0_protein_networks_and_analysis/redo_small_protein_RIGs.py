# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
from Amitai.retrieve_PISCES_data import *
from parallel_arpeggio import *

# Created 5/7/2025
# This is a redo of the prior run 
# Input: PDB files for "small" proteins (file size 350 kb or less)
# Steps will include the following:
# (optional) download arpeggio, might be better to create a new environment for this
# 1. Using the filtered CSV file to sort through all of the PDB.gz files - open all of the ones of interest
# 2. Arpeggio requires some particular file format for it to function - need to turn all the pdb files into cleaned, mmCIF files 
# 3. Batch run Arpeggio - output is a JSON file containing all of the interactions

# This file is an edited version of the apreggio_PISCES.py file

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

def unzip_PDBS(pdb_directory, output_directory):
    # create directory if it doesn't already exist
    os.makedirs(output_directory, exist_ok=True)
    #iterate through the files and unzip all of them
    for filename in os.listdir(pdb_directory):
        file_path = os.path.join(pdb_directory, filename)
        if os.path.isfile(file_path):  # only process files, not directories
            with gzip.open(file_path, 'rb') as f_in:
                new_filename = filename.split('.')[0] + '.pdb'
                output_path = os.path.join(output_directory, new_filename)
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":

    # 1. Using the filtered CSV file to sort through all of the PDB.gz files - open all of the ones of interest

    csv_dir = PISCES_DIR / "Centralities"
    all_pdbs = PISCES_DIR / "PDBs"
    output_dir = IMP_DIR / "Cleaned PDBs"
    cif_dir = IMP_DIR / "mmCIF files"

    ## pretty much working backwards, i want the PDB file names of what is in the centralities folder
    for pdb_file in Path(csv_dir).glob("*.csv"):
        # this is just where you want to save those cleaned PDBs
        dest_path = IMP_DIR / "Cleaned PDBs"

        base_name = pdb_file.stem
        base_name = base_name.split("_")
        pdb_name = all_pdbs + "/"+base_name[0] + ".pdb"
        # check existence of the centrality file
        ##### 2. Arpeggio requires some particular file format for it to function - need to turn all the pdb files into cleaned, mmCIF files 
        # if os.path.exists(pdb_name):
        #     # print("wee!")
        #     clean_name = "REDO_" + base_name[0] + "_clean.pdb"
        #     if os.path.exists(clean_name):
        #         print(f"Exists and skipping: {clean_name}")
        #         continue
        #     else:
        #         # this is to access that file with that function
        #         clean_pdb_path = "/home/epi2melabs/projects/pdbtools/clean_pdb.py"
        #         print(f"Creating clean file for: {base_name[0]}")
        #         # create what we want to name it
        #         clean_name = os.path.join(output_dir, "REDO_" + base_name[0] + "_clean.pdb")
        #         result = subprocess.run(f"python3 {clean_pdb_path} -kh '{pdb_name}'", shell=True,capture_output=True,text=True)
        #         expected_clean_file = PISCES_DIR / "PDBs/"+base_name[0] + ".clean.pdb"
        #         if result.returncode == 0:
        #             # Check if the clean file was created
        #             if os.path.exists(expected_clean_file):
        #                 pdb_file = dest_path+"/"+base_name[0]+".clean.pdb"
        #                 cif_name = cif_dir+"/"+base_name[0]+".clean.cif"
        #                 if os.path.exists(cif_name):
        #                     print(f"Exists and skipping: {cif_name}")
        #                     continue
        #                 else:
        #                     # Move the file to the destination with the new name
        #                     shutil.move(expected_clean_file, dest_path)
        #                     print(f"Clean process completed successfully")
        #                     print(f"Moved to: {dest_path}")
        #                     print(f"Creating the mmCIF file")

        #                     ## Important note: if you have arpeggio in a separate environment - here is where you activate that environment
        #                     ## otherwise, this script will not run and throw a "Errno 2: No such file or directory: 'gemmi'"
        #                     subprocess.run(["gemmi", "convert", str(pdb_file), str(cif_name)])
        #             else:
        #                 print(f"Warning: Expected clean file not found: {expected_clean_file}")
        #         else:
        #             print(f"Error running clean_pdb.py: {result.stderr}")
    # 3. Batch run Arpeggio - output is a JSON file containing all of the interactions - we can do this in parallel
    all_cifs_dir = IMP_DIR / "mmCIF files"  
    out_dir = IMP_DIR / "Cleaned JSONs"
    results = process_all_structures(all_cifs_dir, out_dir, max_workers=4, max_size_kb=350)

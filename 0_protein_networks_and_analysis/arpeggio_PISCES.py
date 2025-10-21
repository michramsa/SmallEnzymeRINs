# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
from Amitai.retrieve_PISCES_data import *

# This file is primarily for taking the PISCES data and running it through arpeggio
# Steps will include the following:
# (optional) download arpeggio, might be better to create a new environment for this
# 1. Using the filtered CSV file to sort through all of the PDB.gz files - open all of the ones of interest
# 2. Arpeggio requires some particular file format for it to function - need to turn all the pdb files into cleaned, mmCIF files 
# 3. Batch run Arpeggio - output is a JSON file containing all of the interactions

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
    filtered_csv_file = PISCES_DIR / "PISCES_data_filtered.csv"
    source_dir = PISCES_DIR / "PDBs_compressed"
    output_directory = PISCES_DIR / "PDBs"

    ### taking the csv file, sifting through and opening the ones we care about
    # df = pd.read_csv(filtered_csv_file)
    # print(df.head())
    # id_list = df['id'].tolist()
    # no_chain_ids = []
    # print(id_list)
    # for item in id_list:
    #     no_chain_ids.append(item[:4])

    # to_be_unzipped = []
    # failed_files = []

    # for file in os.listdir(source_dir):
    #     file_name = file.split('.')
    #     id = file_name[0]
    #     # print(id)
    #     if id in no_chain_ids:
    #         full_name = source_dir+"/"+file
    #         try:
    #             with gzip.open(full_name, 'rb') as f_in:
    #                 new_filename = id + '.pdb'
    #                 try:
    #                     output_path = os.path.join(output_directory, new_filename)
    #                     with open(output_path, 'wb') as f_out:
    #                         shutil.copyfileobj(f_in, f_out)
    #                 except EOFError as e:
    #                     print("Failed! : " + file)
    #                     failed_files.append(file)
    #         except Exception as e2:
    #             print("Failed initial gzip")
    #             failed_files.append(file)

    ## now we want to iterate through the PDBs and turn them into the proper format for arpeggio
    ## i have arpeggio in a different anaconda environment - hypothetically you can have it in the same environment but could cause issues
    ## i would honestly suggest just keeping it separate

    ## the PDBs need to be in the mmCIF format, which is done via a terminal command, so we're gonna use subprocess to do that
    ## you can make this into a bash script as well

    pdbs_dir = PISCES_DIR / "PDBs"
    pdbs_clean_dir = PISCES_DIR / "PDBs_cleaned"
    cifs_dir = PISCES_DIR / "CIFs"

    ### this clean command is available with pdb-tools which can be installed with "pip install pdb-tools"
    ### it was recommended on the arpeggio github to run this prior
    ### you can stop this at any time - note that if you keyboard exit out of terminal, it may still make the file but it will be empty
    ### simply look into where ever you are saving these files and delete the ones with nothing in it aka have a size of 0 KB
    # for pdb_file in Path(pdbs_dir).glob("*.pdb"):
    #     base_name = pdb_file.stem
    #     clean_name = pdbs_clean_dir+ "/"+base_name+"_clean.pdb"
    #     if os.path.exists(clean_name):
    #         print("Exists and skipping: " + base_name)
    #         continue
    #     else:
    #         print("Creating clean file for:" + base_name)
    #         subprocess.run("pdb_fetch "+ base_name + " > "+ clean_name, shell=True)
    
    ### after running the clean command, then we can turn everything into the mmCIF file format using gemmi
    # for pdb_file in Path(pdbs_clean_dir).glob("*.pdb"):
    #     base_name = pdb_file.stem
    #     cif_name = cifs_dir+ "/"+base_name+".cif"
    #     if os.path.exists(cif_name):
    #         print("Exists and skipping: " + base_name)
    #         continue
    #     else:
    #         print("Creating CIF file for:" + base_name)
    #         subprocess.run(["gemmi", "convert", str(pdb_file), str(cif_name)])
    
    ### and once we have the proper format, we can run arpeggio on all of these
    # out_dir = "/home/epi2melabs/projects/AcrNetworks/PISCES/JSONs"
    # failed_arp = []
    # for cif_file in Path(cifs_dir).glob("*.cif"):
    #     base_name = cif_file.stem
    #     try:
    #         print("Trying arpeggio")
    #         subprocess.run(["pdbe-arpeggio", str(cif_file), "-sa", "-co", "1.0", "-o", out_dir], check=True, timeout=1800)
    #     except Exception as e3:
    #         print("Failed arpeggio")
    #         failed_arp.append(base_name)
    # print(failed_arp)

    ######
    # First, clone the repository if you haven't already
    # if not os.path.exists("/home/epi2melabs/projects/pdbtools"):
    #     subprocess.run("git clone https://github.com/harryjubb/pdbtools.git /home/epi2melabs/projects/pdbtools", shell=True)

    known_dir = KNOWN_DIR
    save_dir = PISCES_DIR / "known acr data"
    # # Here, take the 3 Acrs we're mainly testing, turn them into the proper format for arpeggio
    # for pdb_file in Path(known_dir).glob("*.pdb"):
    #     base_name = pdb_file.stem
    #     print(base_name)
    #     clean_name = os.path.join(save_dir, f"{base_name}_clean.pdb")
        
    #     if os.path.exists(clean_name):
    #         print("Exists and skipping: " + base_name)
    #         continue
    #     else:
    #         print("Creating clean file for:" + base_name)
    #         source_file = str(pdb_file)
            
    #         # need clean_pdb.py file
    #         clean_pdb_path = "/home/epi2melabs/projects/pdbtools/clean_pdb.py"
            
    #         # Run with python explicitly to interpret the Python script
    #         result = subprocess.run(f"python3 {clean_pdb_path} -kh '{source_file}'", 
    #                             shell=True,
    #                             capture_output=True,
    #                             text=True)
            
    #         if result.returncode == 0:
    #             print("Clean process completed successfully")
    #         else:
    #             print(f"Error running clean_pdb.py: {result.stderr}")
                
    #         # The clean_pdb.py script creates a file named {base_name}.clean.pdb
    #         # We need to rename it to our desired output name
    #         source_dir = os.path.dirname(source_file)
    #         source_base = os.path.basename(os.path.splitext(source_file)[0])
    #         cleaned_output = os.path.join(source_dir, f"{source_base}.clean.pdb")
            
    #         if os.path.exists(cleaned_output):
    #             # Move the file to our desired location
    #             os.rename(cleaned_output, clean_name)
    #             print(f"Successfully cleaned {base_name}")
    #         else:
    #             print(f"Error: Could not find the cleaned output file {cleaned_output}")
    #             print(f"Available files in directory: {os.listdir(source_dir)}")

    # for pdb_file in Path(save_dir).glob("*.pdb"):
    #     base_name = pdb_file.stem
    #     cif_name = save_dir+ "/"+base_name+".cif"
    #     if os.path.exists(cif_name):
    #         print("Exists and skipping: " + base_name)
    #         continue
    #     else:
    #         print("Creating CIF file for:" + base_name)
    #         subprocess.run(["gemmi", "convert", str(pdb_file), str(cif_name)])

    # failed_arp = []
    # for cif_file in Path(save_dir).glob("*.cif"):
    #     base_name = cif_file.stem
    #     try:
    #         print("Trying arpeggio")
    #         subprocess.run(["pdbe-arpeggio", str(cif_file), "-sa", "-co", "1.0", "-o", save_dir], check=True, timeout=1800)
    #     except Exception as e3:
    #         print("Failed arpeggio")
    #         failed_arp.append(base_name)
    # print(failed_arp)

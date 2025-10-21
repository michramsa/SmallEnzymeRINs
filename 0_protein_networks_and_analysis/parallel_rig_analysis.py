# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013) Maximum Allowed Solvent Accessibilites of Residues in Proteins. PLoS ONE 8(11):
# e80635. doi:10.1371/journal.pone.0080635
max_sasa_theo_dict = {
    "ALA": 129,
    "ARG": 274,
    "ASN": 195,
    "ASP": 193,
    "CYS": 167,
    "GLN": 225,
    "GLU": 223,
    "GLY": 104,
    "HIS": 224,
    "ILE": 197,
    "LEU": 201,
    "LYS": 236,
    "MET": 224,
    "PHE": 240,
    "PRO": 159,
    "SER": 155,
    "THR": 172,
    "TRP": 285,
    "TYR": 263,
    "VAL": 174
}

max_sasa_emp_dict = {
    "ALA": 121,
    "ARG": 265,
    "ASN": 187,
    "ASP": 187,
    "CYS": 148,
    "GLN": 214,
    "GLU": 214,
    "GLY": 97,
    "HIS": 216,
    "ILE": 195,
    "LEU": 191,
    "LYS": 230,
    "MET": 203,
    "PHE": 228,
    "PRO": 154,
    "SER": 143,
    "THR": 163,
    "TRP": 264,
    "TYR": 255,
    "VAL": 165
}

# inputs: graphML directory, output: list containing Paths to graphMLs of interest for performing data analysis on
def size_sort(graphml_dir, out_dir, json_dir, max_size_kb):
    # check existence of an output directory, create if not there
    os.makedirs(out_dir, exist_ok=True)

    # sort all CIFs by size
    cif_files = list(Path(graphml_dir).glob("*.graphml"))
    cif_files = sorted(cif_files, key=lambda p: os.path.getsize(p))

    # check existing files already turned
    processed_files = set(f.stem for f in Path(out_dir).glob("*.csv"))
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

# inputs: graphML graph, a directory for the output CSV file, the file name associated with the graphML file
def calculate_centrality_measures(graph, output_dir, file_name):
    base_name = os.path.splitext(file_name)[0]
    base_name = base_name.split("_")[0]
    print(f"Processing centrality for {base_name}")
    
    # check for output dir, if not there, then create it
    os.makedirs(output_dir, exist_ok=True)
    
    # if file doesn't exist, save as a CSV
    path_name = Path(output_dir)
    path_name = path_name / f"{base_name}_centrality.csv"
    if not path_name.exists():
        # # calc centralities
        degree_centrality = nx.degree_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)
        
        # combine
        combined_data = []
        for node, attrs in graph.nodes(data=True):
            combined_data.append({
                "Node": node,
                "Residue": attrs['residue_type'],
                "Chain": attrs['chain'],
                "Residue number": attrs['residue_num'],
                "DegreeCentrality": degree_centrality.get(node, 0),
                "ClosenessCentrality": closeness_centrality.get(node, 0)
            })

        output_file = os.path.join(output_dir, f"{base_name}_centrality.csv")

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Node", "Residue", "Chain", "Residue number", "DegreeCentrality", "ClosenessCentrality"])
            writer.writeheader()
            writer.writerows(combined_data)
    else:
        print(f"{path_name} already exists!")
        
    return 0

def calculate_sasa(pdb_dir, file_name, sasa_output_dir):
    base_name = Path(os.path.splitext(file_name)[0])
    base_name = base_name.stem
    base_name = base_name.split("_")[0]
    print(f"Processing SASA for {base_name}")
    
    # check for pdb file
    pdb_file = None
    for pdb in os.listdir(pdb_dir):
        if pdb.startswith(base_name) and pdb.endswith('.pdb'):
            print(f"Found PDB for {base_name}")
            pdb_file = os.path.join(pdb_dir, pdb)
            break
    
    if not pdb_file:
        print(f"No PDB file found for {base_name}")
        return None
    
    # calculate SASA
    p = PDBParser(QUIET=1)
    try:
        struct = p.get_structure(base_name, pdb_file)
        sr = ShrakeRupley()
        sr.compute(struct, level="R")
        
        sasa_data = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        res_name = residue.resname
                        sasa_value = round(residue.sasa, 2)
                        
                        # Calculate RSA values
                        theo_max = max_sasa_theo_dict.get(res_name)
                        emp_max = max_sasa_emp_dict.get(res_name)
                        
                        if theo_max is None or emp_max is None:
                            rsa_theo = -1
                            rsa_emp = -1
                        else:
                            rsa_theo = sasa_value / theo_max
                            rsa_emp = sasa_value / emp_max
                        
                        sasa_data.append({
                            "Model": model.id,
                            "Chain": chain.id,
                            "Residue": res_name,
                            "Residue ID": residue.id[1],
                            "SASA (Å²)": sasa_value,
                            "RSA-theoretical": rsa_theo,
                            "RSA-empirical": rsa_emp
                        })
        
        os.makedirs(sasa_output_dir, exist_ok=True)
        
        output_file = os.path.join(sasa_output_dir, f"{base_name}_sasa.csv")
        df = pd.DataFrame(sasa_data)
        df.to_csv(output_file, index=False)
        
        return 0
    
    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        return None

# Function to process a single file (to be used in parallel)
def process_file(file, graphs_dir, pdb_dir, centrality_output_dir, sasa_output_dir):
    try:
        full_name = os.path.join(graphs_dir, file)
        print(f"Processing {full_name}")
        
        # open the graph
        graph = nx.read_graphml(full_name)
        print(f"Loaded graph: {file}")
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
        
        # calculate centralities and place in CSV file
        calculate_centrality_measures(graph, centrality_output_dir, file)
        
        # calculate SASA and place in CSV file
        calculate_sasa(pdb_dir, full_name, sasa_output_dir)
        
        return f"Completed processing {file}"
    except Exception as e:
        return f"Error processing {file}: {str(e)}"

def main():
    ## original run
    # graphs_dir = PISCES_DIR / "GraphMLs"
    # pdb_dir = PISCES_DIR / "PDBs"
    # sasa_output_dir = PISCES_DIR / "SASAs"
    # centrality_output_dir = PISCES_DIR / "Centralities"

    ## second run - 5/19/2025
    graphs_dir = IMP_DIR / "Cleaned RIGs"
    pdb_dir = IMP_DIR / "Cleaned PDBs"
    sasa_output_dir = IMP_DIR / "Cleaned SASAs"
    centrality_output_dir = IMP_DIR / "Cleaned Centralities"
    
    os.makedirs(centrality_output_dir, exist_ok=True)
    os.makedirs(sasa_output_dir, exist_ok=True)
    
    file_list = [f for f in os.listdir(graphs_dir) if f.endswith('.graphml')]
    
    process_file_partial = partial(
        process_file, 
        graphs_dir=graphs_dir, 
        pdb_dir=pdb_dir, 
        centrality_output_dir=centrality_output_dir, 
        sasa_output_dir=sasa_output_dir
    )
    
    num_cores = 4
    print(f"Starting parallel processing with {num_cores} cores")
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_file_partial, file_list)
    
    for result in results:
        print(result)
    
    print("All processing complete!")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
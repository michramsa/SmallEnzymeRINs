# The purpose of this file is to take the RIGs and perform some network analysis on them.
# The network analysis includes:
# - counts + degree and closeness centrality measures for the residues within each protein
# - taking these measurements to then use for comparison to the Acr protein rigs themselves
# In this case, we're going from GraphML files -> CSV files with our data of interest

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
    print(base_name)
    
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

# inputs: directory containing GraphML files, directory containing existing centrality CSVs
# Appends a "Degree" column to each centrality CSV based on node degrees from the corresponding GraphML
def append_degree_to_centrality(graphml_dir, centrality_dir):
    graphml_dir = Path(graphml_dir)
    centrality_dir = Path(centrality_dir)
    
    if not centrality_dir.exists():
        print(f"Centrality directory {centrality_dir} does not exist.")
        return 0
    
    # loop through GraphML files
    for graphml_file in graphml_dir.glob("*.graphml"):
        # match the base_name convention used in calculate_centrality_measures
        base_name = graphml_file.stem.split("_")[0]
        print(base_name)
        
        # find the corresponding centrality CSV
        csv_path = centrality_dir / f"{base_name}_centrality.csv"
        if not csv_path.exists():
            print(f"No centrality CSV found for {base_name}, skipping.")
            continue
        
        # load existing CSV
        df = pd.read_csv(csv_path)
        
        # skip if Degree column already exists
        if "Degree" in df.columns:
            print(f"{csv_path.name} already has a Degree column, skipping.")
            continue
        
        # load graph and compute degree
        graph = nx.read_graphml(graphml_file)
        degree_dict = dict(graph.degree())
        
        # map degree onto the Node column
        # cast Node column to str to ensure type match with GraphML node IDs
        df["Degree"] = df["Node"].astype(str).map(degree_dict)
        
        # check for unmatched nodes
        unmatched = df["Degree"].isna().sum()
        if unmatched > 0:
            print(f"Warning: {unmatched} nodes in {csv_path.name} did not match a degree value.")
        
        # write back
        df.to_csv(csv_path, index=False)
        print(f"Appended Degree column to {csv_path.name}")
    
    return 0

# inputs: directory containing CIF files, file name from graphml loop, output directory for SASA CSVs
# Calculates SASA and RSA from mmCIF files using BioPython MMCIFParser + ShrakeRupley
def calculate_sasa(cif_dir, file_name, sasa_output_dir):
    base_name = Path(os.path.splitext(file_name)[0])
    base_name = base_name.stem
    base_name = base_name.split("_")[0]
    print(base_name)
    
    # check for cif file
    cif_file = None
    for cif in os.listdir(cif_dir):
        if cif.startswith(base_name) and cif.endswith('.cif'):
            print("Found!")
            cif_file = os.path.join(cif_dir, cif)
            print(cif_file)
            break
    
    if not cif_file:
        print(f"No CIF file found for {base_name}")
        return None
    
    # Calculate SASA
    p = MMCIFParser(QUIET=True)
    try:
        struct = p.get_structure(base_name, cif_file)
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
        
        # Ensure the output directory exists
        os.makedirs(sasa_output_dir, exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join(sasa_output_dir, f"{base_name}_sasa.csv")
        df = pd.DataFrame(sasa_data)
        df.to_csv(output_file, index=False)
        
        return 0
    
    except Exception as e:
        print(f"Error processing CIF file {cif_file}: {e}")
        return None


# inputs: directory containing PDB files, file name from graphml loop, output directory for SASA CSVs
# Calculates SASA and RSA from PDB files using BioPython PDBParser + ShrakeRupley
# Logic is identical to calculate_sasa() — only the parser and file extension differ
def calculate_sasa_pdb(pdb_dir, file_name, sasa_output_dir):
    base_name = Path(os.path.splitext(file_name)[0])
    base_name = base_name.stem
    base_name = base_name.split("_")[0]
    print(base_name)

    # check for pdb file
    pdb_file = None
    for pdb in os.listdir(pdb_dir):
        if pdb.startswith(base_name) and pdb.endswith('.pdb'):
            print("Found!")
            pdb_file = os.path.join(pdb_dir, pdb)
            print(pdb_file)
            break

    if not pdb_file:
        print(f"No PDB file found for {base_name}")
        return None

    # Calculate SASA — PDBParser instead of MMCIFParser; everything else is identical
    p = PDBParser(QUIET=True)
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

        # Ensure the output directory exists
        os.makedirs(sasa_output_dir, exist_ok=True)

        # Save to CSV
        output_file = os.path.join(sasa_output_dir, f"{base_name}_sasa.csv")
        df = pd.DataFrame(sasa_data)
        df.to_csv(output_file, index=False)

        return 0

    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        return None


# inputs: path to a FASTA file of representative sequences
# outputs: set of uppercase 4-character PDB IDs (e.g. {"2A11", "4A1G", "6A3Z"})
def parse_rep_seq_fasta(fasta_path):
    rep_ids = set()
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            # header lines contain the sequence ID; first 4 characters are the PDB ID
            if line.startswith('>'):
                pdb_id = line[1:5].upper()
                rep_ids.add(pdb_id)
    return rep_ids

# So for every graph, we need to:
# 1) open it
# 2) degree and closeness centrality values on that graph
# 3) RSA calculations for those proteins using the RAW CIF/PDB files (NOT the cleaned ones used for Arpeggio)

def main():

    # 5/19/26 - for the BioLiP set - I did filter out the data leakage proteins out of these directories
    # Uses CIF files -> calculate_sasa()
    graphs_dir = DATA_DIR_2 / "RINs"
    cif_dir = DATA_DIR_2 / "enzymes_CIF_unflagged" / "CIFs"
    sasa_output_dir = DATA_DIR_2 / "SASAs"
    fasta_path = ENZYMES_DIR / "mmseq" / "runs" / "050626_enzymes_nr30_rep_seq.fasta"

    # parse representative sequence IDs from FASTA to determine which files to process
    rep_ids = parse_rep_seq_fasta(fasta_path)
    print(f"Found {len(rep_ids)} representative sequences in FASTA.")

    # loop to open graphs
    ### this gets stuck for some reason, i'm guessing there is no catch for a missing graph potentially, not sure
    for file in os.listdir(graphs_dir):
        # extract the PDB ID from the filename (e.g. "2A11_clean.graphml" -> "2A11")
        pdb_id = file.split("_")[0].upper()

        # only process files whose PDB ID appears in the representative sequences - should be 449
        if pdb_id not in rep_ids:
            continue

        full_name = graphs_dir / file
        print(full_name)
        # open the graph
        graph = nx.read_graphml(full_name)
        print(f"Loaded graph: {file}")
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
        print("---")
        
        # calculate centralities and place in CSV file
        calculate_centrality_measures(graph, DATA_DIR_2 / "Centralities", file)
        # calculate SASA and place in CSV file
        calculate_sasa(cif_dir, full_name, sasa_output_dir)
    
    append_degree_to_centrality(
        DATA_DIR_2 / "RINs",
        DATA_DIR_2 / "Centralities"
    )

    # for the old PISCES set
    # Uses PDB files -> calculate_sasa_pdb()
    graphs_dir = DATA_DIR_2 / "old_PISCES" / "GraphMLs_filtered_v2"
    pdb_dir = DATA_DIR_2 / "old_PISCES" / "PDBs"  ## folder holding all PDB files
    sasa_output_dir = DATA_DIR_2 / "old_PISCES" / "SASAs"

    # loop to open graphs
    for file in os.listdir(graphs_dir):
        # extract the PDB ID from the filename (e.g. "2A11_clean.graphml" -> "2A11")
        pdb_id = file.split("_")[0].upper()

        full_name = graphs_dir / file
        print(full_name)
        # open the graph
        graph = nx.read_graphml(full_name)
        print(f"Loaded graph: {file}")
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
        print("---")
        
        # calculate centralities and place in CSV file
        calculate_centrality_measures(graph, DATA_DIR_2 / "old_PISCES" / "Centralities", file)
        # calculate SASA from PDB files and place in CSV file
        calculate_sasa_pdb(pdb_dir, full_name, sasa_output_dir)

    append_degree_to_centrality(
        DATA_DIR_2 / "old_PISCES" / "GraphMLs_filtered_v2",
        DATA_DIR_2 / "old_PISCES" / "Centralities"
    )

if __name__ == "__main__":
    main()

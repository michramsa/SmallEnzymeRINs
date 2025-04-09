# The purpose of this file is to take the RIGs and perform some network analysis on them.
# The network analysis includes:
# - counts + degree and closeness centrality measures for the residues within each protein
# - taking these measurements to then use for comparison to the Acr protein rigs themselves
# In this case, we're going from GraphML files -> CSV files with our data of interest

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

# So for every graph, we need to:
# 1) open it
# 2) degree and closeness centrality values on that graph
# 3) RSA calculations for those proteins using the RAW PDB files (NOT the cleaned ones used for Arpeggio)



if __name__ == "__main__":
    main()
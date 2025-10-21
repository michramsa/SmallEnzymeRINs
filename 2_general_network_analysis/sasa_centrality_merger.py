import pandas as pd
import os
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

# Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013) Maximum Allowed Solvent Accessibilites of Residues in Proteins. PLoS ONE 8(11):
# e80635. doi:10.1371/journal.pone.0080635
max_sasa_theo_dict = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174
}

max_sasa_emp_dict = {
    "ALA": 121, "ARG": 265, "ASN": 187, "ASP": 187, "CYS": 148,
    "GLN": 214, "GLU": 214, "GLY": 97, "HIS": 216, "ILE": 195,
    "LEU": 191, "LYS": 230, "MET": 203, "PHE": 228, "PRO": 154,
    "SER": 143, "THR": 163, "TRP": 264, "TYR": 255, "VAL": 165
}

def identify_molecule_type(residue_name):
    """
    Identify if a residue is protein, RNA, or DNA based on residue name.
    
    Parameters:
    -----------
    residue_name : str
        3-letter residue code
    
    Returns:
    --------
    str : 'Protein', 'RNA', 'DNA', or 'Other'
    """
    # Standard amino acids
    amino_acids = set(max_sasa_theo_dict.keys())
    
    # Common RNA nucleotides
    rna_nucleotides = {'A', 'U', 'G', 'C', 'AMP', 'UMP', 'GMP', 'CMP'}
    
    # Common DNA nucleotides  
    dna_nucleotides = {'DA', 'DT', 'DG', 'DC', 'DMP', 'TMP', 'GDP', 'CDP'}
    
    if residue_name in amino_acids:
        return 'Protein'
    elif residue_name in rna_nucleotides:
        return 'RNA'
    elif residue_name in dna_nucleotides:
        return 'DNA'
    else:
        return 'Other'

def calculate_sasa_for_centrality(pdb_file, centrality_csv):
    """
    Calculate SASA values for residues in a PDB file and match them to centrality data.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file
    centrality_csv : str
        Path to the centrality CSV file
    
    Returns:
    --------
    pd.DataFrame : Merged dataframe with centrality and SASA data
    """
    
    # Load centrality data
    print(f"Loading centrality data from: {centrality_csv}")
    centrality_df = pd.read_csv(centrality_csv)
    
    # Add molecule type column based on residue names
    centrality_df['Molecule_Type'] = centrality_df['Residue'].apply(identify_molecule_type)
    
    # Parse PDB structure
    print(f"Loading PDB structure from: {pdb_file}")
    parser = PDBParser(QUIET=True)
    base_name = Path(pdb_file).stem
    structure = parser.get_structure(base_name, pdb_file)
    
    # Calculate SASA for entire complex
    print("Calculating SASA for entire complex...")
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    
    # Extract SASA data for protein residues only
    sasa_data = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip if no CA atom (not a standard residue)
                if "CA" not in residue:
                    continue
                    
                res_name = residue.resname
                chain_id = chain.id
                res_num = residue.id[1]
                
                # Only process protein residues for SASA
                if identify_molecule_type(res_name) == 'Protein':
                    sasa_value = round(residue.sasa, 2)
                    
                    # Calculate RSA values
                    theo_max = max_sasa_theo_dict.get(res_name)
                    emp_max = max_sasa_emp_dict.get(res_name)
                    
                    if theo_max is not None and emp_max is not None:
                        rsa_theo = round(sasa_value / theo_max, 4)
                        rsa_emp = round(sasa_value / emp_max, 4)
                    else:
                        rsa_theo = None
                        rsa_emp = None
                    
                    # Create matching key
                    node_key = f"{chain_id}_{res_num}"
                    
                    sasa_data.append({
                        "Node": node_key,
                        "Chain": chain_id,
                        "Residue": res_name,
                        "Residue_number": res_num,
                        "SASA_A2": sasa_value,
                        "RSA_theoretical": rsa_theo,
                        "RSA_empirical": rsa_emp
                    })
    
    # Convert to DataFrame
    sasa_df = pd.DataFrame(sasa_data)
    print(f"Calculated SASA for {len(sasa_df)} protein residues")
    
    # Merge with centrality data
    print("Merging SASA data with centrality data...")
    merged_df = centrality_df.merge(sasa_df[['Node', 'SASA_A2', 'RSA_theoretical', 'RSA_empirical']], 
                                   on='Node', how='left')
    
    # Print summary statistics
    protein_residues = merged_df[merged_df['Molecule_Type'] == 'Protein']
    non_protein = merged_df[merged_df['Molecule_Type'] != 'Protein']
    
    print(f"\nMerge Summary:")
    print(f"Total nodes in centrality data: {len(merged_df)}")
    print(f"Protein residues: {len(protein_residues)}")
    print(f"Non-protein residues: {len(non_protein)}")
    print(f"Protein residues with SASA data: {len(protein_residues.dropna(subset=['SASA_A2']))}")
    print(f"Protein residues missing SASA data: {len(protein_residues[protein_residues['SASA_A2'].isna()])}")
    
    # Show molecule type breakdown
    print(f"\nMolecule type breakdown:")
    type_counts = merged_df['Molecule_Type'].value_counts()
    for mol_type, count in type_counts.items():
        print(f"  {mol_type}: {count}")
    
    return merged_df

def save_merged_data(merged_df, output_path):
    """
    Save the merged dataframe to CSV.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with centrality and SASA data
    output_path : str
        Path where to save the merged CSV file
    """
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged data saved to: {output_path}")
    
    # Show first few rows as preview
    print("\nPreview of merged data:")
    print(merged_df.head())

def calculate_sasa_for_centrality_nmr(pdb_file, centrality_csv, model_choice='first'):
    """
    Calculate SASA values for NMR structures with multiple models and match them to centrality data.
    
    Parameters:
    -----------
    pdb_file : str
        Path to the PDB file (potentially multi-model NMR structure)
    centrality_csv : str
        Path to the centrality CSV file
    model_choice : str
        How to handle multiple models:
        - 'first': Use only the first model (default)
        - 'average': Average SASA across all models
        - 'best': Use model with highest overall quality (if available)
    
    Returns:
    --------
    pd.DataFrame : Merged dataframe with centrality and SASA data
    """
    
    # Load centrality data
    print(f"Loading centrality data from: {centrality_csv}")
    centrality_df = pd.read_csv(centrality_csv)
    
    # Add molecule type column based on residue names
    centrality_df['Molecule_Type'] = centrality_df['Residue'].apply(identify_molecule_type)
    
    # Parse PDB structure
    print(f"Loading PDB structure from: {pdb_file}")
    parser = PDBParser(QUIET=True)
    base_name = Path(pdb_file).stem
    structure = parser.get_structure(base_name, pdb_file)
    
    # Check number of models
    models = list(structure.get_models())
    print(f"Found {len(models)} model(s) in PDB file")
    
    if len(models) == 1:
        print("Single model detected - using standard processing")
        return calculate_sasa_for_centrality(pdb_file, centrality_csv)
    
    # Handle multi-model structures
    print(f"Multi-model structure detected - using '{model_choice}' strategy")
    
    if model_choice == 'first':
        # Use only the first model
        selected_model = models[0]
        print(f"Using model {selected_model.id} (first model)")
        
        # Create a new structure with only the first model
        from Bio.PDB.Structure import Structure
        single_model_structure = Structure(structure.id)
        single_model_structure.add(selected_model.copy())
        
        # Calculate SASA for single model
        sr = ShrakeRupley()
        sr.compute(single_model_structure, level="R")
        
        # Extract SASA data for protein residues only
        sasa_data = []
        for model in single_model_structure:
            for chain in model:
                for residue in chain:
                    # Skip if no CA atom
                    if "CA" not in residue:
                        continue
                        
                    res_name = residue.resname
                    chain_id = chain.id
                    res_num = residue.id[1]
                    
                    # Only process protein residues for SASA
                    if identify_molecule_type(res_name) == 'Protein':
                        sasa_value = round(residue.sasa, 2)
                        
                        # Calculate RSA values
                        theo_max = max_sasa_theo_dict.get(res_name)
                        emp_max = max_sasa_emp_dict.get(res_name)
                        
                        if theo_max is not None and emp_max is not None:
                            rsa_theo = round(sasa_value / theo_max, 4)
                            rsa_emp = round(sasa_value / emp_max, 4)
                        else:
                            rsa_theo = None
                            rsa_emp = None
                        
                        # Create matching key
                        node_key = f"{chain_id}_{res_num}"
                        
                        sasa_data.append({
                            "Node": node_key,
                            "Chain": chain_id,
                            "Residue": res_name,
                            "Residue_number": res_num,
                            "SASA_A2": sasa_value,
                            "RSA_theoretical": rsa_theo,
                            "RSA_empirical": rsa_emp,
                            "Model_Used": selected_model.id
                        })
    
    elif model_choice == 'average':
        # Calculate SASA for all models and average
        print("Calculating SASA for all models and averaging...")
        
        all_sasa_data = {}  # Will store {node_key: [sasa_values_across_models]}
        
        for model_idx, model in enumerate(models):
            print(f"Processing model {model.id}...")
            
            # Create structure with single model
            from Bio.PDB.Structure import Structure
            temp_structure = Structure(structure.id)
            temp_structure.add(model.copy())
            
            # Calculate SASA
            sr = ShrakeRupley()
            sr.compute(temp_structure, level="R")
            
            # Extract SASA data
            for chain in temp_structure[0]:  # Only one model in temp_structure
                for residue in chain:
                    if "CA" not in residue:
                        continue
                        
                    res_name = residue.resname
                    chain_id = chain.id
                    res_num = residue.id[1]
                    
                    if identify_molecule_type(res_name) == 'Protein':
                        node_key = f"{chain_id}_{res_num}"
                        sasa_value = residue.sasa
                        
                        if node_key not in all_sasa_data:
                            all_sasa_data[node_key] = {
                                'sasa_values': [],
                                'chain': chain_id,
                                'residue': res_name,
                                'res_num': res_num
                            }
                        
                        all_sasa_data[node_key]['sasa_values'].append(sasa_value)
        
        # Average SASA values across models
        sasa_data = []
        for node_key, data in all_sasa_data.items():
            avg_sasa = round(np.mean(data['sasa_values']), 2)
            std_sasa = round(np.std(data['sasa_values']), 2)
            
            res_name = data['residue']
            theo_max = max_sasa_theo_dict.get(res_name)
            emp_max = max_sasa_emp_dict.get(res_name)
            
            if theo_max is not None and emp_max is not None:
                rsa_theo = round(avg_sasa / theo_max, 4)
                rsa_emp = round(avg_sasa / emp_max, 4)
            else:
                rsa_theo = None
                rsa_emp = None
            
            sasa_data.append({
                "Node": node_key,
                "Chain": data['chain'],
                "Residue": res_name,
                "Residue_number": data['res_num'],
                "SASA_A2": avg_sasa,
                "RSA_theoretical": rsa_theo,
                "RSA_empirical": rsa_emp,
                "SASA_StdDev": std_sasa,
                "Models_Averaged": len(data['sasa_values'])
            })
    
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}. Use 'first' or 'average'")
    
    # Convert to DataFrame
    sasa_df = pd.DataFrame(sasa_data)
    print(f"Calculated SASA for {len(sasa_df)} protein residues")
    
    # Merge with centrality data
    print("Merging SASA data with centrality data...")
    
    # Define merge columns based on model_choice
    if model_choice == 'first':
        merge_columns = ['Node', 'SASA_A2', 'RSA_theoretical', 'RSA_empirical', 'Model_Used']
    else:  # average
        merge_columns = ['Node', 'SASA_A2', 'RSA_theoretical', 'RSA_empirical', 'SASA_StdDev', 'Models_Averaged']
    
    merged_df = centrality_df.merge(sasa_df[merge_columns], on='Node', how='left')
    
    # Print summary statistics
    protein_residues = merged_df[merged_df['Molecule_Type'] == 'Protein']
    non_protein = merged_df[merged_df['Molecule_Type'] != 'Protein']
    
    print(f"\nMerge Summary:")
    print(f"Total nodes in centrality data: {len(merged_df)}")
    print(f"Protein residues: {len(protein_residues)}")
    print(f"Non-protein residues: {len(non_protein)}")
    print(f"Protein residues with SASA data: {len(protein_residues.dropna(subset=['SASA_A2']))}")
    print(f"Protein residues missing SASA data: {len(protein_residues[protein_residues['SASA_A2'].isna()])}")
    
    if model_choice == 'average':
        print(f"Average SASA standard deviation: {merged_df['SASA_StdDev'].mean():.3f}")
    
    # Show molecule type breakdown
    print(f"\nMolecule type breakdown:")
    type_counts = merged_df['Molecule_Type'].value_counts()
    for mol_type, count in type_counts.items():
        print(f"  {mol_type}: {count}")
    
    return merged_df

def process_centrality_sasa_batch(centrality_files, pdb_dir, output_dir, nmr_files=None):
    """
    Process multiple centrality files and match them with corresponding PDB files.
    Handles both single-model and multi-model (NMR) structures.
    
    Parameters:
    -----------
    centrality_files : list
        List of paths to centrality CSV files
    pdb_dir : str
        Directory containing PDB files
    output_dir : str
        Directory to save merged CSV files
    nmr_files : list, optional
        List of base names (e.g., ['2LW5']) that are NMR structures requiring special handling
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if nmr_files is None:
        nmr_files = []
    
    for centrality_file in centrality_files:
        print(f"\n{'='*60}")
        print(f"Processing: {Path(centrality_file).name}")
        print(f"{'='*60}")
        
        # Extract base name to find corresponding PDB
        base_name = Path(centrality_file).stem.split('_')[0]  # Get first part before underscore
        
        # Find corresponding PDB file
        pdb_file = None
        for pdb in os.listdir(pdb_dir):
            if pdb.startswith(base_name) and pdb.endswith('.pdb'):
                pdb_file = os.path.join(pdb_dir, pdb)
                break
        
        if not pdb_file:
            print(f"Warning: No PDB file found for {base_name}")
            continue
        
        try:
            # Choose processing method based on whether it's an NMR structure
            if base_name in nmr_files:
                print(f"Processing {base_name} as NMR structure")
                merged_df = calculate_sasa_for_centrality_nmr(pdb_file, centrality_file, model_choice='first')
            else:
                print(f"Processing {base_name} as standard structure")
                merged_df = calculate_sasa_for_centrality(pdb_file, centrality_file)
            
            # Save merged data
            output_filename = f"{base_name}_centrality_with_sasa.csv"
            output_path = os.path.join(output_dir, output_filename)
            save_merged_data(merged_df, output_path)
            
        except Exception as e:
            print(f"Error processing {centrality_file}: {e}")
            continue

# Example usage
if __name__ == "__main__":
    
    print("Example 1: NMR structure processing")
    print("-" * 40)
    
    nmr_centrality_file = "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/2LW5_RIG_centrality_standardized.csv"
    nmr_pdb_file = "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr PDB files/2LW5.pdb"  
    nmr_output_file = "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/2LW5_centrality_with_sasa_nmr.csv"
    
    # Process NMR file (using first model)
    merged_data = calculate_sasa_for_centrality_nmr(nmr_pdb_file, nmr_centrality_file, model_choice='first')
    save_merged_data(merged_data, nmr_output_file)
    
    # Alternative: use averaging across models
    # merged_data_avg = calculate_sasa_for_centrality_nmr(nmr_pdb_file, nmr_centrality_file, model_choice='average')
    # save_merged_data(merged_data_avg, "/path/to/output/2LW5_centrality_with_sasa_averaged.csv")
    
    
    # # Example 2: Batch processing with mixed structure types
    # print("\nExample 2: Batch processing with NMR detection")
    # print("-" * 40)
    
    # # List of centrality files
    # centrality_files = [
    #     "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/2LW5_RIG_centrality_standardized.csv",
    #     "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/6CB8_RIG_centrality_standardized.csv",
    #     "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/3QNR_RIG_centrality_standardized.csv"
    # ]
    
    # pdb_directory = "/path/to/pdb/files/"  # Update this path
    # output_directory = "/path/to/output/"  # Update this path
    
    # # Specify which files are NMR structures
    # nmr_structure_names = ['2LW5']  # Add other NMR structure names if you have them
    
    # # Process all files with automatic NMR detection
    # # process_centrality_sasa_batch(centrality_files, pdb_directory, output_directory, nmr_files=nmr_structure_names)
    
    
    # # Example 3: Quick test with your NMR data
    # print("\nTo run with your actual NMR data, update the paths and uncomment:")
    # print("merged_data = calculate_sasa_for_centrality_nmr(")
    # print("    pdb_file='/your/pdb/path/2LW5.pdb',")
    # print("    centrality_csv='/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/2LW5_RIG_centrality_standardized.csv',")
    # print("    model_choice='first'")
    # print(")")
    # print("save_merged_data(merged_data, '/your/output/path/2LW5_merged_nmr.csv')")
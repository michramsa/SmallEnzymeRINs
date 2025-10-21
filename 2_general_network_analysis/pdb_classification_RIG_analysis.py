from imports import *

# Dictionary mapping 3-letter amino acid codes to full names
amino_acid_names = {
    'ALA': 'Alanine',
    'ARG': 'Arginine',
    'ASN': 'Asparagine',
    'ASP': 'Aspartic acid',
    'CYS': 'Cysteine',
    'GLN': 'Glutamine',
    'GLU': 'Glutamic acid',
    'GLY': 'Glycine',
    'HIS': 'Histidine',
    'ILE': 'Isoleucine',
    'LEU': 'Leucine',
    'LYS': 'Lysine',
    'MET': 'Methionine',
    'PHE': 'Phenylalanine',
    'PRO': 'Proline',
    'SER': 'Serine',
    'THR': 'Threonine',
    'TRP': 'Tryptophan',
    'TYR': 'Tyrosine',
    'VAL': 'Valine'
}

# List of valid 3-letter amino acid codes
valid_amino_acids = list(amino_acid_names.keys())

def load_classifications(csv_path):
    """Load the PDB ID to classification mapping from CSV"""
    try:
        print(f"Loading classifications from: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"ERROR: Classification file not found: {csv_path}")
            return {}
            
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} entries and columns: {df.columns.tolist()}")
        
        # Assumes first column is PDB ID and second column is classification
        # Get column names for clarity
        pdb_col = df.columns[0]
        class_col = df.columns[1]
        
        print(f"Using columns: PDB ID = '{pdb_col}', Classification = '{class_col}'")
        
        # Count each classification
        class_counts = {}
        for cls in df[class_col]:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Filter for classifications with at least 10 entries
        valid_classes = {cls: count for cls, count in class_counts.items() if count >= 10}
        print(f"Found {len(valid_classes)} classifications with at least 10 entries")
        
        # Create a dictionary of classification -> PDB IDs
        class_to_pdbs = {}
        for pdb, cls in zip(df[pdb_col], df[class_col]):
            if cls in valid_classes:
                if cls not in class_to_pdbs:
                    class_to_pdbs[cls] = []
                class_to_pdbs[cls].append(pdb)
        
        print(f"Created {len(class_to_pdbs)} classification to PDB mappings")
        return class_to_pdbs
    except Exception as e:
        print(f"Error loading classifications: {e}")
        import traceback
        traceback.print_exc()
        return {}

def find_centrality_files(pdb_ids, centrality_dir):
    """Find centrality CSV files for a list of PDB IDs"""
    if not os.path.exists(centrality_dir):
        print(f"ERROR: Centrality directory not found: {centrality_dir}")
        return {}
        
    print(f"Searching for centrality CSV files in: {centrality_dir}")
    
    # Get all CSV files in the directory to understand naming pattern
    all_csvs = glob.glob(os.path.join(centrality_dir, "*.csv"))
    print(f"Found {len(all_csvs)} total CSV files in directory")
    if all_csvs:
        print(f"Sample filenames: {all_csvs[:3]}")
    
    # Create mapping of PDB ID to CSV file
    centrality_files = {}
    for pdb_id in pdb_ids:
        # Look for files containing the PDB ID
        potential_files = glob.glob(os.path.join(centrality_dir, f"*{pdb_id}*.csv"))
        
        if potential_files:
            centrality_files[pdb_id] = potential_files[0]  # Take the first matching file
    
    print(f"Found matching centrality files for {len(centrality_files)}/{len(pdb_ids)} PDB IDs")
    return centrality_files

def process_centrality_by_classification(class_to_pdbs, centrality_dir):
    """Process centrality CSV files grouped by classification"""
    classification_results = {}
    centrality_files_by_class = {}
    
    for classification, pdb_ids in class_to_pdbs.items():
        print(f"\nProcessing classification: {classification} with {len(pdb_ids)} PDB entries")
        
        # Find centrality files for this classification's PDB IDs
        centrality_files = find_centrality_files(pdb_ids, centrality_dir)
        
        # Store the centrality files for this classification
        centrality_files_by_class[classification] = centrality_files
        
        if not centrality_files:
            print(f"  No centrality files found for classification: {classification}")
            continue
            
        print(f"  Found {len(centrality_files)} centrality files for classification: {classification}")
        
        # Initialize data structure to collect values for each amino acid
        amino_acid_data = {aa: {
            'degree_values': [],
            'closeness_values': []
        } for aa in valid_amino_acids}
        
        # Process each centrality file
        files_processed = 0
        for pdb_id, file_path in centrality_files.items():
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Check if expected columns exist
                required_cols = ['Residue', 'DegreeCentrality', 'ClosenessCentrality']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"  Warning: File {file_path} is missing columns: {missing_cols}")
                    continue
                
                # Convert residue names to uppercase and filter for valid amino acids
                df['Residue'] = df['Residue'].str.upper()
                filtered_df = df[df['Residue'].isin(valid_amino_acids)]
                
                # Group by residue type and collect centrality values
                for aa in valid_amino_acids:
                    aa_df = filtered_df[filtered_df['Residue'] == aa]
                    if not aa_df.empty:
                        amino_acid_data[aa]['degree_values'].extend(aa_df['DegreeCentrality'].tolist())
                        amino_acid_data[aa]['closeness_values'].extend(aa_df['ClosenessCentrality'].tolist())
                
                files_processed += 1
                
            except Exception as e:
                print(f"  Error processing file {file_path} for PDB {pdb_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  Successfully processed {files_processed}/{len(centrality_files)} files for {classification}")
        
        # Calculate statistics for each amino acid in this classification
        amino_acid_stats = []
        for aa, data in amino_acid_data.items():
            if len(data['degree_values']) > 0:
                # Calculate statistics
                avg_degree = np.mean(data['degree_values'])
                std_degree = np.std(data['degree_values'], ddof=1)
                avg_closeness = np.mean(data['closeness_values'])
                std_closeness = np.std(data['closeness_values'], ddof=1)
                count = len(data['degree_values'])
                
                amino_acid_stats.append({
                    'amino_acid': aa,
                    'full_name': amino_acid_names[aa],
                    'avg_degree_centrality': avg_degree,
                    'std_degree_centrality': std_degree,
                    'avg_closeness_centrality': avg_closeness,
                    'std_closeness_centrality': std_closeness,
                    'count': count
                })
        
        # Sort by amino acid code
        amino_acid_stats.sort(key=lambda x: x['amino_acid'])
        
        # Store results if we have data
        if amino_acid_stats:
            classification_results[classification] = amino_acid_stats
            print(f"  Added results for {classification} with data for {len(amino_acid_stats)} amino acids")
        else:
            print(f"  No amino acid data found for classification {classification}")
    
    print(f"\nTotal classifications with results: {len(classification_results)}")
    return classification_results, centrality_files_by_class

def write_classification_files(results, output_dir, class_to_pdbs, centrality_files_by_class):
    """Write a text file for each classification's amino acid statistics"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Check if results dictionary is empty
        if not results:
            print("No results to write. Results dictionary is empty.")
            return
        
        print(f"Writing results for {len(results)} classifications")
        
        # Process each classification
        for classification, data in results.items():
            try:
                # Create safe filename from classification
                safe_filename = ''.join(c if c.isalnum() else '_' for c in classification)
                output_path = os.path.join(output_dir, f"{safe_filename}.txt")
                
                # Write formatted text file
                with open(output_path, 'w') as f:
                    f.write(f"Amino Acid Analysis Results for {classification}\n")
                    f.write("==========================\n\n")
                    
                    # Write classification metadata
                    total_pdbs = len(class_to_pdbs.get(classification, []))
                    pdbs_found = len(centrality_files_by_class.get(classification, {}))
                    
                    f.write(f"Total PDB entries in classification: {total_pdbs}\n")
                    f.write(f"PDB entries with centrality data: {pdbs_found}\n\n")
                    
                    # Write list of PDB IDs used
                    f.write("PDB IDs used in this analysis:\n")
                    if classification in centrality_files_by_class:
                        pdb_ids = list(centrality_files_by_class[classification].keys())
                        # Format PDB IDs in rows of 10
                        for i in range(0, len(pdb_ids), 10):
                            f.write("  " + ", ".join(pdb_ids[i:i+10]) + "\n")
                    else:
                        f.write("  None\n")
                    f.write("\n")
                    
                    # Write header for amino acid statistics
                    f.write(f"{'Full Name':<20} {'Code':<5} {'Avg Degree':<12} {'StdDev Degree':<15} {'Avg Closeness':<15} {'StdDev Closeness':<15} {'Count':<10}\n")
                    f.write("-" * 95 + "\n")
                    
                    # Write data for each amino acid
                    for aa_stats in data:
                        f.write(f"{aa_stats['full_name']:<20} {aa_stats['amino_acid']:<5} "
                               f"{aa_stats['avg_degree_centrality']:12.6f} {aa_stats['std_degree_centrality']:15.6f} "
                               f"{aa_stats['avg_closeness_centrality']:15.6f} {aa_stats['std_closeness_centrality']:15.6f} "
                               f"{aa_stats['count']:<10}\n")
                
                print(f"  Successfully wrote statistics for {classification} to {output_path}")
                
                # Also write as CSV for easy data analysis
                csv_path = os.path.join(output_dir, f"{safe_filename}.csv")
                pd.DataFrame(data).to_csv(csv_path, index=False)
                print(f"  Also wrote CSV data to {csv_path}")
                
            except Exception as e:
                print(f"  Error writing results for {classification}: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error in write_classification_files: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Path to your PDB classifications CSV
    classifications_csv = "pdb_classifications_altered.csv"
    
    # Directory containing centrality CSV files
    centrality_dir = "/home/epi2melabs/projects/AcrNetworks/Important Files/Cleaned Centralities"
    
    # Output directory for classification statistics
    output_dir = "classification_statistics"
    
    print("Loading classifications...")
    class_to_pdbs = load_classifications(classifications_csv)
    
    if not class_to_pdbs:
        print("No valid classifications found with at least 10 entries. Exiting.")
        return
        
    print(f"Found {len(class_to_pdbs)} classifications with at least 10 entries.")
    
    print("Processing centrality files by classification...")
    classification_results, centrality_files_by_class = process_centrality_by_classification(class_to_pdbs, centrality_dir)
    
    print("Writing results to text files...")
    write_classification_files(classification_results, output_dir, class_to_pdbs, centrality_files_by_class)
    
    print("Done!")

if __name__ == "__main__":
    main()
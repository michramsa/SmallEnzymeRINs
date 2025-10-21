# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

def process_csv_files(directory_path):
    # Dictionary to store accumulated values and all individual values for each amino acid
    amino_acid_data = {aa: {
        'degree_values': [],
        'closeness_values': []
    } for aa in valid_amino_acids}
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    for file_path in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Filter for valid amino acids
            df['Residue'] = df['Residue'].str.upper()  # Convert to uppercase for consistency
            filtered_df = df[df['Residue'].isin(valid_amino_acids)]
            
            # Append individual values for each amino acid
            for aa in valid_amino_acids:
                aa_df = filtered_df[filtered_df['Residue'] == aa]
                if not aa_df.empty:
                    amino_acid_data[aa]['degree_values'].extend(aa_df['DegreeCentrality'].tolist())
                    amino_acid_data[aa]['closeness_values'].extend(aa_df['ClosenessCentrality'].tolist())
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return amino_acid_data

def calculate_statistics(amino_acid_data):
    # Calculate statistics for each amino acid
    results = []
    
    for aa, data in amino_acid_data.items():
        if len(data['degree_values']) > 0:
            # Calculate mean
            avg_degree = np.mean(data['degree_values'])
            avg_closeness = np.mean(data['closeness_values'])
            
            # Calculate standard deviation
            std_degree = np.std(data['degree_values'], ddof=1)  # ddof=1 for sample standard deviation
            std_closeness = np.std(data['closeness_values'], ddof=1)
            
            results.append({
                'full_name': amino_acid_names[aa],
                'code': aa,
                'avg_degree_centrality': avg_degree,
                'std_degree_centrality': std_degree,
                'avg_closeness_centrality': avg_closeness,
                'std_closeness_centrality': std_closeness,
                'count': len(data['degree_values'])
            })
    
    # Sort by amino acid code
    results.sort(key=lambda x: x['code'])
    return results

def write_results_to_file(results, output_file):
    with open(output_file, 'w') as f:
        f.write("Amino Acid Analysis Results\n")
        f.write("==========================\n\n")
        
        # Write header
        f.write(f"{'Full Name':<20} {'Code':<5} {'Avg Degree':<12} {'StdDev Degree':<15} {'Avg Closeness':<15} {'StdDev Closeness':<15} {'Count':<10}\n")
        f.write("-" * 95 + "\n")
        
        # Write data
        for r in results:
            f.write(f"{r['full_name']:<20} {r['code']:<5} "
                   f"{r['avg_degree_centrality']:12.6f} {r['std_degree_centrality']:15.6f} "
                   f"{r['avg_closeness_centrality']:15.6f} {r['std_closeness_centrality']:15.6f} "
                   f"{r['count']:<10}\n")

def main():
    # Directory containing CSV files - replace with your directory path
    directory_path = IMP_DIR / "Cleaned Centralities"  # Current directory, change as needed
    output_file = "05192025_amino_acid_centrality_analysis.txt" # change as needed
    
    print(f"Processing CSV files in {directory_path}...")
    amino_acid_data = process_csv_files(directory_path)
    
    print("Calculating statistics...")
    results = calculate_statistics(amino_acid_data)
    
    print(f"Writing results to {output_file}...")
    write_results_to_file(results, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
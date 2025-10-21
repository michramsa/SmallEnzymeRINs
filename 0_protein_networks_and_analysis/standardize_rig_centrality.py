# creates the standardization based on the amino acid statistics
# uses values sourced from 

from imports import *
# Define the reference amino acid statistics
aa_stats = {
    'ALA': {'mean_degree': 0.038135, 'std_degree': 0.023368, 'mean_closeness': 0.222118, 'std_closeness': 0.057134},
    'ARG': {'mean_degree': 0.042682, 'std_degree': 0.026054, 'mean_closeness': 0.224645, 'std_closeness': 0.057626},
    'ASN': {'mean_degree': 0.036143, 'std_degree': 0.022526, 'mean_closeness': 0.217900, 'std_closeness': 0.055903},
    'ASP': {'mean_degree': 0.034368, 'std_degree': 0.021318, 'mean_closeness': 0.215629, 'std_closeness': 0.054589},
    'CYS': {'mean_degree': 0.052503, 'std_degree': 0.037489, 'mean_closeness': 0.239511, 'std_closeness': 0.067734},
    'GLN': {'mean_degree': 0.039120, 'std_degree': 0.024660, 'mean_closeness': 0.218478, 'std_closeness': 0.058151},
    'GLU': {'mean_degree': 0.036145, 'std_degree': 0.023315, 'mean_closeness': 0.214700, 'std_closeness': 0.056377},
    'GLY': {'mean_degree': 0.032332, 'std_degree': 0.019853, 'mean_closeness': 0.216836, 'std_closeness': 0.054469},
    'HIS': {'mean_degree': 0.041497, 'std_degree': 0.025530, 'mean_closeness': 0.227178, 'std_closeness': 0.059724},
    'ILE': {'mean_degree': 0.048374, 'std_degree': 0.028319, 'mean_closeness': 0.235073, 'std_closeness': 0.063210},
    'LEU': {'mean_degree': 0.048398, 'std_degree': 0.028406, 'mean_closeness': 0.233644, 'std_closeness': 0.062289},
    'LYS': {'mean_degree': 0.037657, 'std_degree': 0.024022, 'mean_closeness': 0.215633, 'std_closeness': 0.056670},
    'MET': {'mean_degree': 0.045572, 'std_degree': 0.027090, 'mean_closeness': 0.231771, 'std_closeness': 0.061161},
    'PHE': {'mean_degree': 0.051824, 'std_degree': 0.029607, 'mean_closeness': 0.240883, 'std_closeness': 0.063045},
    'PRO': {'mean_degree': 0.033768, 'std_degree': 0.020658, 'mean_closeness': 0.216954, 'std_closeness': 0.053688},
    'SER': {'mean_degree': 0.035365, 'std_degree': 0.021483, 'mean_closeness': 0.219265, 'std_closeness': 0.055748},
    'THR': {'mean_degree': 0.037972, 'std_degree': 0.023229, 'mean_closeness': 0.222005, 'std_closeness': 0.057524},
    'TRP': {'mean_degree': 0.055449, 'std_degree': 0.033289, 'mean_closeness': 0.242541, 'std_closeness': 0.064827},
    'TYR': {'mean_degree': 0.050873, 'std_degree': 0.030606, 'mean_closeness': 0.236764, 'std_closeness': 0.062960},
    'VAL': {'mean_degree': 0.045333, 'std_degree': 0.027076, 'mean_closeness': 0.231924, 'std_closeness': 0.061815}
}

def standardize_centrality(observed_value, mean_value, std_value):
    """Calculate standardized centrality value (z-score)"""
    return (observed_value - mean_value) / std_value

def process_protein_rig(input_file, output_file):
    """
    Process a single protein RIG file and create standardized version
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    
    # Read the RIG data
    df = pd.read_csv(input_file)
    
    # Initialize columns for standardized values
    df['StandardizedDegree'] = np.nan
    df['StandardizedCloseness'] = np.nan
    
    # Process each residue
    for idx, row in df.iterrows():
        aa_type = row['Residue']
        observed_degree = row['DegreeCentrality']
        observed_closeness = row['ClosenessCentrality']
        
        # Get reference statistics for this amino acid type
        if aa_type in aa_stats:
            stats = aa_stats[aa_type]
            
            # Calculate standardized degree
            std_degree = standardize_centrality(
                observed_degree, 
                stats['mean_degree'], 
                stats['std_degree']
            )
            
            # Calculate standardized closeness
            std_closeness = standardize_centrality(
                observed_closeness, 
                stats['mean_closeness'], 
                stats['std_closeness']
            )
            
            # Store standardized values
            df.at[idx, 'StandardizedDegree'] = std_degree
            df.at[idx, 'StandardizedCloseness'] = std_closeness
            
        else:
            print(f"Warning: No reference statistics found for amino acid {aa_type} in {input_file}")
    
    # Save the results
    df.to_csv(output_file, index=False)
    print(f"Processed {input_file} -> {output_file}")
    
    # Print some summary statistics
    print(f"  Total residues: {len(df)}")
    print(f"  Mean standardized degree: {df['StandardizedDegree'].mean():.3f}")
    print(f"  Mean standardized closeness: {df['StandardizedCloseness'].mean():.3f}")
    print(f"  Max standardized degree: {df['StandardizedDegree'].max():.3f}")
    print(f"  Max standardized closeness: {df['StandardizedCloseness'].max():.3f}")
    print()

# Process each protein
protein_ids = ['2LW5', '5Y6A', '6VRB']

for protein_id in protein_ids:
    input_file = "/home/epi2melabs/projects/AcrNetworks/Important Files/3 main Acr data/6vrb_centrality.csv"
    output_file = "6VRB_RIG_centrality_standardized.csv"
    
    try:
        process_protein_rig(input_file, output_file)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please check the filename and path.")
    except Exception as e:
        print(f"Error processing {protein_id}: {str(e)}")

print("Standardization complete!")

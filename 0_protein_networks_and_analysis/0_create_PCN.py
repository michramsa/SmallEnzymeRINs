# 2/11/2026
# Create PCNs (protein contact networks)
# Only uses first model for NMR structures
# Threshold is 7 angstroms

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
from constants import *

def calc_residue_distance(res1, res2, choice):
    """Calculate distance between two residues based on specified atoms."""
    try:
        if choice == 'alpha':
            ca1 = res1["CA"]
            ca2 = res2["CA"]
            difference = ca1 - ca2
            return difference
        elif choice == 'beta':
            cb1 = res1["CB"]
            cb2 = res2["CB"]
            difference = cb1 - cb2
            return difference
        else:
            return float('inf')  # Return large value if choice is invalid
    except KeyError:
        # Handle cases where atoms don't exist (e.g., glycine has no CB)
        return float('inf')

def calc_dist_matrix(chain1, chain2, threshold, choice):
    """Calculate distance matrix between residues in two chains."""
    ans = np.zeros((len(chain1), len(chain2)), float)
    for row, res1 in enumerate(chain1):
        for col, res2 in enumerate(chain2):
            try:
                if choice == 'alpha':
                    dist = calc_residue_distance(res1, res2, 'alpha')
                elif choice == 'beta':
                    dist = calc_residue_distance(res1, res2, 'beta')
                else:
                    continue

                if dist <= threshold:
                    ans[row, col] = 1
            except Exception as e:
                print("Calc_residue_distance failure")
                continue
    
    return ans

# an NMR structure will have multiple models - returns a boolean
def is_nmr_structure(structure):
    return len(list(structure.get_models())) > 1

def create_nmr_graph(structure_name, structure_file, chain_choice, atom_choice, threshold):
    """
    Create a protein structure network specifically for NMR structures,
    using only the first model (model 0).
    
    Parameters:
    - structure_name: Name identifier for the structure
    - structure_file: Path to PDB file
    - chain_choice: 'both' for all chains, or specific chain ID
    - atom_choice: 'alpha' for CA atoms, 'beta' for CB atoms
    - threshold: Distance threshold in Angstroms
    
    Returns:
    - graph: NetworkX graph object
    - dipole_dict: Dictionary of dipole moments (placeholder)
    - volume_dict: Dictionary of volumes (placeholder)
    """
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_name, structure_file)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None, {}, {}
    
    # Get only the first model for NMR structures
    models = list(structure.get_models())
    if len(models) == 0:
        print("No models found in structure")
        return None, {}, {}
    
    first_model = models[0]
    print(f"Processing NMR structure - using model {first_model.id} (total models: {len(models)})")
    
    # Print available chains in the first model
    print("Available chains in first model:")
    for chain in first_model:
        print(f"Chain ID: {chain.id}")
    
    # Get residues based on chain choice from the first model only
    if chain_choice == "both":
        print("Processing all chains from first model")
        res_list = Selection.unfold_entities(first_model, "R")
    else:
        print(f"Processing chain {chain_choice} from first model")
        try:
            chain_obj = first_model[chain_choice]
            res_list = Selection.unfold_entities(chain_obj, "R")
        except KeyError:
            print(f"Chain {chain_choice} not found in first model")
            return None, {}, {}

    # Filter residues based on atom choice
    res_list_clean = []
    if atom_choice == 'alpha':
        for res in res_list:
            if res.has_id("CA"):
                res_list_clean.append(res)
    elif atom_choice == 'beta':
        for res in res_list:
            if res.has_id("CB"):
                res_list_clean.append(res)

    print(f"Number of residues after filtering: {len(res_list_clean)}")

    # Create residue names dictionary with chain information
    res_names_dict = {}
    chain_counters = {}  # Keep track of residue numbers per chain
    
    for i, res in enumerate(res_list_clean):
        chain_id = res.get_parent().id
        
        # Initialize counter for new chains
        if chain_id not in chain_counters:
            chain_counters[chain_id] = 1
        
        # Create residue identifier
        if chain_choice == "both":
            res_identifier = f"{res.get_resname()}_{chain_id}_{chain_counters[chain_id]}"
        else:
            res_identifier = f"{res.get_resname()}_{chain_counters[chain_id]}"
        
        res_names_dict[i] = {
            "residue": res_identifier,
            "chain_id": chain_id,
            "res_number": res.get_id()[1],
            "res_name": res.get_resname(),
            "model_id": first_model.id
        }
        
        chain_counters[chain_id] += 1

    # Create contact map
    print("Calculating contact map...")
    contact_map = calc_dist_matrix(res_list_clean, res_list_clean, threshold, atom_choice)
    print(f"Contact map dimensions: {contact_map.shape}")

    # Create NetworkX graph
    g = nx.Graph()
    
    # Add nodes
    g.add_nodes_from(list(range(len(res_list_clean))))
    
    # Set node attributes
    nx.set_node_attributes(g, res_names_dict)

    # Generate edges from contact map
    edge_list = []
    for index, res in enumerate(contact_map):
        indices = np.where(res == 1)[0]
        for contact_pt_index in indices:
            if index != contact_pt_index:  # Avoid self-loops
                new_edge = (index, contact_pt_index)
                edge_list.append(new_edge)

    print(f"Number of edges: {len(edge_list)}")
    
    # Add edges to graph
    g.add_edges_from(edge_list)

    # Create placeholder dictionaries for dipoles and volumes
    dipole_dict = {i: 0.0 for i in range(len(res_list_clean))}
    volume_dict = {i: 0.0 for i in range(len(res_list_clean))}

    print(f"NMR graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    
    return g, dipole_dict, volume_dict

def create_multi_chain_graph(structure_name, structure_file, chain_choice, atom_choice, threshold):
    """
    Create a protein structure network that can handle multiple chains.
    Now automatically detects NMR structures and uses appropriate method.
    
    Parameters:
    - structure_name: Name identifier for the structure
    - structure_file: Path to PDB file
    - chain_choice: 'both' for all chains, or specific chain ID
    - atom_choice: 'alpha' for CA atoms, 'beta' for CB atoms
    - threshold: Distance threshold in Angstroms
    
    Returns:
    - graph: NetworkX graph object
    - dipole_dict: Dictionary of dipole moments (placeholder)
    - volume_dict: Dictionary of volumes (placeholder)
    """
    
    # Parse the PDB file first to check if it's NMR
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_name, structure_file)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None, {}, {}
    
    # Check if this is an NMR structure
    if is_nmr_structure(structure):
        print("NMR structure detected - using first model only")
        return create_nmr_graph(structure_name, structure_file, chain_choice, atom_choice, threshold)
    
    # Original code for non-NMR structures
    print("Single model structure detected - processing normally")
    
    # Print available chains
    print("Available chains:")
    for chain in structure[0]:  # Access first model
        print(f"Chain ID: {chain.id}")
    
    # Get residues based on chain choice
    if chain_choice == "both":
        print("Processing all chains")
        res_list = Selection.unfold_entities(structure, "R")
    else:
        print(f"Processing chain {chain_choice}")
        try:
            chain_obj = structure[0][chain_choice]
            res_list = Selection.unfold_entities(chain_obj, "R")
        except KeyError:
            print(f"Chain {chain_choice} not found in structure")
            return None, {}, {}

    # Filter residues based on atom choice
    res_list_clean = []
    if atom_choice == 'alpha':
        for res in res_list:
            if res.has_id("CA"):
                res_list_clean.append(res)
    elif atom_choice == 'beta':
        for res in res_list:
            if res.has_id("CB"):
                res_list_clean.append(res)

    print(f"Number of residues after filtering: {len(res_list_clean)}")

    # Create residue names dictionary with chain information
    res_names_dict = {}
    chain_counters = {}  # Keep track of residue numbers per chain
    
    for i, res in enumerate(res_list_clean):
        chain_id = res.get_parent().id
        
        # Initialize counter for new chains
        if chain_id not in chain_counters:
            chain_counters[chain_id] = 1
        
        # Create residue identifier
        if chain_choice == "both":
            res_identifier = f"{res.get_resname()}_{chain_id}_{chain_counters[chain_id]}"
        else:
            res_identifier = f"{res.get_resname()}_{chain_counters[chain_id]}"
        
        res_names_dict[i] = {
            "residue": res_identifier,
            "chain_id": chain_id,
            "res_number": res.get_id()[1],
            "res_name": res.get_resname()
        }
        
        chain_counters[chain_id] += 1

    # Create contact map
    print("Calculating contact map...")
    contact_map = calc_dist_matrix(res_list_clean, res_list_clean, threshold, atom_choice)
    print(f"Contact map dimensions: {contact_map.shape}")

    # Create NetworkX graph
    g = nx.Graph()
    
    # Add nodes
    g.add_nodes_from(list(range(len(res_list_clean))))
    
    # Set node attributes
    nx.set_node_attributes(g, res_names_dict)

    # Generate edges from contact map
    edge_list = []
    for index, res in enumerate(contact_map):
        indices = np.where(res == 1)[0]
        for contact_pt_index in indices:
            if index != contact_pt_index:  # Avoid self-loops
                new_edge = (index, contact_pt_index)
                edge_list.append(new_edge)

    print(f"Number of edges: {len(edge_list)}")
    
    # Add edges to graph
    g.add_edges_from(edge_list)

    # Create placeholder dictionaries for dipoles and volumes
    dipole_dict = {i: 0.0 for i in range(len(res_list_clean))}
    volume_dict = {i: 0.0 for i in range(len(res_list_clean))}

    print(f"Graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
    
    return g, dipole_dict, volume_dict

def analyze_graph_by_chain(graph):
    """Analyze the graph and print statistics by chain."""
    if graph is None:
        return
    
    chain_stats = {}
    
    for node_id, node_data in graph.nodes(data=True):
        print(node_id)
        print(node_data)
        chain_id = node_data.get('chain_id', 'Unknown')
        
        if chain_id not in chain_stats:
            chain_stats[chain_id] = {
                'node_count': 0,
                'residue_types': set(),
                'degrees': []
            }
        
        chain_stats[chain_id]['node_count'] += 1
        chain_stats[chain_id]['residue_types'].add(node_data.get('res_name', 'Unknown'))
        chain_stats[chain_id]['degrees'].append(graph.degree(node_id))
    
    print("\n=== Chain Analysis ===")
    for chain_id, stats in chain_stats.items():
        print(f"Chain {chain_id}:")
        print(f"  - Nodes: {stats['node_count']}")
        print(f"  - Unique residue types: {len(stats['residue_types'])}")
        print(f"  - Average degree: {np.mean(stats['degrees']):.2f}")
        print(f"  - Degree range: {min(stats['degrees'])} - {max(stats['degrees'])}")

def save_graph_with_chain_info(graph, save_path_base, structure_name):
    """Save the graph in multiple formats with chain information preserved."""
    if graph is None:
        return
    
    # Save as GraphML (preserves all attributes)
    graphml_path = f"{save_path_base}{structure_name}_PCN.graphml"
    nx.write_graphml(graph, graphml_path)
    print(f"GraphML saved to: {graphml_path}")
    
    # Save as adjacency list
    adjlist_path = f"{save_path_base}{structure_name}_PCN.adjlist"
    nx.write_adjlist(graph, adjlist_path)
    print(f"Adjacency list saved to: {adjlist_path}")
    
    # Save node information as CSV for easy inspection
    node_info_path = f"{save_path_base}{structure_name}_PCN_nodes.csv"
    
    node_data = []
    for node_id, node_attr in graph.nodes(data=True):
        node_data.append({
            'node_id': node_id,
            'residue': node_attr.get('residue', ''),
            'chain_id': node_attr.get('chain_id', ''),
            'res_number': node_attr.get('res_number', ''),
            'res_name': node_attr.get('res_name', ''),
            'model_id': node_attr.get('model_id', 0),  # Will be 0 for non-NMR structures
            'degree': graph.degree(node_id)
        })
    
    df = pd.DataFrame(node_data)
    df.to_csv(node_info_path, index=False)
    print(f"Node information saved to: {node_info_path}")

def get_all_pdb_files(base_dir):
    """
    Recursively find all PDB files in the directory structure.
    
    Parameters:
    - base_dir: Base directory containing Acr_PDB folder structure
    
    Returns:
    - List of tuples: (pdb_file_path, acr_type, structure_name)
    """
    pdb_files = []
    acr_pdb_dir = Path(base_dir) / "all_acrs" / "Acr_PDB"
    
    if not acr_pdb_dir.exists():
        print(f"Warning: {acr_pdb_dir} does not exist")
        return pdb_files
    
    # Iterate through Acr type directories (e.g., AcrIA, AcrIB, etc.)
    for acr_type_dir in acr_pdb_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            
            # Find all .pdb files in this Acr type directory
            for pdb_file in acr_type_dir.glob("*.pdb"):
                structure_name = pdb_file.stem  # Filename without extension
                pdb_files.append((pdb_file, acr_type, structure_name))
    
    return pdb_files

def load_processed_files(log_file):
    """
    Load the set of successfully processed PDB files from the log.
    
    Parameters:
    - log_file: Path to the processing log CSV file
    
    Returns:
    - Set of PDB file paths that were successfully processed
    """
    processed = set()
    
    if Path(log_file).exists():
        try:
            df = pd.read_csv(log_file)
            # Only include files that were successfully processed
            successful = df[df['status'] == 'success']['pdb_file'].tolist()
            processed = set(successful)
            print(f"Loaded {len(processed)} previously processed files from log")
        except Exception as e:
            print(f"Warning: Could not load log file: {e}")
    
    return processed

def log_processed_file(log_file, pdb_file, acr_type, status, error_message=""):
    """
    Log the processing status of a PDB file.
    
    Parameters:
    - log_file: Path to the log CSV file
    - pdb_file: Path to the PDB file
    - acr_type: Type of Acr protein
    - status: 'success' or 'failure'
    - error_message: Error details if status is 'failure'
    """
    from datetime import datetime
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pdb_file': str(pdb_file),
        'acr_type': acr_type,
        'status': status,
        'error_message': error_message
    }
    
    # Create DataFrame with single row
    df_new = pd.DataFrame([log_entry])
    
    # Append to existing log or create new one
    if Path(log_file).exists():
        df_new.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(log_file, mode='w', header=True, index=False)

def process_single_pdb(pdb_file, acr_type, structure_name, output_base_dir, threshold, atom_choice, chain_choice):
    """
    Process a single PDB file to create protein contact network.
    
    Parameters:
    - pdb_file: Path to PDB file
    - acr_type: Type of Acr protein (e.g., 'AcrIA')
    - structure_name: Name for the structure (filename without extension)
    - output_base_dir: Base directory for output files
    - threshold: Distance threshold in Angstroms
    - atom_choice: 'alpha' or 'beta'
    - chain_choice: 'both' or specific chain ID
    
    Returns:
    - True if successful, False if failed
    """
    try:
        # Create output directory matching input structure
        output_dir = Path(output_base_dir) / acr_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create graph
        graph, dipoles, volumes = create_multi_chain_graph(
            structure_name,
            pdb_file,
            chain_choice,
            atom_choice,
            threshold
        )
        
        if graph is None:
            return False
        
        # Save graph files
        save_graph_with_chain_info(graph, str(output_dir) + "/", structure_name)
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return False

def get_closeness_value(node_id, closeness_centrality):
    """
    Get closeness centrality value for a node, handling type conversion between node ID formats.
    
    Parameters:
    - node_id: The node identifier (can be int or string)
    - closeness_centrality: Dictionary mapping node IDs to closeness centrality values
    
    Returns:
    - Closeness centrality value for the node, or None if not found
    """
    # Try the node_id as-is first
    if node_id in closeness_centrality:
        return closeness_centrality[node_id]
    # Try converting to string
    elif str(node_id) in closeness_centrality:
        return closeness_centrality[str(node_id)]
    # Try converting to int (if the node_id is a string that represents an int)
    else:
        try:
            int_node_id = int(node_id)
            if int_node_id in closeness_centrality:
                return closeness_centrality[int_node_id]
        except (ValueError, TypeError):
            pass
    return None

def add_closeness_centrality_to_csv(graphml_file_path, csv_file_path, output_csv_path=None):
    """
    Load an existing GraphML file, calculate closeness centrality for each node,
    and add it as a column to an existing CSV file.
    
    Parameters:
    - graphml_file_path: Path to the GraphML file
    - csv_file_path: Path to the existing CSV file with node information
    - output_csv_path: Path for the output CSV file (if None, overwrites the input CSV)
    
    Returns:
    - None (saves the updated CSV file)
    """
    
    # Load the graph from GraphML file
    try:
        graph = nx.read_graphml(graphml_file_path)
        print(f"Successfully loaded graph from {graphml_file_path}")
        print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading GraphML file: {e}")
        return
    
    # Calculate closeness centrality for all nodes
    try:
        closeness_centrality = nx.closeness_centrality(graph)
        print("Successfully calculated closeness centrality")
    except Exception as e:
        print(f"Error calculating closeness centrality: {e}")
        return
    
    # Load the existing CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV file with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Debug: Check the node ID types in both graph and CSV
    graph_node_sample = list(graph.nodes())[:5]
    csv_node_sample = df['node_id'].head().tolist()
    print(f"Sample graph node IDs: {graph_node_sample} (type: {type(graph_node_sample[0]) if graph_node_sample else 'N/A'})")
    print(f"Sample CSV node IDs: {csv_node_sample} (type: {type(csv_node_sample[0]) if csv_node_sample else 'N/A'})")
    
    # Add closeness centrality values to the dataframe
    # lambda being used to shorthand define a function because apply can only pass one argument but get_closeness_value needs 2 arguments
    # you can expand this if you want - it will just be "chunkier" (aka you'd need another function to make apply work)
    df['closeness_centrality'] = df['node_id'].apply(lambda node_id: get_closeness_value(node_id, closeness_centrality))
    
    # Check for any missing values (nodes that don't exist in the graph)
    missing_nodes = df[df['closeness_centrality'].isna()]
    if not missing_nodes.empty:
        print(f"Warning: {len(missing_nodes)} nodes from CSV not found in graph:")
        print(missing_nodes['node_id'].tolist())
        # Fill missing values with 0 or NaN as appropriate
        df.loc[df['closeness_centrality'].isna(), 'closeness_centrality'] = 0
    
    # Determine output path
    if output_csv_path is None:
        output_csv_path = csv_file_path
    
    # Save the updated CSV file
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved updated CSV to {output_csv_path}")
        print(f"Added closeness centrality column with values ranging from {df['closeness_centrality'].min():.4f} to {df['closeness_centrality'].max():.4f}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")
        return
    
if __name__ == "__main__":
    
    TESTING = True  # Set to False to process all files
    MAX_TEST_FILES = 3  # Number of files to process in testing mode
    OUTPUT_DIR = DATA_DIR / "acr_results"
    LOG_FILE = DATA_DIR / "processing_log.csv"
    
    # Processing parameters
    THRESHOLD = 7  # Angstroms
    ATOM_CHOICE = 'alpha'  # 'alpha' or 'beta'
    CHAIN_CHOICE = 'both'  # 'both' or specific chain ID
    
    print("="*80)
    print("PROTEIN CONTACT NETWORK BATCH PROCESSING")
    print("="*80)
    print(f"Testing mode: {TESTING}")
    print(f"Base directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Parameters: threshold={THRESHOLD}Å, atom={ATOM_CHOICE}, chains={CHAIN_CHOICE}")
    print("="*80)
    
    # Find all PDB files
    print("\nScanning for PDB files...")
    all_pdb_files = get_all_pdb_files(DATA_DIR)
    print(f"Found {len(all_pdb_files)} total PDB files")
    
    # if len(all_pdb_files) == 0:
    #     print("No PDB files found. Exiting.")
    #     exit()
    
    # # Load already processed files
    # print("\nChecking for previously processed files...")
    # processed_files = load_processed_files(LOG_FILE)
    
    # # Filter out already processed files
    # remaining_files = [
    #     (pdb_file, acr_type, structure_name) 
    #     for pdb_file, acr_type, structure_name in all_pdb_files 
    #     if str(pdb_file) not in processed_files
    # ]
    
    # print(f"Already processed: {len(processed_files)}")
    # print(f"Remaining to process: {len(remaining_files)}")
    
    # # Apply testing limit if needed
    # if TESTING:
    #     files_to_process = remaining_files[:MAX_TEST_FILES]
    #     print(f"\n*** TESTING MODE: Processing only {len(files_to_process)} files ***\n")
    # else:
    #     files_to_process = remaining_files
    
    # if len(files_to_process) == 0:
    #     print("\nNo files to process. All files have been completed.")
    #     exit()
    
    # # Process each file
    # print(f"\nStarting processing of {len(files_to_process)} files...")
    # print("="*80)
    
    # success_count = 0
    # failure_count = 0
    
    # for idx, (pdb_file, acr_type, structure_name) in enumerate(files_to_process, 1):
    #     print(f"\n[{idx}/{len(files_to_process)}] Processing: {structure_name} ({acr_type})")
    #     print(f"File: {pdb_file}")
        
    #     # Process the PDB file
    #     success = process_single_pdb(
    #         pdb_file=pdb_file,
    #         acr_type=acr_type,
    #         structure_name=structure_name,
    #         output_base_dir=OUTPUT_DIR,
    #         threshold=THRESHOLD,
    #         atom_choice=ATOM_CHOICE,
    #         chain_choice=CHAIN_CHOICE
    #     )
        
    #     # Log the result
    #     if success:
    #         log_processed_file(LOG_FILE, pdb_file, acr_type, 'success')
    #         success_count += 1
    #         print(f"✓ SUCCESS: {structure_name}")
    #     else:
    #         error_msg = "Graph creation returned None"
    #         log_processed_file(LOG_FILE, pdb_file, acr_type, 'failure', error_msg)
    #         failure_count += 1
    #         print(f"✗ FAILED: {structure_name}")
        
    #     # Progress summary
    #     print(f"Progress: {success_count} succeeded, {failure_count} failed")
    #     print("-"*80)
    
    # # Final summary
    # print("\n" + "="*80)
    # print("PROCESSING COMPLETE")
    # print("="*80)
    # print(f"Total processed: {len(files_to_process)}")
    # print(f"Successful: {success_count}")
    # print(f"Failed: {failure_count}")
    # print(f"Success rate: {100*success_count/len(files_to_process):.1f}%")
    # print(f"\nLog file: {LOG_FILE}")
    # print(f"Output directory: {OUTPUT_DIR}")
    # print("="*80)
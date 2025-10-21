## This code is what creates the figures for the degree and closeness centrality distributions (Figs 1-6).
# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib.pyplot as plt

def compare_degree_distributions_hatching(pcn_csv_path, rig_csv_path, output_dir=None, save_plot=True):
    """
    Compare degree distributions between PCN and RIG graphs.
   
    Parameters:
    - pcn_csv_path: Path to PCN degrees CSV file
    - rig_csv_path: Path to RIG degrees CSV file
    - output_dir: Directory to save the plot (optional)
    - save_plot: Whether to save the plot as a file
   
    Returns:
    - None (displays and optionally saves the plot)
    """
   
    # Read the CSV files
    try:
        pcn_df = pd.read_csv(pcn_csv_path)
        rig_df = pd.read_csv(rig_csv_path)
        print(f"Loaded PCN data: {len(pcn_df)} nodes")
        print(f"Loaded RIG data: {len(rig_df)} nodes")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
   
    # Extract degree values
    pcn_degrees = pcn_df['degree'].values
    rig_degrees = rig_df['degree'].values
   
    # Calculate statistics
    pcn_mean = np.mean(pcn_degrees)
    rig_mean = np.mean(rig_degrees)
   
    print(f"PCN mean degree: {pcn_mean:.2f}")
    print(f"RIG mean degree: {rig_mean:.2f}")
   
    plt.figure(figsize=(5.5, 3))
   
    # Define colors
    pcn_color = '#111184'
    rig_color = '#E1BE6A'  
   
    # Calculate histogram data
    max_degree = max(max(pcn_degrees), max(rig_degrees))
    min_degree = min(min(pcn_degrees), min(rig_degrees))
    bins = np.arange(min_degree, max_degree + 2) - 0.5  # Center bins on integers
   
    # Get histogram counts for overlap detection
    pcn_counts, bin_edges = np.histogram(pcn_degrees, bins=bins)
    rig_counts, _ = np.histogram(rig_degrees, bins=bins)
    
    # Create histograms - plot each component separately
    # First, plot PCN histogram
    plt.hist(pcn_degrees, bins=bins, alpha=0.7, color=pcn_color,
             label=f'PCN (mean: {pcn_mean:.2f})', edgecolor=pcn_color, linewidth=1.5)
    
    # Plot RIG histogram without overlap areas first
    rig_no_overlap = []
    for i, degree in enumerate(rig_degrees):
        bin_idx = np.digitize(degree, bin_edges) - 1
        if bin_idx >= 0 and bin_idx < len(pcn_counts) and pcn_counts[bin_idx] == 0:
            rig_no_overlap.append(degree)
    
    if rig_no_overlap:
        plt.hist(rig_no_overlap, bins=bins, alpha=0.7, color=rig_color,
                 edgecolor=rig_color, linewidth=1.5)
    
    # Plot only the overlapping portions with hatching
    rig_overlap = []
    for i, degree in enumerate(rig_degrees):
        bin_idx = np.digitize(degree, bin_edges) - 1
        if bin_idx >= 0 and bin_idx < len(pcn_counts) and pcn_counts[bin_idx] > 0:
            rig_overlap.append(degree)
    
    if rig_overlap:
        plt.hist(rig_overlap, bins=bins, alpha=0.7, color=rig_color,
                 hatch='.', edgecolor='#111184', linewidth=1.5,
                 label=f'RIG (mean: {rig_mean:.2f})')
   
    # Add mean lines
    plt.axvline(pcn_mean, color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(rig_mean, color='black', linestyle='--', linewidth=2, alpha=0.8)
   
    # Formatting
    plt.xlabel('Degree', fontsize=9)
    plt.ylabel('Residue Count', fontsize=9)
    # plt.title('TITLE HERE IF DESIRED', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Enhanced legend to explain hatching
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=pcn_color, alpha=0.7, edgecolor='white', linewidth=1.5, label=f'PCN (mean: {pcn_mean:.2f})'),
        plt.Rectangle((0,0),1,1, facecolor=rig_color, alpha=0.7, edgecolor='white', linewidth=1.5, label=f'RIG (mean: {rig_mean:.2f})'),
        plt.Rectangle((0,0),1,1, facecolor=rig_color, alpha=0.7, edgecolor='#111184', linewidth=2, hatch='.', label='Overlap')
    ]
    plt.legend(handles=legend_elements, fontsize=9)
   
    # Set integer ticks on x-axis
    plt.xticks(range(int(min_degree), int(max_degree) + 1))
   
    plt.tight_layout()
   
    # Save the plot if requested
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        ## EDIT SAVEFILE NAME HERE!
        output_file = os.path.join(output_dir, 'CB_6VRB_degree_distribution_comparison_hatched.png')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
   
    plt.show()
   
    # Print additional statistics
    print("\n=== Degree Statistics ===")
    print(f"PCN - Min: {min(pcn_degrees)}, Max: {max(pcn_degrees)}, Std: {np.std(pcn_degrees):.2f}")
    print(f"RIG - Min: {min(rig_degrees)}, Max: {max(rig_degrees)}, Std: {np.std(rig_degrees):.2f}")

def create_degree_comparison_plots(pcn_csv_path, rig_csv_path, output_dir):
    
    print("Creating overlapping degree distribution comparison...")
    compare_degree_distributions_hatching(pcn_csv_path, rig_csv_path, output_dir)

def compare_closeness_distributions(pcn_csv_path, rig_csv_path, output_dir=None, save_plot=True):
    """
        Compare closeness centrality distributions between PCN and RIG graphs.
    
        Parameters:
        - pcn_csv_path: Path to PCN nodes CSV file
        - rig_csv_path: Path to RIG nodes CSV file
        - output_dir: Directory to save the plot (optional)
        - save_plot: Whether to save the plot as a file
    
        Returns:
        - None (displays and optionally saves the plot)
    """
   
    # Read the CSV files
    try:
        pcn_df = pd.read_csv(pcn_csv_path)
        rig_df = pd.read_csv(rig_csv_path)
        print(f"Loaded PCN data: {len(pcn_df)} nodes")
        print(f"Loaded RIG data: {len(rig_df)} nodes")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return
   
    # Extract closeness centrality values
    pcn_closeness = pcn_df['closeness_centrality'].values
    rig_closeness = rig_df['closeness_centrality'].values
   
    # Calculate statistics
    pcn_mean = np.mean(pcn_closeness)
    rig_mean = np.mean(rig_closeness)
   
    print(f"PCN mean closeness centrality: {pcn_mean:.4f}")
    print(f"RIG mean closeness centrality: {rig_mean:.4f}")
   
    # # Create the plot
    # plt.rcParams.update({
    #     'font.family': 'sans-serif',
    #     'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
    #     'font.size': 12,
    #     'axes.labelsize': 9,
    #     'axes.titlesize': 9,
    #     'xtick.labelsize': 9,
    #     'ytick.labelsize': 9,
    #     'legend.fontsize': 9,
    # })
    plt.figure(figsize=(5.5, 3))
   
    # Define colors
    pcn_color = '#111184'
    rig_color = '#E1BE6A' 
   
    # Calculate appropriate bins for closeness centrality (typically ranges 0-1)
    min_closeness = min(min(pcn_closeness), min(rig_closeness))
    max_closeness = max(max(pcn_closeness), max(rig_closeness))
    bins = np.linspace(min_closeness, max_closeness, 30)  # 30 bins for smooth distribution
   
    # Get histogram counts for overlap detection
    pcn_counts, bin_edges = np.histogram(pcn_closeness, bins=bins)
    rig_counts, _ = np.histogram(rig_closeness, bins=bins)
    
    # Create histograms - plot each component separately
    # First, plot PCN histogram
    plt.hist(pcn_closeness, bins=bins, alpha=0.7, color=pcn_color,
             label=f'PCN (mean: {pcn_mean:.4f})', edgecolor=pcn_color, linewidth=1.5)
    
    # Plot RIG histogram without overlap areas first
    rig_no_overlap = []
    for i, closeness in enumerate(rig_closeness):
        bin_idx = np.digitize(closeness, bin_edges) - 1
        if bin_idx >= 0 and bin_idx < len(pcn_counts) and pcn_counts[bin_idx] == 0:
            rig_no_overlap.append(closeness)
    
    if rig_no_overlap:
        plt.hist(rig_no_overlap, bins=bins, alpha=0.7, color=rig_color,
                 edgecolor=rig_color, linewidth=1.5)
    
    # Plot only the overlapping portions with hatching
    rig_overlap = []
    for i, closeness in enumerate(rig_closeness):
        bin_idx = np.digitize(closeness, bin_edges) - 1
        if bin_idx >= 0 and bin_idx < len(pcn_counts) and pcn_counts[bin_idx] > 0:
            rig_overlap.append(closeness)
    
    if rig_overlap:
        plt.hist(rig_overlap, bins=bins, alpha=0.7, color=rig_color,
                 hatch='.', edgecolor='#111184', linewidth=1.5,
                 label=f'RIG (mean: {rig_mean:.2f})')
    else:
        # Add RIG label even if no overlap exists
        plt.hist([], bins=bins, alpha=0.7, color=rig_color,
                 edgecolor=rig_color, linewidth=1.5,
                 label=f'RIG (mean: {rig_mean:.4f})')
   
    # Add mean lines
    plt.axvline(pcn_mean, color='black', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(rig_mean, color='black', linestyle='--', linewidth=2, alpha=0.8)
   
    # Formatting
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Residue Count')
    # plt.title('TITLE HERE IF DESIRED', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Enhanced legend to explain hatching
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=pcn_color, alpha=0.7, edgecolor='white', linewidth=1.5, label=f'PCN (mean: {pcn_mean:.4f})'),
        plt.Rectangle((0,0),1,1, facecolor=rig_color, alpha=0.7, edgecolor='white', linewidth=1.5, label=f'RIG (mean: {rig_mean:.4f})'),
        plt.Rectangle((0,0),1,1, facecolor=rig_color, alpha=0.7, edgecolor='#111184', linewidth=2, hatch='.', label='Overlap')
    ]
    plt.legend(handles=legend_elements, fontsize=9)
   
    plt.tight_layout()
   
    # Save the plot if requested
    if save_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        ## EDIT SAVEFILE NAME HERE!
        output_file = os.path.join(output_dir, 'CB_6VRB_closeness_distribution_comparison_hatched.png')

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
   
    plt.show()
   
    # Print additional statistics
    print("\n=== Closeness Centrality Statistics ===")
    print(f"PCN - Min: {min(pcn_closeness):.4f}, Max: {max(pcn_closeness):.4f}, Std: {np.std(pcn_closeness):.4f}")
    print(f"RIG - Min: {min(rig_closeness):.4f}, Max: {max(rig_closeness):.4f}, Std: {np.std(rig_closeness):.4f}")

def create_closeness_comparison_plots(pcn_csv_path, rig_csv_path, output_dir): 
    
    print("Creating overlapping closeness centrality distribution comparison...")
    compare_closeness_distributions(pcn_csv_path, rig_csv_path, output_dir)
if __name__ == "__main__":

    pcn_csv_path = IMP_DIR / "3 main Acr data/6vrb_PCN_nodes.csv"  # PCN degrees file
    rig_csv_path = IMP_DIR / "3 main Acr data/6vrb_RIG_nodes.csv"   # RIG degrees file
    output_dir = IMP_DIR / "3 main Acr data" # where you want the figures output into

    # create_degree_comparison_plots(pcn_csv_path,rig_csv_path, output_dir)
    create_closeness_comparison_plots(pcn_csv_path,rig_csv_path, output_dir)
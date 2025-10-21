## REGARDING COLORS - https://matplotlib.org/stable/users/explain/colors/colormaps.html
## This file creates the figures for RSA/Closeness with or without the residues of interest highlighted, corresponding to figures 7,8,9
from imports import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

def validate_residue_of_interest(df, residue_id, expected_residue=None):
    """
    Validate that a residue of interest exists and optionally check its type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The merged dataframe with centrality and SASA data
    residue_id : str
        Residue identifier (e.g., "A_6")
    expected_residue : str, optional
        Expected 3-letter residue code (e.g., "TYR")
    
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    # Check if residue exists
    if residue_id not in df['Node'].values:
        print(f"Warning: Residue {residue_id} not found in dataset")
        return False
    
    # Check residue type if specified
    if expected_residue:
        actual_residue = df[df['Node'] == residue_id]['Residue'].iloc[0]
        if actual_residue != expected_residue:
            print(f"Warning: Residue {residue_id} is {actual_residue}, not {expected_residue}")
            return False
        else:
            print(f"Confirmed: {residue_id} is {actual_residue}")
    
    return True

def categorize_residues_by_sasa(df, rsa_column, residues_of_interest=None):
    """
    Categorize residues based on SASA values and highlight residues of interest.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The merged dataframe
    rsa_column : str
        Either 'RSA_theoretical' or 'RSA_empirical'
    residues_of_interest : list, optional
        List of residue IDs to highlight in red (e.g., ["A_6", "B_123"])
    
    Returns:
    --------
    pd.DataFrame : DataFrame with color categories added
    """
    df = df.copy()
    
    # Initialize color category
    df['Color_Category'] = 'Other [0.0 < RSA ≤ 0.5]'
    df['Color'] = 'grey'
    
    # Only process protein residues with SASA data
    protein_mask = (df['Molecule_Type'] == 'Protein') & (df[rsa_column].notna())
    
    # Categorize based on SASA values
    highly_exposed = protein_mask & (df[rsa_column] > 0.5)
    core_residues = protein_mask & (df[rsa_column] == 0.0)
    
    df.loc[highly_exposed, 'Color_Category'] = 'Highly Exposed [RSA > 0.5]'
    df.loc[highly_exposed, 'Color'] = 'orange'
    
    df.loc[core_residues, 'Color_Category'] = 'Core [RSA = 0.0]'
    df.loc[core_residues, 'Color'] = 'green'
    
    # Handle residues of interest
    if residues_of_interest:
        validated_residues = []
        for residue_id in residues_of_interest:
            if isinstance(residue_id, tuple) and len(residue_id) == 2:
                # Format: ("A_6", "TYR")
                res_id, expected_type = residue_id
                if validate_residue_of_interest(df, res_id, expected_type):
                    validated_residues.append(res_id)
            else:
                # Format: "A_6"
                if validate_residue_of_interest(df, residue_id):
                    validated_residues.append(residue_id)
        
        if validated_residues:
            interest_mask = df['Node'].isin(validated_residues)
            df.loc[interest_mask, 'Color_Category'] = 'Residues of Interest'
            df.loc[interest_mask, 'Color'] = 'black'
            print(f"Highlighted {len(validated_residues)} residues of interest in black")
    
    return df

def plot_centrality_by_sasa(csv_file, rsa_type='theoretical', residues_of_interest=None, 
                           figsize=(12, 8), alpha=0.7, save_path=None):
    """
    Create a scatter plot of standardized closeness vs standardized degree,
    color-coded by SASA categories.
    
    Parameters:
    -----------
    csv_file : str
        Path to the merged CSV file with centrality and SASA data
    rsa_type : str
        Either 'theoretical' or 'empirical' for RSA calculation type
    residues_of_interest : list, optional
        List of residue IDs to highlight. Can be:
        - ["A_6", "B_123"] (just IDs)
        - [("A_6", "TYR"), ("B_123", "LYS")] (IDs with expected residue types)
    figsize : tuple
        Figure size (width, height) in inches
    alpha : float
        Transparency of points (0-1)
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Select RSA column
    rsa_column = f'RSA_{rsa_type}'
    if rsa_column not in df.columns:
        raise ValueError(f"Column {rsa_column} not found. Available columns: {df.columns.tolist()}")
    
    # Categorize residues
    df_colored = categorize_residues_by_sasa(df, rsa_column, residues_of_interest)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define plotting order (residues of interest last so they appear on top)
    color_order = ['Other [0.0 < RSA ≤ 0.5]', 'Highly Exposed [RSA > 0.5]', 'Core [RSA = 0.0]', 'Residues of Interest']
    
    # Plot each category
    for category in color_order:
        if category not in df_colored['Color_Category'].values:
            continue
            
        category_data = df_colored[df_colored['Color_Category'] == category]
        if len(category_data) == 0:
            continue
        
        # Get color
        color = category_data['Color'].iloc[0]
        
        # Adjust marker properties for residues of interest
        if category == 'Residues of Interest':
            marker_size = 120
            edge_color = 'black'
            edge_width = 2
            alpha_val = 1.0
        else:
            marker_size = 60
            edge_color = 'white'
            edge_width = 0.5
            alpha_val = alpha
        
        ax.scatter(category_data['StandardizedCloseness'], 
                  category_data['StandardizedDegree'],
                  label=f'{category} (n={len(category_data)})',
                  color=color,
                  alpha=alpha_val,
                  s=marker_size,
                  edgecolors=edge_color,
                  linewidth=edge_width)
    
    # Customize the plot
    ax.set_xlabel('Standardized Closeness', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Degree', fontsize=12, fontweight='bold')
    
    title = f'AcrIIA1 Network Centrality Analysis - Color-coded by {rsa_type.title()} RSA'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid and reference lines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # # Add text box with SASA criteria
    # textstr = '\n'.join([
    #     'SASA Categories:',
    #     '• Core: RSA = 0.0',
    #     '• Highly Exposed: RSA > 0.5',
    #     '• Other: 0.0 < RSA ≤ 0.5',
    #     '• Red: Residues of Interest'
    # ])
    
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
    #         verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nDataset Summary for {rsa_type.title()} RSA:")
    print(f"Total data points: {len(df)}")
    
    category_counts = df_colored['Color_Category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    
    # Print SASA statistics for protein residues
    protein_df = df_colored[df_colored['Molecule_Type'] == 'Protein']
    if len(protein_df) > 0:
        rsa_values = protein_df[rsa_column].dropna()
        if len(rsa_values) > 0:
            print(f"\nRSA Statistics (Protein residues only):")
            print(f"Mean RSA: {rsa_values.mean():.3f}")
            print(f"Median RSA: {rsa_values.median():.3f}")
            print(f"Min RSA: {rsa_values.min():.3f}")
            print(f"Max RSA: {rsa_values.max():.3f}")
    
    return fig, ax

def plot_both_sasa_types(csv_file, residues_of_interest=None, figsize=(15, 6), 
                        alpha=0.7, save_path=None):
    """
    Create side-by-side plots for both theoretical and empirical RSA.
    
    Parameters:
    -----------
    csv_file : str
        Path to the merged CSV file
    residues_of_interest : list, optional
        List of residue IDs to highlight
    figsize : tuple
        Figure size for the combined plot
    alpha : float
        Transparency of points
    save_path : str, optional
        Path to save the combined figure
    
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axis objects
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Process data for both types
    for ax, rsa_type, rsa_column in zip([ax1, ax2], 
                                       ['theoretical', 'empirical'],
                                       ['RSA_theoretical', 'RSA_empirical']):
        
        # Categorize residues
        df_colored = categorize_residues_by_sasa(df, rsa_column, residues_of_interest)
        
        # Define plotting order
        color_order = ['Other', 'Highly Exposed', 'Core', 'Residues of Interest']
        
        # Plot each category
        for category in color_order:
            if category not in df_colored['Color_Category'].values:
                continue
                
            category_data = df_colored[df_colored['Color_Category'] == category]
            if len(category_data) == 0:
                continue
            
            color = category_data['Color'].iloc[0]
            
            if category == 'Residues of Interest':
                marker_size = 120
                edge_color = 'black'
                edge_width = 2
                alpha_val = 1.0
            else:
                marker_size = 60
                edge_color = 'white'
                edge_width = 0.5
                alpha_val = alpha
            
            ax.scatter(category_data['StandardizedCloseness'], 
                      category_data['StandardizedDegree'],
                      label=f'{category} (n={len(category_data)})',
                      color=color,
                      alpha=alpha_val,
                      s=marker_size,
                      edgecolors=edge_color,
                      linewidth=edge_width)
        
        # Customize each subplot
        ax.set_xlabel('Standardized Closeness', fontsize=11)
        ax.set_ylabel('Standardized Degree', fontsize=11)
        ax.set_title(f'{rsa_type.title()} RSA', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend only to the second plot
        if ax == ax2:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.suptitle('Network Centrality Analysis - SASA Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined figure saved to: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)

def plot_rsa_vs_closeness(csv_file, rsa_type='theoretical', figsize=(12, 8), 
                         alpha=0.7, save_path=None, color_by_chain=False):
    """
    Create a scatter plot of RSA values vs Closeness Centrality.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file with centrality and SASA data
    rsa_type : str
        Either 'theoretical' or 'empirical' for RSA calculation type
    figsize : tuple
        Figure size (width, height) in inches
    alpha : float
        Transparency of points (0-1)
    save_path : str, optional
        Path to save the figure
    color_by_chain : bool
        If True, color points by chain. If False, use single color.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Select RSA column
    rsa_column = f'RSA_{rsa_type}'
    if rsa_column not in df.columns:
        raise ValueError(f"Column {rsa_column} not found. Available columns: {df.columns.tolist()}")
    
    # Filter for protein residues only (excluding any non-protein molecules)
    protein_df = df[df['Molecule_Type'] == 'Protein'].copy()
    
    # Remove any rows with missing data
    protein_df = protein_df.dropna(subset=[rsa_column, 'ClosenessCentrality'])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by_chain and 'Chain' in protein_df.columns:
        # Get unique chains and assign colors
        chains = protein_df['Chain'].unique()
        colors = [plt.cm.Pastel1(3), plt.cm.Pastel2(0), plt.cm.Pastel2(1)]
        
        for i, chain in enumerate(chains):
            chain_data = protein_df[protein_df['Chain'] == chain]
            ax.scatter(chain_data[rsa_column], 
                      chain_data['ClosenessCentrality'],
                      label=f'Chain {chain}',
                      color=colors[i],
                      alpha=alpha,
                      s=60,
                      edgecolors='white',
                      linewidth=0.5)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # Single color plot
        ax.scatter(protein_df[rsa_column], 
                  protein_df['ClosenessCentrality'],
                  color='steelblue',
                  alpha=alpha,
                  s=60,
                  edgecolors='white',
                  linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel(f'{rsa_type.title()} RSA', fontsize=12, fontweight='bold')
    ax.set_ylabel('Closeness Centrality', fontsize=12, fontweight='bold')
    
    title = f'RSA vs Closeness Centrality ({rsa_type.title()} RSA)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set axis limits with some padding
    x_margin = (protein_df[rsa_column].max() - protein_df[rsa_column].min()) * 0.05
    y_margin = (protein_df['ClosenessCentrality'].max() - protein_df['ClosenessCentrality'].min()) * 0.05
    
    ax.set_xlim(protein_df[rsa_column].min() - x_margin, 
                protein_df[rsa_column].max() + x_margin)
    ax.set_ylim(protein_df['ClosenessCentrality'].min() - y_margin, 
                protein_df['ClosenessCentrality'].max() + y_margin)
    
    # Add correlation coefficient to the plot
    correlation = protein_df[rsa_column].corr(protein_df['ClosenessCentrality'])
    ax.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nDataset Summary for {rsa_type.title()} RSA vs Closeness Centrality:")
    print(f"Total protein residues: {len(protein_df)}")
    print(f"RSA range: {protein_df[rsa_column].min():.3f} - {protein_df[rsa_column].max():.3f}")
    print(f"Closeness Centrality range: {protein_df['ClosenessCentrality'].min():.3f} - {protein_df['ClosenessCentrality'].max():.3f}")
    print(f"Correlation coefficient: {correlation:.3f}")
    
    if 'Chain' in protein_df.columns:
        chain_counts = protein_df['Chain'].value_counts()
        print(f"Residues per chain: {dict(chain_counts)}")
    
    return fig, ax


def plot_rsa_vs_closeness_advanced(csv_file, rsa_type='theoretical', figsize=(12, 8), 
                                 alpha=0.7, save_path=None, highlight_residues=None,
                                 add_trendline=False, residue_labels=False):
    """
    Advanced version with additional features like highlighting specific residues.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    rsa_type : str
        Either 'theoretical' or 'empirical'
    figsize : tuple
        Figure size
    alpha : float
        Point transparency
    save_path : str, optional
        Save path
    highlight_residues : list, optional
        List of residue identifiers (Node column values) to highlight in red
    add_trendline : bool
        Whether to add a linear trendline
    residue_labels : bool
        Whether to label points with residue information
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Load and process data
    df = pd.read_csv(csv_file)
    rsa_column = f'RSA_{rsa_type}'
    
    if rsa_column not in df.columns:
        raise ValueError(f"Column {rsa_column} not found.")
    
    protein_df = df[df['Molecule_Type'] == 'Protein'].copy()
    protein_df = protein_df.dropna(subset=[rsa_column, 'ClosenessCentrality'])
    
    # Debug: Print available Node values if highlight_residues is provided
    if highlight_residues:
        print(f"Looking for residues: {highlight_residues}")
        print(f"Available Node values (first 10): {protein_df['Node'].head(10).tolist()}")
        print(f"Total available nodes: {len(protein_df['Node'].unique())}")
        
        # Check which residues are found
        found_residues = protein_df[protein_df['Node'].isin(highlight_residues)]['Node'].tolist()
        missing_residues = [r for r in highlight_residues if r not in protein_df['Node'].values]
        
        print(f"Found residues: {found_residues}")
        if missing_residues:
            print(f"Missing residues: {missing_residues}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Highlight specific residues if provided - PLOT THESE FIRST so they appear on top
    highlighted_data = pd.DataFrame()
    if highlight_residues:
        highlighted_data = protein_df[protein_df['Node'].isin(highlight_residues)]
        if len(highlighted_data) > 0:
            ax.scatter(highlighted_data[rsa_column],
                      highlighted_data['ClosenessCentrality'],
                      color='black',
                      alpha=1.0,
                      s=120,
                      edgecolors='black',
                      linewidth=2,
                      label=f'Highlighted Residues (n={len(highlighted_data)})',
                      zorder=5)  # Higher zorder to ensure they appear on top
            print(f"Successfully highlighted {len(highlighted_data)} residues")
            
            # Print details of highlighted residues
            for _, row in highlighted_data.iterrows():
                print(f"  {row['Node']}: {row['Residue']} at RSA={row[rsa_column]:.3f}, Closeness={row['ClosenessCentrality']:.3f}")
        else:
            print("Warning: No residues found to highlight!")
    
    # Plot main data points (non-highlighted)
    if len(highlighted_data) > 0:
        # Exclude highlighted residues from main plot
        main_data = protein_df[~protein_df['Node'].isin(highlight_residues)]
    else:
        main_data = protein_df
    
    ax.scatter(main_data[rsa_column], 
               main_data['ClosenessCentrality'],
               color='steelblue',
               alpha=alpha,
               s=60,
               edgecolors='white',
               linewidth=0.5,
               label=f'Other Residues (n={len(main_data)})',
               zorder=3)
    
    # Add trendline if requested
    if add_trendline:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            protein_df[rsa_column], protein_df['ClosenessCentrality'])
        
        x_trend = np.array([protein_df[rsa_column].min(), protein_df[rsa_column].max()])
        y_trend = slope * x_trend + intercept
        
        ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2, 
                label=f'Trendline (R² = {r_value**2:.3f})', zorder=4)
    
    # Add residue labels if requested (only for small datasets)
    if residue_labels and len(protein_df) <= 50:
        for idx, row in protein_df.iterrows():
            ax.annotate(f"{row['Residue']}{row['Residue number']}", 
                       (row[rsa_column], row['ClosenessCentrality']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    elif residue_labels and len(protein_df) > 50:
        print("Warning: Too many residues for labeling. Skipping labels.")
    
    # Customize plot
    ax.set_xlabel(f'{rsa_type.title()} RSA', fontsize=12, fontweight='bold')
    ax.set_ylabel('Closeness Centrality', fontsize=12, fontweight='bold')
    ax.set_title(f'RSA vs Closeness Centrality Analysis ({rsa_type.title()} RSA)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Always show legend if we have highlights or trendline
    if highlight_residues or add_trendline or len(highlighted_data) > 0:
        ax.legend(loc='best')
    
    # Add correlation info
    correlation = protein_df[rsa_column].corr(protein_df['ClosenessCentrality'])
    ax.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, ax


def plot_centrality_by_sasa_v2(csv_file, rsa_type='theoretical', residues_of_interest=None, 
                           figsize=(5.5, 3), alpha=0.7, save_path=None):
    """
    Create a scatter plot of standardized closeness vs standardized degree,
    color-coded by SASA categories.
    
    Parameters:
    -----------
    csv_file : str
        Path to the merged CSV file with centrality and SASA data
    rsa_type : str
        Either 'theoretical' or 'empirical' for RSA calculation type
    residues_of_interest : list, optional
        List of residue IDs to highlight. Can be:
        - ["A_6", "B_123"] (just IDs)
        - [("A_6", "TYR"), ("B_123", "LYS")] (IDs with expected residue types)
    figsize : tuple
        Figure size (width, height) in inches
    alpha : float
        Transparency of points (0-1)
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Select RSA column
    rsa_column = f'RSA_{rsa_type}'
    if rsa_column not in df.columns:
        raise ValueError(f"Column {rsa_column} not found. Available columns: {df.columns.tolist()}")
    
    # Categorize residues
    df_colored = categorize_residues_by_sasa(df, rsa_column, residues_of_interest)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define plotting order (residues of interest last so they appear on top)
    color_order = ['Other [0.0 < RSA ≤ 0.5]', 'Highly Exposed [RSA > 0.5]', 'Core [RSA = 0.0]', 'Residues of Interest']
    
    # Store handles and labels for custom legend
    legend_handles = []
    legend_labels = []
    
    # Plot each category
    for category in color_order:
        if category not in df_colored['Color_Category'].values:
            continue
            
        category_data = df_colored[df_colored['Color_Category'] == category]
        if len(category_data) == 0:
            continue
        
        # Get color
        color = category_data['Color'].iloc[0]
        
        # Use consistent marker properties for all categories
        marker_size = 60
        edge_width = 0.5  # Same edge width for all categories to ensure consistent legend appearance
        if category == 'Residues of Interest':
            edge_color = 'black'
            alpha_val = 1.0
        else:
            edge_color = 'white'
            alpha_val = alpha
        
        scatter = ax.scatter(category_data['StandardizedCloseness'], 
                  category_data['StandardizedDegree'],
                  color=color,
                  alpha=alpha_val,
                  s=marker_size,
                  edgecolors=edge_color,
                  linewidth=edge_width)
        
        # Store for legend - use the same scatter object for all categories
        legend_handles.append(scatter)
        legend_labels.append(f'{category} (n={len(category_data)})')
    
    # Calculate equal axis limits to center the origin (0,0)
    x_data = df_colored['StandardizedCloseness']
    y_data = df_colored['StandardizedDegree']
    
    # Find the maximum absolute value from both axes
    max_abs_x = max(abs(x_data.min()), abs(x_data.max()))
    max_abs_y = max(abs(y_data.min()), abs(y_data.max()))
    max_abs_value = max(max_abs_x, max_abs_y)
    
    # Add 10% padding to the maximum absolute value
    axis_limit = max_abs_value * 1.1
    
    # Set equal limits for both axes, centered on origin
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Customize the plot
    ax.set_xlabel('Standardized Closeness', fontsize=9, fontweight='bold')
    ax.set_ylabel('Standardized Degree', fontsize=9, fontweight='bold')
    
    # Add grid and reference lines
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.axhline(y=0, color='dimgray', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='dimgray', linestyle='--', alpha=0.5, linewidth=2)
    
    # Create legend with proper ordering and compact formatting
    if legend_handles:
        # Reverse order to match your desired arrangement
        order = [3, 2, 1, 0] if len(legend_handles) == 4 else list(range(len(legend_handles)-1, -1, -1))
        ordered_handles = [legend_handles[i] for i in order if i < len(legend_handles)]
        ordered_labels = [legend_labels[i] for i in order if i < len(legend_labels)]
        
        # Create legend with smaller font and tighter spacing
        legend = ax.legend(ordered_handles, ordered_labels,
                          bbox_to_anchor=(1.02, 1), 
                          loc='upper left',
                          fontsize=8,  # Reduced from 9 to 8
                          frameon=True, 
                          fancybox=True, 
                          shadow=False,       # Removed shadow
                          borderpad=0.3,      # Reduce padding inside legend
                          columnspacing=0.5,  # Reduce spacing between columns
                          handletextpad=0.3,  # Reduce space between marker and text
                          handlelength=1.0)   # Reduce length of legend markers
        
        # Adjust legend frame
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.9)
        
        # Create second legend for residues of interest with coordinates
        if residues_of_interest and 'Residues of Interest' in df_colored['Color_Category'].values:
            roi_data = df_colored[df_colored['Color_Category'] == 'Residues of Interest']
            if len(roi_data) > 0:
                # Create text for residue coordinates
                roi_text_lines = []
                for _, row in roi_data.iterrows():
                    residue_type = row.get('Residue', 'UNK')
                    residue_num = row.get('Residue number', 'X')
                    residue_label = f"{residue_type}_{residue_num}"
                    x_val = row['StandardizedCloseness']
                    y_val = row['StandardizedDegree']
                    rsa_emp = row.get('RSA_empirical', 'N/A')
                    rsa_text = f"{rsa_emp:.3f}" if isinstance(rsa_emp, (int, float)) else str(rsa_emp)
                    roi_text_lines.append(f"{residue_label}: ({x_val:.2f}, {y_val:.2f}), RSA: {rsa_text}")
                
                # Add text box below the main legend, aligned with it
                roi_text = '\n'.join(roi_text_lines)
                text_box = ax.text(1.03, 0.5, f"Residues of Interest:\n{roi_text}", 
                                 transform=ax.transAxes,
                                 fontsize=9,
                                 verticalalignment='top',
                                 bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor='white', 
                                         alpha=0.8,
                                         edgecolor='gray',
                                         linewidth=0.5))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nDataset Summary for {rsa_type.title()} RSA:")
    print(f"Total data points: {len(df)}")
    
    category_counts = df_colored['Color_Category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    
    # Print SASA statistics for protein residues
    protein_df = df_colored[df_colored['Molecule_Type'] == 'Protein']
    if len(protein_df) > 0:
        rsa_values = protein_df[rsa_column].dropna()
        if len(rsa_values) > 0:
            print(f"\nRSA Statistics (Protein residues only):")
            print(f"Mean RSA: {rsa_values.mean():.3f}")
            print(f"Median RSA: {rsa_values.median():.3f}")
            print(f"Min RSA: {rsa_values.min():.3f}")
            print(f"Max RSA: {rsa_values.max():.3f}")
    
    return fig, ax

# Debugging function to help identify correct residue identifiers
def debug_residue_identifiers(csv_file, sample_size=10):
    """
    Helper function to examine the Node identifiers in your CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    sample_size : int
        Number of sample Node identifiers to display
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    
    print("=== CSV File Structure Debug ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    if 'Molecule_Type' in df.columns:
        print(f"Molecule types: {df['Molecule_Type'].unique()}")
        protein_df = df[df['Molecule_Type'] == 'Protein']
        print(f"Protein rows: {len(protein_df)}")
    else:
        protein_df = df
    
    if 'Node' in protein_df.columns:
        print(f"\nSample Node identifiers:")
        sample_nodes = protein_df['Node'].head(sample_size).tolist()
        for i, node in enumerate(sample_nodes):
            row = protein_df[protein_df['Node'] == node].iloc[0]
            print(f"  {i+1}. '{node}' -> {row.get('Residue', 'N/A')} Chain {row.get('Chain', 'N/A')} Pos {row.get('Residue number', 'N/A')}")
        
        print(f"\nAll unique chains: {sorted(protein_df['Chain'].unique()) if 'Chain' in protein_df.columns else 'No Chain column'}")
        print(f"Node format examples: {protein_df['Node'].head(3).tolist()}")
    else:
        print("Error: No 'Node' column found!")
    
    return protein_df['Node'].tolist() if 'Node' in protein_df.columns else []

# Example usage
if __name__ == "__main__":
    
    # Example file path - update to your actual path
    csv_file = IMP_DIR / "3 main Acr data/5Y6A_centrality_with_sasa.csv"

    ## these are what you change for whatever you'd like the save file to be named!
    save_1 = "./82525_5Y6A_empirical_RSA_Closeness_ROI_v2"
    # save_2 = "./5Y6A_theoretical_RSA_Closeness_ROI"

    # Example 1: Basic SASA plot (theoretical)
    # print("Example 1: Theoretical RSA plot")
    # plot_centrality_by_sasa(csv_file, rsa_type='empirical', save_path=save)
    
    # Example 2: Empirical RSA plot with specific residues highlighted
    # 2LW5 - Tyr6, Tyr20, Glu31 (CHAIN A)
    # lw5_res = ['A_6', 'A_20', 'A_31']

    # 5Y6A - Phe115 (CHAIN A AND B are identical, dimer)
    y6a_res = ['A_115']

    # 6VRB - Tyr39A, Ser40, Asn43, Ser93, Gln96 (CHAIN C)
    # vrb_res = ['C_39', 'C_40', 'C_43', 'C_93', 'C_96']

    # print("\nExample 2: Empirical RSA plot with residues of interest")
    # residues_to_highlight = [("C_39", "TYR"), ("C_40", "SER"), ("C_43", "ASN"), ("C_93", "SER"), ("C_96", "GLN")]

    df = pd.read_csv(csv_file)
    chain_c_residues = df[df['Chain'] == 'A']['Node'].tolist()

    plot_centrality_by_sasa_v2(csv_file, rsa_type='empirical', residues_of_interest=y6a_res, save_path=save_1)
    # plot_centrality_by_sasa_v2(csv_file, rsa_type='theoretical', residues_of_interest=chain_c_residues, save_path=None)
    
    
    # # Example 3: Side-by-side comparison
    # print("\nExample 3: Side-by-side comparison")
    # plot_both_sasa_types(csv_file, residues_of_interest=residues_to_highlight)
    
    # print("\nTo use with your actual data:")
    # print("plot_centrality_by_sasa(")
    # print("    '/your/path/merged_data.csv',")
    # print("    rsa_type='theoretical',")  
    # print("    residues_of_interest=[('A_6', 'TYR')]")
    # print(")")


    lw5_res = ['A_6', 'A_20', 'A_31']
    y6a_res = ['A_115']
    vrb_res = ['C_39', 'C_40', 'C_43', 'C_93', 'C_96']

    # plot_rsa_vs_closeness(csv_file,'empirical',save_path=None)
    # plot_rsa_vs_closeness(csv_file,'theoretical',save_path=None)

    # plot_rsa_vs_closeness_advanced(csv_file,'empirical',save_path=None, highlight_residues=y6a_res)
    # plot_rsa_vs_closeness_advanced(csv_file,'theoretical',save_path=None, highlight_residues=y6a_res)

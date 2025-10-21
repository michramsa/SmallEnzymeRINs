This Github contains the analysis code used for the paper -INSERT FINAL TITLE HERE-. This is a “cleaned” version of the code (includes all intermediary drafted files that were used to build more refined files) - that said, any possible missing parts or issues, I can be contacted directly to help with here (via Issues).

Data used within the paper can be found in Zenodo here: -INSERT LINK HERE-

## Computing Environment Used:

All work was performed on an Alienware x15 R2 laptop utilizing the VSCode IDE in a Windows Subsystem for Linux (WSL) Ubuntu environment.

- Processor: 12th Gen Intel(R) Core(TM) i7-12700H (2.30 GHz)
- Cores: 14
- Logical processors: 20
- Installed RAM: 32.0 GB
- System type: 64-bit operating system, x64-based processor
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Python version - Python 3.10.12

## Dependencies:

An Anaconda environment was created and used with the code. All active dependencies and their version numbers can be found in “requirements.txt”. 

Arpeggio was installed separately and ran within its own environment. Dependencies needed to run Arpeggio are NOT included within the above "requirements.txt" file. To learn how to install Arpeggio (maintained by PDBeurope), please refer to its Github found here: https://github.com/PDBeurope/arpeggio . 

## The general high-level workflow is the following:

1. Source PDB data

2. For protein contact network (PCN) creation:
- Calculate residue distances provided a given threshold in angstroms
- Provided within the threshold, utilize Biopython and NetworkX to create PCN based on gathered information

3. For residue interaction graph (RIG) creation:

- Utilize Arpeggio’s cleaning function to create cleaned mmCIF files of the PDBs of interest
- Run Arpeggio on the cleaned mmCIF files, obtain JSON files
- Read through the JSON files
- Provided an interaction exists between two residues, use NetworkX to build the RIG

4. Perform various analyses based on the gathered PCNs and RIGs

- Small-world index 
- Standardized degree and closeness centrality values based on created PCNs and RIGs 
- Creation of CSVs to hold PCN/RIG data such as keeping track of individual residue degree or closeness centrality values

---

# File Directory (CTRL+F a filename for a description): 

acr_small_world.py - Holds the individual function calls for the small world analysis, using functions found in the small_world_networkx.py file.

small_world_networkx.py - Holds functions for calculating the small-world values (sigma, omega, and SWI). Relies on NetworkX for these calculations. 

enhanced_small_world_analysis.py - Performs the main analysis and creates figures to show small-world values. WARNING: This file takes a long time to run. Notably, PDB: 6VRB is roughly 24 hrs+ to run.

arpeggio_PISCES.py - A file that contains functions to take the PDB files found using PISCES and run pdbe-arpeggio on them. Consider this as a “draft” of the process used in redo_small_protein_RIGs.py and parallel_arpeggio.py.

redo_small_protein_RIGs.py - A file containing the run for the specific proteins used within the paper. Has a max size restriction on the file size to allow processing on software.

parallel_arpeggio.py - A parallel implementation that mimics the arpeggio_PISCES.py file, just can process multiple ones at a given time. This is more refined and what was used primarily for the paper.

calculate_residue_averages.py - Calculates the averages and standard deviations of the degree, degree centrality, closeness centrality values for each residue. Calculations are based on corresponding CSV files for each protein and combines it into one single text file.

create_PCN.py - Creates PCN provided a PDB. If NMR data is entered, it generally uses the first model (standard assignment for the “best” model chosen by those who created the structure). 

create_PCN_with_rna.py - Alternative version of create_PCN.py that takes into account PDB files which include RNA information.

create_RIG.py - Creates RIG provided a JSON (the output you’d get from pdbe-arpeggio). 

parallel_rig_analysis - A parallel implementation that mimics the rig_analysis.py file, just can process multiple ones at a given time. 

parallel_rig.py - A parallel implementation that mimics the create_RIG.py file, just can process multiple ones at a given time.

retrieve_PISCES_data.py - Uses the output FASTA file from the PISCES server, and then prompts the PDB API to batch download the data. The chosen PDB files are also filtered here (such as checking for mutants, removing membrane proteins, etc.).

rig_analysis.py - Calculations for degree and closeness centrality in batch and as a singular process for graphs. RSA calculations also happen here, but utilize the original, raw PDB files. 

comparison_plots.py - This code creates the figures for the degree and closeness centrality distributions (Figs 1-6). 

sasa_centrality_plots.py - This file creates the figures for RSA/Closeness with or without the residues of interest highlighted, corresponding to Figures 7,8,9.

batch_download.sh - This file is sourced from the RCSB website to help with mass downloading of PDB files. See https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script .

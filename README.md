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

Citation: Harry C Jubb, Alicia P Higueruelo, Bernardo Ochoa-Montaño, Will R Pitt, David B Ascher, Tom L Blundell, Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. Journal of Molecular Biology, Volume 429, Issue 3, 2017, Pages 365-371, ISSN 0022-2836.

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
# Data (roughly 9 GB)

This work uses data from the Protein Data Bank. Data provided by the RCSB PDB (rcsb.org) [1]. 

Citation: H.M. Berman, J. Westbrook, Z. Feng, G. Gilliland, T.N. Bhat, H. Weissig, I.N. Shindyalov, P.E. Bourne. (2000) The Protein Data Bank Nucleic Acids Research, 28: 235-242.

Important Files (Size: 966 MB) - This folder contains files used in paper. 

- 3 main Acr data: centrality and node information for both PCNs and RIGs of AcrIF1, AcrIIA1, and AcrVIA1
- 3 main Acr PDB files: PDB files associated with AcrIF1, AcrIIA1, and AcrVIA1
- Analysis Data: Organism distribution and amino acid analysis distribution
- Cleaned Centralities: Centrality information for all RIGs for the PISCES data set
- Cleaned JSONs: JSON representation for all RIGs for the PISCES data set
- Cleaned PDBs: the converted mmCIF files from the PDBe-Arpeggio cleaning process.
- Cleaned RIGs: RIGs created using cleaned PDBs 
- Cleaned SASAs: SASA and RSA information for each protein in PISCES data set
- mmCIF files: PDBe-Arpeggio cleaning returns a mmCIF, which is then converted to a PDB.

PISCES (Size: 7.54 GB) - This folder contains a mix of used and unused files of the paper and remnants of troubleshooting and testing. Some folders that are explicitly used were copied into Important Files for keeping it clear what was and was not used within the paper. 

Separating the data was an afterthought and many of the .py files path to this folder, and so it must be included. 

[deprecated] implies the file was a testing file or from an old or failed run.

- Centralities: same as Cleaned Centralities (Important Files)
- CIFs: same as mmCIF files (Important Files)
- CIFs_large: unused data due to size constraints
- GraphMLs: same as Cleaned RIGs (Important Files)
- JSONs: same as Cleaned JSONs (Important Files)
- known acr data: PCNs and RIGs for AcrIF1, AcrIIA1, and AcrVIA1
- PDBs: PDB files pulled using Protein Data Bank script
- PDBs_cleaned: [deprecated] from an old run
- PDBs_compressed: compressed versions of the PDBs
- SASAs: same as Cleaned SASAs
- PISCES_culled.fasta: the PISCES call returned some PDBs that had to be filtered for size or for lacking data, these are the remaining PDBs kept for use
- PISCES_data_info_mutant.csv: contains information about proteins which are considered 'mutants' out of the used PDBs
- PISCES_data_info: contains information about all proteins pulled (mutants and non-mutants)
- PISCES_pdbs.txt: a text file with all of the PDB ids used

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
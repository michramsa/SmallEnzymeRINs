Last updated: 7/9/2026

This Github contains the analysis code used for the paper "". 

This is a “cleaned” version of the code (includes all intermediary drafted files that were used to build more refined files) - that said, any possible missing parts or issues, I can be contacted directly to help with here (via Issues).

[Data used within the paper can be found in Zenodo here.]()

## Computing Environment Used:

All work was performed on an Alienware x15 R2 laptop utilizing the VSCode IDE in a Windows Subsystem for Linux (WSL) Ubuntu environment.

- Processor: 12th Gen Intel(R) Core(TM) i7-12700H (2.30 GHz)
- Cores: 14
- Logical processors: 20
- Installed RAM: 32.0 GB
- System type: 64-bit operating system, x64-based processor
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Python version - Python 3.10.12

## AI Use Disclosure (Same as paper):

Claude Sonnet 4.6 and Opus 4.8 were utilized throughout the methodology in the following ways: assistance with data organization (i.e., PDB files to CSV files with data), figure generation (matplotlib), for help with LaTeX formatting for tables or equations, coding/commenting (i.e., creating functions to enable batch processing/parallelization and increase efficiency of larger functions, creation of helper functions to utilize package functions on existing data, descriptions of functions), and general debugging purposes (i.e. package version issues and incompatibilities in venv). All output has been reviewed by the first author. 

## Dependencies:

An Anaconda environment was created and used with the code. All active dependencies and their version numbers can be found in “requirements.txt”. 

Arpeggio was installed separately and ran within its own environment. Dependencies needed to run Arpeggio are NOT included within the above "requirements.txt" file. To learn how to install Arpeggio (maintained by PDBeurope), please refer to [its Github found here](https://github.com/PDBeurope/arpeggio). 

Citation: Harry C Jubb, Alicia P Higueruelo, Bernardo Ochoa-Montaño, Will R Pitt, David B Ascher, Tom L Blundell, Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. Journal of Molecular Biology, Volume 429, Issue 3, 2017, Pages 365-371, ISSN 0022-2836.

## The general high-level workflow is the following:

1. Source PDB/CIF data
  - This can be done using the batch_download.sh script (original source: [RSCB PDB website](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script))

3. For residue interaction network (RIN) creation:

- Utilize Arpeggio’s cleaning function to create cleaned mmCIF files of the PDBs of interest
  - Files: 0_1_parallel_arpeggio_CIF.py (if starting with CIF structure files), 0_1_parallel_arpeggio_PDB_ver3.py (if starting with PDB structure files)
- Run Arpeggio on the cleaned mmCIF files, obtain JSON files
- Read through the JSON files
- Provided an interaction exists between two residues, use NetworkX to build the RIN
  - Files: parallel_json_to_rin.py

4. Perform various analyses based on the gathered RINs

- Standardized centrality values (degree, closeness, residue, PageRank) using NetworkX built-in functions
- Creation of CSVs to hold RIN data (individual residue degree or closeness centrality values) using [BioLiP](https://zhanggroup.org/BioLiP/).
- SASA calculations using BioPython (on raw PDB/CIF structure files)

---
# Data

This work uses data from the Protein Data Bank. Data provided by the RCSB PDB (rcsb.org) [1]. 

Citation: H.M. Berman, J. Westbrook, Z. Feng, G. Gilliland, T.N. Bhat, H. Weissig, I.N. Shindyalov, P.E. Bourne. (2000) The Protein Data Bank Nucleic Acids Research, 28: 235-242.

Branch: code_history 

This contains old and deprecated iterations of files associated with the project. This is shared mainly for transparency purposes.

Branch: main

0_2_final_check/audits - This folder contains general information about the created RINs as well as each PDB ID within both the reference (ref) and test sets.

0_2_final_check/main - This folder contains files used in paper to generate main results.

# 0_2_final_check/main File Directory (CTRL+F a filename for a description): 

0_1_parallel_arpeggio_CIF.py AND 0_1_parallel_arpeggio_PDB_ver3.py - Uses structure files (CIF or PDB, respectively) and runs Arpeggio. The separate Arpeggio environment described above in the "Dependencies" section is needed for this.
- Dependency: clean_pdb.py (see Arpeggio), clean_cif.py (included)
- 
SASA.py - For computation of SASA and RSA for a set of provided PDB/CIF protein structure files.

biolip_residue_labels.py - UPDATE(?), may be deprecated

cent_figs_by_funct_type.py - Helper file for figures.py.

check_graphml_connectivity.py - Checks for disconnected graphs.

clean_cif.py - Adaptation of the clean_pdb.py file created by Jubb. This file is needed for Arpeggio.

figures.py - File that generates the figures for the paper.

functional_breakdown.py, functional_breakdown_by_ec.py, functional_distribution_by_amino_acid.py - Exploratory files to see figures associated with the distribution of enzymes within the test set.

general_network.py - Provides whole graph topology for RINs in reference or test sets.

inspect_jackknife_stepsizes.py - Helps with calculating the needed parameters for jackknife_experiment.py.

jackknife_experiment.py - The user (based on their own analyses) provides a range of values for centrality values, and runs the jackknife/leave-one-out experiment. These are described within the paper as a figure and in pseudocode. The concise gist is it makes a list of candidate combinations of RSA and a centrality type, looks at each residue within the RIN, and if that residue's values are within the designated range, then it predicts it as a functional residue. Then it checks against the ground truth. It can then collect that information and use the MCC formula for each combination.

jackknife_explore.py - This is taking the information gathered by parallel_RIN_analysis.py (aka the centrality values) and provides the user information that they can use to help them choose their parameters for jackknife_experiment.py. Note: This file does not tell the user what is the "best" to use, etc. You do have to use your own interpretations to determine what thresholds you want to test.

jackknife_figures.py - Generates images related to the output of jackknife_experiment.py.

move_SASA_files.py - A helper function for rearranging SASA files.

parallel_RIN_analysis.py - Adds in all of the raw centrality measures using built-in NetworkX values and the residue centrality definition provided by Sol et al. (INSERT LINK TO PAPER). 

parallel_json_to_rin.py - Turns a JSON outputted by Arpeggio into a RIN using built-in NetworkX values.

standardization.py - Performs the standardization process using raw values. For degree centrality, it uses the reference set. For all other centrality measures, it is in reference to the protein's residues only.

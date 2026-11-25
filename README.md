Last updated: 11/25/26

This Github contains the analysis code used for the paper "Evaluating the stability of using centrality measures for functional residue prediction in residue interaction and contact networks of small enzymes". 

This is a “cleaned” version of the code (includes all intermediary drafted files that were used to build more refined files) - that said, any possible missing parts or issues, I can be contacted directly to help with here (via Issues).

[Data (created RINs and PCNs) used within the paper can be found in Zenodo here.](https://zenodo.org/records/24431440?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIzM2M0YjE1LTk0MGYtNDk3Yy04NDhiLThjOWFjODJjZDM4NyIsImRhdGEiOnt9LCJyYW5kb20iOiJhMTQ0OTI4Nzg2ZjZjZDY0ZjlhNWI4NWRjYzMxNGIwOSJ9.TAEL8DRdCaUhZoxbFjO3kR-flufJxWqUvNCM4-6r59MBHdaU9D9Ca6yBSeBc-4QYKSYFQT6mcuJ8F6Ms7dBnzw)

NOTE: The main thing to keep in mind for this code are the path and file locations, as one file may rely on the output of another file - I try to clarify this as mapping all of these files.

## Computing Environment Used:

Work was performed on an Alienware x15 R2 laptop utilizing the VSCode IDE in a Windows Subsystem for Linux (WSL) Ubuntu environment.

- Processor: 12th Gen Intel(R) Core(TM) i7-12700H (2.30 GHz)
- Cores: 14
- Logical processors: 20
- Installed RAM: 32.0 GB
- System type: 64-bit operating system, x64-based processor
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Python version - Python 3.10.12

Google Colab was also utilized for more intensive experiments (such as the efficiency tests).

## AI Use Disclosure (Same as paper):

Claude Sonnet 4.6 and Opus 4.8 were utilized throughout the methodology in the following ways: assistance with data organization (i.e., PDB files to CSV files with data), figure generation (matplotlib), for help with LaTeX formatting for tables or equations, coding/commenting (i.e., creating functions to enable batch processing/parallelization and increase efficiency of larger functions, creation of helper functions to utilize package functions on existing data, descriptions of functions), and general debugging purposes (i.e. package version issues and incompatibilities in venv). All output has been reviewed by the first author. 

## Dependencies:

An Anaconda environment was created and used with the code. All active dependencies and their version numbers can be found in “requirements.txt”. 

Arpeggio was installed separately and ran within its own environment. Dependencies needed to run Arpeggio are NOT included within the above "requirements.txt" file. To learn how to install Arpeggio (maintained by PDBeurope), please refer to [its Github found here](https://github.com/PDBeurope/arpeggio). 

Citation: Harry C Jubb, Alicia P Higueruelo, Bernardo Ochoa-Montaño, Will R Pitt, David B Ascher, Tom L Blundell, Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. Journal of Molecular Biology, Volume 429, Issue 3, 2017, Pages 365-371, ISSN 0022-2836.

## The general high-level workflow is the following:

1. Source PDB/CIF data
  - This can be done using the batch_download.sh script (original source: [RSCB PDB website](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script))

2. For residue interaction network (RIN) creation:

UPDATE:
You will still need a cleaned mmCIF file for [Arpeggio.](https://github.com/PDBeurope/arpeggio)

CITATION:
Harry C Jubb, Alicia P Higueruelo, Bernardo Ochoa-Montaño, Will R Pitt, David B Ascher, Tom L Blundell, Arpeggio: A Web Server for Calculating and Visualising Interatomic Interactions in Protein Structures. Journal of Molecular Biology, Volume 429, Issue 3, 2017, Pages 365-371, ISSN 0022-2836,

No cleaned file is needed for RING 4.0. [RING 4.0 is available here.](https://ring.biocomputingup.it/)

CITATION: RING 4.0: Faster residue interaction networks with novel interaction types across over 35,000 different chemical structures.
Del Conte, A., Camagni, G. F., Clementel, D., Minervini, G., Monzon, A. M., Ferrari, C., Piovesan, D., & Tosatto, S. C. E. (2024). Nucleic Acids Research, 52(W1), W306–W312. 

3. Perform various analyses based on the gathered RINs
UPDATE:
Residue centrality is no longer analyzed within these experiments.
SASA calculations are done via the Lee-Richards algorithm, [using FreeSASA.](https://freesasa.github.io/)

---


# Data
--------------------------UPDATE:

0_4_revisions - This folder contains all code associated with recent experiments after reviewer feedback.

Subfolders of 0_4_revisions:

1. 0_RIN_construction - Files associated with creating the RINs and PCNs. (Data file: biolip_residue_labels.csv included here). Arpeggio provides a JSON. RING 4.0 makes it's own data files (.ringNodes, .ringEdges).
2. 2_experiments - Files associated with all experiments (efficiency, Jackknife/LOO).


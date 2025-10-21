# we do need a dictionary of the amino acids because not all amino acid abbreviations match their full name
amino_acids = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}

# these labels come from the json files from interpro and the physiochemical properties from the AcrRanker/Eitzinger 2019 paper

# sourced from Table S3 of the Supplemental of AcRanker
# for volume, denote volume scale '-' as 0, and '+' as 1

# cysteine is separated due to its dipole moment being somewhat context dependent 
# might need to think about ordinal vs. one-hot encoding here
acranker_labels = {1:{'dipole_scale':{0.9},'amino':['A', 'G', 'V'], 'volume':{0}},
                   2:{'dipole_scale':{0.9},'amino': ['I', 'L', 'F', 'P'], 'volume':{1}},
                   3:{'dipole_scale':{1.1},'amino': ['Y', 'M', 'T', 'S'], 'volume':{1}},
                   4:{'dipole_scale':{2.1},'amino':['H', 'N', 'Q', 'W'], 'volume':{1}},
                   5:{'dipole_scale':{3.1},'amino': ['R', 'K'], 'volume':{1}},
                   6:{'dipole_scale':{-3.1},'amino': ['D', 'E'], 'volume': {1}},
                   7:{'dipole_scale':{0},'amino':['C'], 'volume':{1}}
}
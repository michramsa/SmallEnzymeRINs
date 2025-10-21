import os 
import numpy
import networkx as nx
import pandas as pd
import subprocess
import re
import requests
import io
import gzip
import json
import matplotlib as plt
import shutil
import time
import concurrent.futures
import multiprocessing as mp
import csv
import random
import glob
import seaborn as sns


from datetime import datetime
from Bio.PDB import *
from Bio import SeqIO
from urllib.request import urlopen
from pathlib import Path
from functools import partial
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from networkx.algorithms import bipartite
from matplotlib.colors import ListedColormap

from pathlib import Path

BASE_DIR = Path(__file__).parent
AMITAI_DIR = BASE_DIR / "Amitai"
FIGS_DIR = BASE_DIR / "Figure creation"
GEN_DATA_DIR = BASE_DIR / "General Data Analysis"
IMP_DIR = BASE_DIR / "Important Files"
KNOWN_DIR = BASE_DIR / "known acrs"
PISCES_DIR = BASE_DIR / "PISCES"



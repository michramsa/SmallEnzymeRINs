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
import matplotlib.pyplot as plt 
import shutil
import time
import concurrent.futures
import multiprocessing as mp
import csv
import random
import glob
import seaborn as sns
import logging
import argparse
import sys


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

BASE_DIR = Path(__file__).parent

# 6/27/26 Final Cleanup 
FINAL_DIR = BASE_DIR / "0_2_final_check"
DATA_DIR_3 = BASE_DIR / "0_2_final_check" / "data"
MAIN_DIR = BASE_DIR / "0_2_final_check" / "main"
FIGS_DIR = BASE_DIR / "0_2_final_check" / "figures"



# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

with open("/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/JSONs/1CBS_clean.json") as f:
    data = json.load(f)
print(data[0])
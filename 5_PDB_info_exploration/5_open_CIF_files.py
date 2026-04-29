# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

from concurrent.futures import ThreadPoolExecutor

def decompress_file(gz_path, output_folder):
    out_filename = os.path.basename(gz_path).replace(".gz", "")
    out_path = os.path.join(output_folder, out_filename)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

input_folder = DATA_DIR / "enzymes" / "enzymes_CIFs_compressed"
output_folder = DATA_DIR / "enzymes" / "enzymes_CIFs"
os.makedirs(output_folder, exist_ok=True)

gz_files = glob.glob(os.path.join(input_folder, "*.cif.gz"))

with ThreadPoolExecutor() as executor:
    executor.map(lambda f: decompress_file(f, output_folder), gz_files)
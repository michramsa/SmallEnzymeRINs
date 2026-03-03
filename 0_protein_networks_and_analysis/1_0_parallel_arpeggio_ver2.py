# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import os, subprocess, logging
from datetime import datetime
DATA_DIR = Path("/home/epi2melabs/AcrNetworks-clean-v2/3_data")  #### REMOVE BEFORE PUBLIC POSTING

# This file runs the Arpeggio pipeline steps that require the Arpeggio conda environment:
# 1. Clean PDB files using clean_pdb.py (harryjubb/pdbtools)
# 2. Convert cleaned PDBs to mmCIF using gemmi
# 3. Run PDBe-Arpeggio on the CIF files -> outputs JSON
#
# The JSON-to-RIG conversion is handled separately in 2_json_to_rig.py,
# which can be run in any environment with networkx.

def setup_logger(log_dir):
    """
    Sets up a logger that writes to both terminal and a timestamped log file.
    
    Parameters:
    - log_dir: Directory to save the log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"arpeggio_run_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),    # writes to file
            logging.StreamHandler()           # still prints to terminal
        ]
    )
    logging.info(f"Log file: {log_file}")
    return logging.getLogger()


def log_step(logger, step, acr_type, structure_name, status, error=None):
    """
    Logs a single pipeline step with consistent formatting.
    
    Parameters:
    - step:           e.g. "CLEAN", "CIF", "ARPEGGIO"
    - acr_type:       e.g. "AcrIA"
    - structure_name: e.g. "1abc"
    - status:         "SKIP", "START", "SUCCESS", "FAILED"
    - error:          optional error message string
    """
    msg = f"[{step:<10}] acr_type={acr_type} | file={structure_name} | status={status}"
    if error:
        msg += f" | error={error}"
    
    if status == "FAILED":
        logger.error(msg)
    elif status == "SKIP":
        logger.debug(msg)
    else:
        logger.info(msg)

def get_all_pdb_files(base_dir):
    """
    Recursively find all PDB files in the directory structure.
    
    Parameters:
    - base_dir: Base directory containing Acr_PDB folder structure
    
    Returns:
    - List of tuples: (pdb_file_path, acr_type, structure_name)
    """
    pdb_files = []
    acr_pdb_dir = Path(base_dir) / "all_acrs" / "Acr_PDB"
    
    if not acr_pdb_dir.exists():
        print(f"Warning: {acr_pdb_dir} does not exist")
        return pdb_files
    
    # Iterate through Acr type directories (e.g., AcrIA, AcrIB, etc.)
    for acr_type_dir in acr_pdb_dir.iterdir():
        if acr_type_dir.is_dir():
            acr_type = acr_type_dir.name
            
            # Find all .pdb files in this Acr type directory
            for pdb_file in acr_type_dir.glob("*.pdb"):
                structure_name = pdb_file.stem  # Filename without extension
                pdb_files.append((pdb_file, acr_type, structure_name))
    
    return pdb_files


if __name__ == "__main__":

    ## Path to clean_pdb.py from harryjubb/pdbtools
    ## Clone if needed:
    # subprocess.run("git clone https://github.com/harryjubb/pdbtools.git /path/to/pdbtools", shell=True)
    CLEAN_PDB_SCRIPT = Path("/home/epi2melabs/projects/pdbtools/clean_pdb.py")

    ## Output directories - all organized under all_acrs, with per-acr_type subdirectories
    cleaned_dir = DATA_DIR / "arp_results" / "PDBs_cleaned"
    cifs_dir    = DATA_DIR / "arp_results" / "CIFs"
    jsons_dir   = DATA_DIR / "arp_results" / "JSONs"

    ## Gather all PDB files from nested folder structure
    pdb_files = get_all_pdb_files(DATA_DIR)
    print(f"Found {len(pdb_files)} PDB files across all Acr types")
    
    ## mini test - comment out when finished testing
    # pdb_files = pdb_files[:2]
    # print(f"test: run on {len(pdb_files)} files: {[s for _, _, s in pdb_files]}")
    ## end mini test

    logger = setup_logger(DATA_DIR / "arp_results" / "logs")
    logger.info(f"Found {len(pdb_files)} PDB files across all Acr types")

    ## Track failures per step for summary at end
    failed_clean = []
    failed_cif   = []
    failed_arp   = []

    for pdb_file, acr_type, structure_name in pdb_files:

        ## Create per-acr_type subdirectories for each output stage
        clean_out_dir = cleaned_dir / acr_type
        cif_out_dir   = cifs_dir    / acr_type
        json_out_dir  = jsons_dir   / acr_type

        for d in [clean_out_dir, cif_out_dir, json_out_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # clean PDBs using clean_pdb.py (harryjubb/pdbtools)
        # ----------------------------------------------------------------
        clean_pdb_path = clean_out_dir / f"{structure_name}_clean.pdb"

        if clean_pdb_path.exists():
            log_step(logger, "CLEAN", acr_type, structure_name, "SKIP")
        else:
            log_step(logger, "CLEAN", acr_type, structure_name, "START")
            result = subprocess.run(
                f"python3 {CLEAN_PDB_SCRIPT} -kh '{str(pdb_file)}'",
                shell=True,
                capture_output=True,
                text=True
            )
            # clean_pdb.py creates {stem}.clean.pdb next to the source file
            expected_output = pdb_file.parent / f"{structure_name}.clean.pdb"

            if result.returncode == 0 and expected_output.exists():
                os.rename(expected_output, clean_pdb_path)
                log_step(logger, "CLEAN", acr_type, structure_name, "SUCCESS")
            else:
                log_step(logger, "CLEAN", acr_type, structure_name, "FAILED", error=result.stderr)
                failed_clean.append((acr_type, structure_name))
                continue  # skip remaining steps for this file

        # ----------------------------------------------------------------
        # convert cleaned PDBs to mmCIF using gemmi
        # ----------------------------------------------------------------
        cif_path = cif_out_dir / f"{structure_name}_clean.cif"
        if cif_path.exists():
            log_step(logger, "CIF", acr_type, structure_name, "SKIP")
        else:
            log_step(logger, "CIF", acr_type, structure_name, "START")
            result = subprocess.run(
                ["gemmi", "convert", str(clean_pdb_path), str(cif_path)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                log_step(logger, "CIF", acr_type, structure_name, "FAILED", error=result.stderr)
                failed_cif.append((acr_type, structure_name))
                continue
            else:
                log_step(logger, "CIF", acr_type, structure_name, "SUCCESS")

        # ----------------------------------------------------------------
        # run PDBe-Arpeggio on the CIF file
        # note: arpeggio should be run from its own conda environment
        # you can stop this at any time - if you keyboard interrupt, check the
        # output folder and delete any JSON files with size 0 KB
        # ----------------------------------------------------------------
        arpeggio_stem = cif_path.name.split(".")[0]
        json_path = json_out_dir / f"{arpeggio_stem}.json"
        print("JSON output path: ", json_path)
        if json_path.exists():
            log_step(logger, "ARPEGGIO", acr_type, structure_name, "SKIP")
        else:
            log_step(logger, "ARPEGGIO", acr_type, structure_name, "START")
            try:
                subprocess.run(
                    ["pdbe-arpeggio", str(cif_path), "-sa", "-co", "1.0", "-o", str(json_out_dir)],
                    check=True,
                    timeout=1800
                )
                log_step(logger, "ARPEGGIO", acr_type, structure_name, "SUCCESS")
            except Exception as e:
                log_step(logger, "ARPEGGIO", acr_type, structure_name, "FAILED", error=str(e))
                failed_arp.append((acr_type, structure_name))
                continue

    logger.info("=== ARPEGGIO PIPELINE COMPLETE ===")
    logger.info(f"Failed cleaning ({len(failed_clean)}):  {failed_clean}")
    logger.info(f"Failed CIF conv ({len(failed_cif)}):  {failed_cif}")
    logger.info(f"Failed arpeggio ({len(failed_arp)}):  {failed_arp}")
    logger.info("Run 2_json_to_rig.py next to convert JSONs to RIG GraphML files.")
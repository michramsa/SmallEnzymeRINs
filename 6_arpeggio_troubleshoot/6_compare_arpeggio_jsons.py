"""
compare_arpeggio_jsons.py
--------------------------
Semantically compares two Arpeggio JSON outputs:
  - CIF method : produced via clean_cif_ver2.py  (mmCIF-native path, gemmi writer)
  - PDB method : produced via clean_pdb.py       (legacy PDB path)

Accounts for the following known behavioural differences between the two
cleaning scripts (clean_cif_ver2.py vs clean_pdb.py):

  1. Alt-loc selection:
       clean_pdb.py    → BioPython default disordered_get() = highest occupancy
       clean_cif_ver2  → explicit altloc 'A', fallback first available
     When the highest-occupancy conformer is NOT 'A', different atom
     coordinates are selected, producing different contacts.

  2. MSE → MET conversion:
       Both scripts convert MSE→MET and SE→SD.  However, clean_pdb.py only
       converts residues that are members of a polypeptide, while
       clean_cif_ver2.py converts ALL residues with resname MSE or MET
       (regardless of polypeptide membership).  This may cause differences
       for non-polypeptide selenomethionine residues.

  3. HETATM → ATOM reclassification:
       clean_pdb.py explicitly sets record='ATOM' for polypeptide residues,
       overriding HETATM.  clean_cif_ver2.py writes via gemmi which preserves
       the original mmCIF group_PDB field.  Arpeggio may treat HETATM vs ATOM
       residues differently (e.g. interacting_entities classification), so
       contacts involving non-standard polypeptide residues may differ.

  4. Hydrogen handling defaults:
       clean_pdb.py    → removes H by default (needs -kh to keep)
       clean_cif_ver2  → keeps H by default   (needs -nh to remove)
     If flags are not matched, hydrogen-involving contacts will only appear
     in one method's output.

  5. Atom name padding:
       clean_pdb.py applies a 3-char atom name fix (' ' + atom.name).
       Gemmi handles atom name padding independently.
     This could cause atom-name mismatches in contact keys for atoms
     with 3-character names.

Edit the parameters in the `if __name__ == "__main__":` block at the
bottom of this file, then run:
    python compare_arpeggio_jsons.py

Outputs (written to OUT_DIR):
  - {PDB_ID}_contacts_only_in_cif.csv     contacts found by CIF method but not PDB
  - {PDB_ID}_contacts_only_in_pdb.csv     contacts found by PDB method but not CIF
  - {PDB_ID}_flag_differences.csv         contacts in both, but with differing flags
  - {PDB_ID}_summary.txt                  copy of the printed summary
"""

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *


# ===========================================================================
# JSON loading
# ===========================================================================

def load_json(path: str) -> list:
    """Load an Arpeggio JSON file and return the contacts as a flat list."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # Some Arpeggio versions wrap contacts under a key
    for key in ("contacts", "results", "data"):
        if key in data:
            return data[key]

    raise ValueError(
        f"Unrecognised JSON structure in {path}. "
        f"Top-level keys: {list(data.keys())}"
    )


# ===========================================================================
# Contact key construction
# ===========================================================================

def normalise_atom_name(name: str) -> str:
    """
    Strip and normalise an atom name so that padding differences
    (clean_pdb.py's 3-char fix vs gemmi's padding) do not cause
    false mismatches.

    Difference #5: clean_pdb.py prepends a space to 3-character atom
    names (' ' + atom.name).  Gemmi may pad differently.  Stripping
    ensures 'CA' == ' CA' == 'CA '.
    """
    return name.strip()


def atom_id(atom: dict) -> str:
    """
    Build a canonical string identifier for one atom endpoint.

    Arpeggio's JSON contacts each have two atom entries.
    The key names vary slightly by Arpeggio version; this function
    tries the most common conventions in order.

    Format returned:  CHAIN/RESNUM[ICODE]/RESNAME/ATOMNAME
    e.g.              A/42/LEU/CA
    """
    # --- Arpeggio mmCIF output (confirmed schema from test output) ---
    if "auth_asym_id" in atom:
        chain   = atom.get("auth_asym_id", "?")
        seq     = str(atom.get("auth_seq_id", "?"))
        ins     = (atom.get("pdbx_PDB_ins_code") or "").strip()
        resname = atom.get("label_comp_id", "?")
        aname   = normalise_atom_name(atom.get("auth_atom_id", "?"))

    # --- Fallback: older Arpeggio flat key style ---
    elif "chain" in atom or "chain_id" in atom:
        chain   = atom.get("chain", atom.get("chain_id", "?"))
        seq     = str(atom.get("res_id", atom.get("seq_id", "?")))
        ins     = (atom.get("ins_code") or "").strip()
        resname = atom.get("res_name", atom.get("comp_id", "?"))
        aname   = normalise_atom_name(atom.get("atom_name", atom.get("atom_id", "?")))

    else:
        raise KeyError(
            f"Cannot parse atom dict — unrecognised keys: {list(atom.keys())}"
        )

    return f"{chain}/{seq}{ins}/{resname}/{aname}"


def contact_key(contact: dict) -> tuple:
    """
    Return an order-normalised tuple key for a contact so that
    (atomA->atomB) and (atomB->atomA) hash identically.

    Tries bgn/end keys first, then atom1/atom2, then positional fallback.
    """
    if "bgn" in contact and "end" in contact:
        a = atom_id(contact["bgn"])
        b = atom_id(contact["end"])

    elif "atom1" in contact and "atom2" in contact:
        a = atom_id(contact["atom1"])
        b = atom_id(contact["atom2"])

    else:
        atom_vals = [v for v in contact.values() if isinstance(v, dict)]
        if len(atom_vals) >= 2:
            a = atom_id(atom_vals[0])
            b = atom_id(atom_vals[1])
        else:
            raise ValueError(
                f"Cannot parse contact — no atom pair found. "
                f"Keys present: {list(contact.keys())}"
            )

    return tuple(sorted([a, b]))


def get_flags(contact: dict) -> dict:
    """
    Extract all interaction fields from a contact dict,
    excluding the atom endpoint dicts and the numeric distance.

    In this version of Arpeggio the flag fields are:
      - contact              : list of strings e.g. ['proximal'],
                               ['hbond', 'weak_hbond']
      - interacting_entities : string e.g. 'INTRA_SELECTION', 'INTER'
      - type                 : string e.g. 'atom-atom'
    """
    skip = {"bgn", "end", "atom1", "atom2", "distance", "dist"}
    return {k: v for k, v in contact.items() if k not in skip}


# ===========================================================================
# Analysis helpers
# ===========================================================================

def flag_type_summary(keys: set, lookup: dict) -> dict:
    """
    For a set of contact keys, tally how often each contact type appears.

    Handles Arpeggio's actual flag formats:
      - contact field is a list of strings: each string is counted separately
        e.g. ['hbond', 'weak_hbond'] increments both 'hbond' and 'weak_hbond'
      - interacting_entities / type are strings: counted as-is
      - legacy boolean flags (True/1) are also handled for compatibility
    """
    tally = defaultdict(int)
    for key in keys:
        for field, val in get_flags(lookup[key]).items():
            if isinstance(val, list):
                # e.g. contact: ['proximal'] or ['hbond', 'vdw_clash']
                for item in val:
                    tally[str(item)] += 1
            elif isinstance(val, str):
                tally[f"{field}={val}"] += 1
            elif val is True or val == 1:
                tally[field] += 1
    return dict(sorted(tally.items()))


def contacts_to_rows(keys: set, lookup: dict) -> list:
    """
    Convert a set of contact keys into a list of flat dicts suitable
    for writing to CSV.
    """
    rows = []
    for key in sorted(keys):
        row = {"contact_key": str(key)}
        row.update(get_flags(lookup[key]))
        rows.append(row)
    return rows


def write_csv(rows: list, path: str):
    """Write a list of dicts to CSV.  Writes a placeholder if empty."""
    if not rows:
        with open(path, "w") as f:
            f.write("(no entries)\n")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Difference #1: Alt-loc divergence detector
# ===========================================================================

def altloc_divergence_report(only_cif: set, only_pdb: set) -> dict:
    """
    Detect contacts that differ between CIF and PDB methods likely due
    to alt-loc selection strategy differences.

    clean_pdb.py uses BioPython's disordered_get() which returns the
    highest-occupancy conformer.  clean_cif_ver2.py explicitly selects
    altloc 'A', falling back to the first available.

    When the highest-occupancy conformer is NOT 'A', different atom
    coordinates are selected.  This shifts interatomic distances and
    can cause contacts to appear in one output but not the other, or
    cause the same residue pair to have different atom-level contacts.

    Strategy: find residue pairs that appear in only_cif XOR only_pdb
    where the contact involves the same chain/resnum/resname but
    differs at the atom name level — a likely sign of different
    conformer selection.
    """
    def residue_pair(key: tuple) -> tuple:
        """Extract (chain/resnum/resname, chain/resnum/resname) from a contact key."""
        parts = []
        for atom_str in key:
            tokens = atom_str.split("/")
            # chain/resnum[icode]/resname  — drop the atom name (last token)
            parts.append("/".join(tokens[:3]))
        return tuple(sorted(parts))

    # Group unique contacts by their residue pair
    cif_by_respair = defaultdict(set)
    for k in only_cif:
        cif_by_respair[residue_pair(k)].add(k)

    pdb_by_respair = defaultdict(set)
    for k in only_pdb:
        pdb_by_respair[residue_pair(k)].add(k)

    # Residue pairs present in BOTH unique sets → likely altloc divergence
    shared_respairs = set(cif_by_respair.keys()) & set(pdb_by_respair.keys())

    return {
        "shared_residue_pairs": len(shared_respairs),
        "cif_contacts_at_shared": sum(len(cif_by_respair[rp]) for rp in shared_respairs),
        "pdb_contacts_at_shared": sum(len(pdb_by_respair[rp]) for rp in shared_respairs),
        "examples": [
            {
                "residue_pair": rp,
                "cif_atoms": sorted(cif_by_respair[rp]),
                "pdb_atoms": sorted(pdb_by_respair[rp]),
            }
            for rp in sorted(shared_respairs)[:5]
        ],
    }


# ===========================================================================
# Difference #3: HETATM / ATOM record type divergence
# ===========================================================================

def hetatm_atom_report(only_cif: set, only_pdb: set,
                       cif_lookup: dict, pdb_lookup: dict) -> dict:
    """
    Identify contacts that may differ due to HETATM vs ATOM record
    reclassification differences.

    clean_pdb.py sets record='ATOM' for all polypeptide residues,
    overriding their original HETATM status.  clean_cif_ver2.py
    preserves original mmCIF group_PDB values via gemmi.

    This affects Arpeggio's interacting_entities classification
    (e.g. INTER vs INTRA_SELECTION vs SELECTION).  We detect this
    by looking for contacts that exist in both outputs (same atoms)
    but with different interacting_entities flags, or contacts
    unique to one side involving non-standard residues.
    """
    # Common non-standard residues that PPBuilder may still include
    # in polypeptides (modified amino acids, etc.)
    NONSTANDARD_3LETTER = {
        'MSE', 'SEP', 'TPO', 'CSE', 'SEC', 'MLY', 'ALY', 'HYP',
        'PTR', 'LLP', 'CSO', 'OCS', 'CME', 'CSD', 'KCX', 'M3L',
        'CGU', 'NEP', 'MEQ',
    }

    def involves_nonstandard(key: tuple) -> bool:
        for atom_str in key:
            tokens = atom_str.split("/")
            if len(tokens) >= 3 and tokens[2] in NONSTANDARD_3LETTER:
                return True
        return False

    cif_nonstandard = {k for k in only_cif if involves_nonstandard(k)}
    pdb_nonstandard = {k for k in only_pdb if involves_nonstandard(k)}

    return {
        "cif_only_involving_nonstandard_residues": len(cif_nonstandard),
        "pdb_only_involving_nonstandard_residues": len(pdb_nonstandard),
        "cif_examples": sorted(cif_nonstandard)[:5],
        "pdb_examples": sorted(pdb_nonstandard)[:5],
    }


# ===========================================================================
# Difference #4: Hydrogen contact detector
# ===========================================================================

def hydrogen_contact_report(only_cif: set, only_pdb: set) -> dict:
    """
    Identify contacts unique to one method that involve hydrogen atoms.

    clean_pdb.py removes hydrogens by default (requires -kh to keep).
    clean_cif_ver2.py keeps hydrogens by default (requires -nh to remove).

    If the flags were not explicitly matched, all hydrogen-involving
    contacts will appear in only one method's output.
    """
    def involves_hydrogen(key: tuple) -> bool:
        for atom_str in key:
            tokens = atom_str.split("/")
            aname = tokens[-1] if tokens else ""
            # Hydrogen atoms typically start with H or are 1H, 2H, 3H etc.
            if aname.startswith("H") or (len(aname) >= 2 and aname[0].isdigit() and aname[1] == 'H'):
                return True
        return False

    cif_h = {k for k in only_cif if involves_hydrogen(k)}
    pdb_h = {k for k in only_pdb if involves_hydrogen(k)}

    return {
        "cif_only_hydrogen_contacts": len(cif_h),
        "pdb_only_hydrogen_contacts": len(pdb_h),
        "total_cif_only": len(only_cif),
        "total_pdb_only": len(only_pdb),
        "cif_pct_explained_by_H": (
            f"{100 * len(cif_h) / max(len(only_cif), 1):.1f}%"
        ),
        "pdb_pct_explained_by_H": (
            f"{100 * len(pdb_h) / max(len(only_pdb), 1):.1f}%"
        ),
    }


# ===========================================================================
# Difference #2 (updated): MSE / MET — both scripts now convert,
# but clean_pdb.py only converts polypeptide members while
# clean_cif_ver2.py converts ALL MSE residues
# ===========================================================================

def mse_met_report(only_cif: set, only_pdb: set) -> dict:
    """
    Check for residual MSE/MET naming differences.

    Both clean_pdb.py and clean_cif_ver2.py convert MSE→MET, but they
    differ in scope:
      - clean_pdb.py:    only converts if `residue in polypeptide_residues`
      - clean_cif_ver2:  converts ALL residues with resname in ('MSE', 'MET')

    For structures with non-polypeptide selenomethionine (e.g. free MSE
    as a ligand), clean_pdb.py would leave it as MSE while
    clean_cif_ver2.py converts it to MET.  Additionally, clean_pdb.py
    converts SE→SD atom name only when element is also 'SE', so the
    atom name 'SD' vs 'SE' may differ.

    This function also checks for any residual MSE keys that survived
    in either output (which would indicate Arpeggio used the original
    resname rather than the cleaned one, or the cleaning didn't fully
    propagate).
    """
    cif_mse = {k for k in only_cif if any("MSE" in p for p in k)}
    pdb_mse = {k for k in only_pdb if any("MSE" in p for p in k)}
    cif_met = {k for k in only_cif if any("MET" in p for p in k)}
    pdb_met = {k for k in only_pdb if any("MET" in p for p in k)}

    # Also look for SE vs SD atom name mismatches
    cif_se = {k for k in only_cif if any("/SE" in p for p in k)}
    pdb_se = {k for k in only_pdb if any("/SE" in p for p in k)}
    cif_sd = {k for k in only_cif if any("/SD" in p for p in k)}
    pdb_sd = {k for k in only_pdb if any("/SD" in p for p in k)}

    # Cross-match: CIF has MET where PDB has MSE (non-polypeptide MSE case)
    matched_pairs = []
    for cif_key in cif_met:
        # Check if this contact exists in PDB side with MSE instead
        mse_key = tuple(sorted(
            part.replace("/MET/", "/MSE/").replace("/SD", "/SE")
            if "/MET/" in part else part
            for part in cif_key
        ))
        if mse_key in pdb_mse:
            matched_pairs.append((cif_key, mse_key))

    # Also check reverse: PDB has MET where CIF has MSE
    for pdb_key in pdb_met:
        mse_key = tuple(sorted(
            part.replace("/MET/", "/MSE/").replace("/SD", "/SE")
            if "/MET/" in part else part
            for part in pdb_key
        ))
        if mse_key in cif_mse:
            matched_pairs.append((mse_key, pdb_key))

    return {
        "cif_only_with_MSE_resname": len(cif_mse),
        "pdb_only_with_MSE_resname": len(pdb_mse),
        "cif_only_with_MET_resname": len(cif_met),
        "pdb_only_with_MET_resname": len(pdb_met),
        "cif_only_with_SE_atom": len(cif_se),
        "pdb_only_with_SE_atom": len(pdb_se),
        "cross_matched_MSE_MET_pairs": len(matched_pairs),
        "example_pairs": matched_pairs[:5],
    }


# ===========================================================================
# Main comparison
# ===========================================================================

def compare(cif_json_path: str,
            pdb_json_path: str,
            out_dir: str = ".",
            pdb_id: str = "XXXX"):

    os.makedirs(out_dir, exist_ok=True)
    lines = []  # mirror all output so we can write summary.txt

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Comparison: clean_cif_ver2.py (CIF) vs clean_pdb.py (PDB)")
    log(f"PDB ID: {pdb_id}")
    log()

    # ---- Load ----
    log(f"Loading CIF-method JSON : {cif_json_path}")
    cif_contacts = load_json(cif_json_path)
    log(f"Loading PDB-method JSON : {pdb_json_path}")
    pdb_contacts = load_json(pdb_json_path)

    log()
    log("=" * 62)
    log(f"  Total contacts  |  CIF: {len(cif_contacts):>6}  |  PDB: {len(pdb_contacts):>6}")
    log("=" * 62)

    # ---- Build lookup dicts keyed by canonical contact tuple ----
    cif_lookup: dict = {}
    for c in cif_contacts:
        cif_lookup[contact_key(c)] = c

    pdb_lookup: dict = {}
    for c in pdb_contacts:
        pdb_lookup[contact_key(c)] = c

    cif_keys = set(cif_lookup.keys())
    pdb_keys = set(pdb_lookup.keys())

    only_cif = cif_keys - pdb_keys
    only_pdb = pdb_keys - cif_keys
    in_both  = cif_keys & pdb_keys

    log()
    log("Contact overlap")
    log(f"  In both            : {len(in_both):>6}  "
        f"({100 * len(in_both) / max(len(cif_keys), 1):.1f}% of CIF, "
        f"{100 * len(in_both) / max(len(pdb_keys), 1):.1f}% of PDB)")
    log(f"  Only in CIF method : {len(only_cif):>6}")
    log(f"  Only in PDB method : {len(only_pdb):>6}")

    # ---- Flag differences among shared contacts ----
    flag_diff_rows = []
    for key in sorted(in_both):
        cf = get_flags(cif_lookup[key])
        pf = get_flags(pdb_lookup[key])
        if cf != pf:
            differing = [
                k for k in set(list(cf.keys()) + list(pf.keys()))
                if cf.get(k) != pf.get(k)
            ]
            flag_diff_rows.append({
                "contact_key"    : str(key),
                "differing_flags": ", ".join(sorted(differing)),
                "cif_values"     : str({k: cf.get(k) for k in differing}),
                "pdb_values"     : str({k: pf.get(k) for k in differing}),
            })

    log(f"  Shared, flags differ:{len(flag_diff_rows):>5}")

    # ---- Flag-type breakdown for unique contacts ----
    log()
    log("Flag breakdown — contacts ONLY in CIF method:")
    cif_summary = flag_type_summary(only_cif, cif_lookup)
    if cif_summary:
        for flag, count in cif_summary.items():
            log(f"    {flag:<42} {count:>5}")
    else:
        log("    (none)")

    log()
    log("Flag breakdown — contacts ONLY in PDB method:")
    pdb_summary = flag_type_summary(only_pdb, pdb_lookup)
    if pdb_summary:
        for flag, count in pdb_summary.items():
            log(f"    {flag:<42} {count:>5}")
    else:
        log("    (none)")

    # ==================================================================
    # KNOWN DIFFERENCES BETWEEN clean_cif_ver2.py AND clean_pdb.py
    # ==================================================================

    log()
    log("=" * 62)
    log("  KNOWN BEHAVIOURAL DIFFERENCES")
    log("=" * 62)

    # ---- Difference #4: Hydrogen contacts ----
    log()
    log("Difference #4: Hydrogen handling defaults")
    log("  clean_pdb.py removes H by default; clean_cif_ver2.py keeps H by default.")
    log("  If -kh / -nh flags were not explicitly matched, hydrogen contacts")
    log("  will appear in only one output.")
    h_report = hydrogen_contact_report(only_cif, only_pdb)
    log(f"  CIF-only contacts involving H atoms : {h_report['cif_only_hydrogen_contacts']}")
    log(f"  PDB-only contacts involving H atoms : {h_report['pdb_only_hydrogen_contacts']}")
    log(f"  % of CIF-only explained by H        : {h_report['cif_pct_explained_by_H']}")
    log(f"  % of PDB-only explained by H        : {h_report['pdb_pct_explained_by_H']}")
    if h_report['cif_only_hydrogen_contacts'] > 0 and h_report['pdb_only_hydrogen_contacts'] == 0:
        log("  >>> LIKELY CAUSE: CIF run kept hydrogens, PDB run did not (-kh not set).")
    elif h_report['pdb_only_hydrogen_contacts'] > 0 and h_report['cif_only_hydrogen_contacts'] == 0:
        log("  >>> LIKELY CAUSE: PDB run used -kh, CIF run used -nh.")

    # ---- Difference #1: Alt-loc divergence ----
    log()
    log("Difference #1: Alt-loc selection strategy")
    log("  clean_pdb.py picks highest occupancy; clean_cif_ver2.py picks altloc 'A'.")
    log("  When the highest-occupancy conformer != 'A', different coordinates")
    log("  are selected → different interatomic distances → different contacts.")
    altloc_report = altloc_divergence_report(only_cif, only_pdb)
    log(f"  Residue pairs with contacts in both unique sets : {altloc_report['shared_residue_pairs']}")
    log(f"    CIF-only contacts at those pairs              : {altloc_report['cif_contacts_at_shared']}")
    log(f"    PDB-only contacts at those pairs              : {altloc_report['pdb_contacts_at_shared']}")
    if altloc_report['examples']:
        log("  Examples (first 5):")
        for ex in altloc_report['examples']:
            log(f"    Residue pair: {ex['residue_pair']}")
            log(f"      CIF atoms : {ex['cif_atoms'][:3]}")
            log(f"      PDB atoms : {ex['pdb_atoms'][:3]}")

    # ---- Difference #2: MSE/MET ----
    log()
    log("Difference #2: MSE → MET conversion scope")
    log("  Both scripts convert MSE→MET, but clean_pdb.py only converts")
    log("  polypeptide members while clean_cif_ver2.py converts all MSE residues.")
    mse_report = mse_met_report(only_cif, only_pdb)
    log(f"  CIF-only contacts with MSE resname  : {mse_report['cif_only_with_MSE_resname']}")
    log(f"  PDB-only contacts with MSE resname  : {mse_report['pdb_only_with_MSE_resname']}")
    log(f"  CIF-only contacts with MET resname  : {mse_report['cif_only_with_MET_resname']}")
    log(f"  PDB-only contacts with MET resname  : {mse_report['pdb_only_with_MET_resname']}")
    log(f"  CIF-only with SE atom name          : {mse_report['cif_only_with_SE_atom']}")
    log(f"  PDB-only with SE atom name          : {mse_report['pdb_only_with_SE_atom']}")
    log(f"  Cross-matched MSE↔MET pairs         : {mse_report['cross_matched_MSE_MET_pairs']}")
    if mse_report['example_pairs']:
        log("  Example matched pairs:")
        for cif_k, pdb_k in mse_report['example_pairs'][:3]:
            log(f"    CIF: {cif_k}")
            log(f"    PDB: {pdb_k}")

    # ---- Difference #3: HETATM/ATOM reclassification ----
    log()
    log("Difference #3: HETATM → ATOM reclassification")
    log("  clean_pdb.py reclassifies polypeptide HETATM residues as ATOM records.")
    log("  clean_cif_ver2.py preserves original mmCIF group_PDB via gemmi.")
    log("  This may affect Arpeggio's interacting_entities classification.")
    hetatm_report = hetatm_atom_report(only_cif, only_pdb, cif_lookup, pdb_lookup)
    log(f"  CIF-only contacts involving non-standard residues : "
        f"{hetatm_report['cif_only_involving_nonstandard_residues']}")
    log(f"  PDB-only contacts involving non-standard residues : "
        f"{hetatm_report['pdb_only_involving_nonstandard_residues']}")
    if hetatm_report['cif_examples']:
        log(f"  CIF examples: {hetatm_report['cif_examples'][:3]}")
    if hetatm_report['pdb_examples']:
        log(f"  PDB examples: {hetatm_report['pdb_examples'][:3]}")

    # ---- Check interacting_entities differences in shared contacts ----
    entity_diffs = [
        row for row in flag_diff_rows
        if "interacting_entities" in row.get("differing_flags", "")
    ]
    if entity_diffs:
        log(f"  Shared contacts with interacting_entities differences: {len(entity_diffs)}")
        log("  (This is likely caused by HETATM vs ATOM record differences)")

    # ---- Difference #5: Atom name padding note ----
    log()
    log("Difference #5: Atom name padding")
    log("  clean_pdb.py applies a 3-char atom name fix (' ' + name).")
    log("  Gemmi handles padding independently.")
    log("  This comparison normalises atom names (strips whitespace)")
    log("  before key construction to avoid false mismatches.")

    # ---- Write output files ----
    p1 = os.path.join(out_dir, f"{pdb_id}_contacts_only_in_cif.csv")
    p2 = os.path.join(out_dir, f"{pdb_id}_contacts_only_in_pdb.csv")
    p3 = os.path.join(out_dir, f"{pdb_id}_flag_differences.csv")
    p4 = os.path.join(out_dir, f"{pdb_id}_summary.txt")

    write_csv(contacts_to_rows(only_cif, cif_lookup), p1)
    write_csv(contacts_to_rows(only_pdb, pdb_lookup), p2)
    write_csv(flag_diff_rows, p3)

    with open(p4, "w") as f:
        f.write("\n".join(lines))

    log()
    log("Output files written:")
    log(f"  {p1}")
    log(f"  {p2}")
    log(f"  {p3}")
    log(f"  {p4}")
    log()
    log("Done.")


# ===========================================================================
# EDIT PARAMETERS HERE, THEN RUN
# ===========================================================================

if __name__ == "__main__":

    # PDB identifier — used for output filenames
    PDB_ID = "3ERK"

    # Path to the Arpeggio JSON produced by the CIF-native cleaning method
    CIF_JSON = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/JSONs/3ERK_clean.json"

    # Path to the Arpeggio JSON produced by the legacy PDB cleaning method
    PDB_JSON = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arp_results/pdb_JSONs/3ERK_clean_PDB.json"

    # Directory where output CSVs and summary.txt will be written
    OUT_DIR = "/home/epi2melabs/AcrNetworks-clean-v2/6_arpeggio_troubleshoot/arpeggio_comparison"

    # ----------------------------------------------------------------------
    compare(
        cif_json_path=CIF_JSON,
        pdb_json_path=PDB_JSON,
        out_dir=OUT_DIR,
        pdb_id=PDB_ID,
    )

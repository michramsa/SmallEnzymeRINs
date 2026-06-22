import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

# =============================================================================
# STANDARD AA SET (mirrors 0_2_parallel_RIN_from_jsons_ver2.py)
# =============================================================================
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
}


# =============================================================================
# CORE AUDIT FUNCTION
# =============================================================================
def audit_json(json_path):
    """
    Parse a single Arpeggio JSON and count interactions at the atom level,
    applying the same residue filters used in arpeggio_to_rin, then
    classifying each surviving residue pair by whether it is proximal-only.

    Returns a dict with per-file counts, or None on read/parse error.

    Counts explained
    ----------------
    n_raw            : total atom-level records in the JSON
    n_after_residue  : records surviving water / non-standard / self-loop filters
    n_residue_pairs  : unique residue pairs (after deduplication — first contact wins,
                       matching arpeggio_to_rin behaviour)
    n_proximal_only  : unique residue pairs whose first-seen contact is proximal-only
    n_specific       : unique residue pairs with at least one specific contact type
    pct_proximal_only: n_proximal_only / n_residue_pairs * 100
    """
    try:
        with open(json_path, 'r') as f:
            interactions = json.load(f)
    except Exception as exc:
        print(f"  [ERROR] Could not read {json_path.name}: {exc}")
        return None

    n_raw           = len(interactions)
    n_after_residue = 0
    residue_pairs   = {}   # res_pair -> contact list (first occurrence only)

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']

        # -- same filters as arpeggio_to_rin --
        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            continue
        if (bgn['label_comp_id'] not in STANDARD_AA or
                end['label_comp_id'] not in STANDARD_AA):
            continue
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
                bgn['auth_seq_id'] == end['auth_seq_id'] and
                bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        n_after_residue += 1

        bgn_node = f"{bgn['auth_asym_id']}_{bgn['auth_seq_id']}"
        end_node = f"{end['auth_asym_id']}_{end['auth_seq_id']}"
        res_pair = tuple(sorted([bgn_node, end_node]))

        # first-contact-wins (matches arpeggio_to_rin deduplication)
        if res_pair not in residue_pairs:
            contact = interaction['contact']
            if not isinstance(contact, list):
                contact = [contact]
            residue_pairs[res_pair] = set(contact)

    n_residue_pairs = len(residue_pairs)

    # classify each unique pair
    n_proximal_only = sum(
        1 for contact_set in residue_pairs.values()
        if contact_set == {"proximal"}
    )
    n_specific = n_residue_pairs - n_proximal_only

    pct_proximal_only = (
        (n_proximal_only / n_residue_pairs * 100)
        if n_residue_pairs > 0 else 0.0
    )

    return {
        'n_raw':             n_raw,
        'n_after_residue':   n_after_residue,
        'n_residue_pairs':   n_residue_pairs,
        'n_proximal_only':   n_proximal_only,
        'n_specific':        n_specific,
        'pct_proximal_only': pct_proximal_only,
    }


# =============================================================================
# CONTACT-TYPE FREQUENCY AUDIT
# =============================================================================
def collect_contact_type_frequencies(json_path):
    """
    After residue-level filtering and first-contact-wins deduplication,
    count how often each contact label appears (a pair tagged
    ['proximal', 'hydrophobic'] contributes 1 to both counters).

    Returns a Counter, or None on error.
    """
    try:
        with open(json_path, 'r') as f:
            interactions = json.load(f)
    except Exception:
        return None

    seen_pairs = set()
    counter    = defaultdict(int)

    for interaction in interactions:
        bgn = interaction['bgn']
        end = interaction['end']

        if bgn['label_comp_id'] == 'HOH' or end['label_comp_id'] == 'HOH':
            continue
        if (bgn['label_comp_id'] not in STANDARD_AA or
                end['label_comp_id'] not in STANDARD_AA):
            continue
        if (bgn['auth_asym_id'] == end['auth_asym_id'] and
                bgn['auth_seq_id'] == end['auth_seq_id'] and
                bgn['pdbx_PDB_ins_code'] == end['pdbx_PDB_ins_code']):
            continue

        bgn_node = f"{bgn['auth_asym_id']}_{bgn['auth_seq_id']}"
        end_node = f"{end['auth_asym_id']}_{end['auth_seq_id']}"
        res_pair = tuple(sorted([bgn_node, end_node]))

        if res_pair in seen_pairs:
            continue
        seen_pairs.add(res_pair)

        contact = interaction['contact']
        if not isinstance(contact, list):
            contact = [contact]
        for label in contact:
            counter[label] += 1

    return counter


# =============================================================================
# MAIN
# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # EDIT THESE PATHS
    # -------------------------------------------------------------------------
    csv_path  = DATA_DIR_2 / "figures" / "controlled_fig1_recreation_ver0.7" / "test_standardized.csv"           # CSV with pdb_id column
    json_dir  = DATA_DIR_2 / "enzymes_CIF_unflagged" / "JSONs"
    out_dir   = DATA_DIR_2 / "audit"                     # audit outputs go here
    # -------------------------------------------------------------------------

    out_dir.mkdir(parents=True, exist_ok=True)

    # -- load CSV and extract unique PDB IDs --
    df       = pd.read_csv(csv_path)
    pdb_ids  = df['pdb_id'].dropna().unique().tolist()
    print(f"Loaded {len(pdb_ids)} unique PDB IDs from CSV")

    # -- match PDB IDs to JSON files --
    rows          = []
    missing       = []
    type_counter  = defaultdict(int)   # global contact-type tally

    for pdb_id in tqdm(pdb_ids, desc="Auditing JSONs"):
        json_path = json_dir / f"{pdb_id}_clean.json"

        if not json_path.exists():
            missing.append(pdb_id)
            continue

        result = audit_json(json_path)
        if result is None:
            missing.append(pdb_id)
            continue

        rows.append({'pdb_id': pdb_id, **result})

        freq = collect_contact_type_frequencies(json_path)
        if freq:
            for label, count in freq.items():
                type_counter[label] += count

    # =========================================================================
    # REPORT 1: per-file proximal-only summary
    # =========================================================================
    results_df = pd.DataFrame(rows)

    per_file_csv = out_dir / "audit_per_file.csv"
    results_df.to_csv(per_file_csv, index=False)
    print(f"\nPer-file results saved to: {per_file_csv}")

    # -- console summary --
    print("\n" + "=" * 60)
    print("PROXIMAL-ONLY EDGE AUDIT SUMMARY")
    print("=" * 60)
    print(f"  JSONs found and audited : {len(rows)}")
    print(f"  JSONs missing / errored : {len(missing)}")

    if len(rows) > 0:
        total_pairs    = results_df['n_residue_pairs'].sum()
        total_prox     = results_df['n_proximal_only'].sum()
        total_specific = results_df['n_specific'].sum()
        overall_pct    = total_prox / total_pairs * 100 if total_pairs > 0 else 0

        print(f"\n  Total unique residue pairs : {total_pairs:,}")
        print(f"  Proximal-only pairs        : {total_prox:,}  ({overall_pct:.1f}%)")
        print(f"  Pairs with specific contact: {total_specific:,}  ({100 - overall_pct:.1f}%)")

        print(f"\n  Per-file proximal-only % — distribution:")
        desc = results_df['pct_proximal_only'].describe()
        for stat in ['min', '25%', '50%', '75%', 'max', 'mean']:
            print(f"    {stat:>5}: {desc[stat]:.1f}%")

    # =========================================================================
    # REPORT 2: contact-type frequency table
    # =========================================================================
    type_df = (
        pd.DataFrame(
            list(type_counter.items()),
            columns=['contact_type', 'n_residue_pairs']
        )
        .sort_values('n_residue_pairs', ascending=False)
        .reset_index(drop=True)
    )

    type_df['pct_of_all_pairs'] = (
        type_df['n_residue_pairs'] / type_df['n_residue_pairs'].sum() * 100
    ).round(2)

    type_csv = out_dir / "audit_contact_type_frequencies.csv"
    type_df.to_csv(type_csv, index=False)

    print("\n" + "=" * 60)
    print("CONTACT TYPE FREQUENCIES (across all audited JSONs)")
    print("(a pair tagged ['proximal','hydrophobic'] counts toward both)")
    print("=" * 60)
    print(type_df.to_string(index=False))
    print(f"\nContact-type table saved to: {type_csv}")

    # =========================================================================
    # REPORT 3: missing files
    # =========================================================================
    if missing:
        missing_path = out_dir / "audit_missing_jsons.txt"
        with open(missing_path, 'w') as f:
            for pdb_id in missing:
                f.write(pdb_id + "\n")
        print(f"\nMissing/errored IDs written to: {missing_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

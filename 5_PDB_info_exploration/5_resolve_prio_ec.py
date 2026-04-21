# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

INPUT_CSV  = DATA_DIR / "enzymes" / "pdb_ids_with_ec.csv"
OUTPUT_CSV = DATA_DIR / "enzymes" / "resolved_prio_ec.csv"

RCSB_REST_URL = "https://data.rcsb.org/rest/v1/core"

SOURCE_PRIORITY = ["UniProt", "PDB Primary Data"]

def _fetch_entity_ids(pdb_id: str) -> list[str]:
    """Return all polymer entity IDs for a given PDB entry."""
    url = f"{RCSB_REST_URL}/entry/{pdb_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    except Exception as exc:
        print(f"  [{pdb_id}] Entry fetch ERROR: {exc}")
        return []


def _fetch_provenance_for_entry(pdb_id: str) -> dict[str, dict[str, str]]:
    """
    For each chain in a PDB entry, return a mapping:
        chain_id -> { ec_number -> provenance_source }

    Only EC numbers present in rcsb_enzyme_class_combined are included.
    If the same EC appears under multiple sources, the highest-priority source wins.
    """
    pdb_id = pdb_id.strip().upper()
    chain_ec_source: dict[str, dict[str, str]] = {}

    entity_ids = _fetch_entity_ids(pdb_id)
    if not entity_ids:
        return chain_ec_source

    for entity_id in entity_ids:
        url = f"{RCSB_REST_URL}/polymer_entity/{pdb_id}/{entity_id}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            entity = resp.json()
        except Exception as exc:
            print(f"  [{pdb_id}] Entity {entity_id} fetch ERROR: {exc}")
            continue

        # Chain IDs
        strand_id_str = entity.get("entity_poly", {}).get("pdbx_strand_id", "")
        chain_ids = [c.strip().upper() for c in strand_id_str.split(",") if c.strip()]

        # Build ec -> best_source map for this entity
        ec_combined = (
            entity.get("rcsb_polymer_entity", {})
                  .get("rcsb_enzyme_class_combined") or []
        )

        ec_to_source: dict[str, str] = {}
        for entry in ec_combined:
            ec  = entry.get("ec", "").strip()
            src = entry.get("provenance_source", "Unknown").strip()
            if not ec:
                continue
            # Keep the highest-priority source seen for this EC
            existing = ec_to_source.get(ec)
            if existing is None:
                ec_to_source[ec] = src
            else:
                existing_rank = SOURCE_PRIORITY.index(existing) if existing in SOURCE_PRIORITY else 999
                new_rank      = SOURCE_PRIORITY.index(src)      if src      in SOURCE_PRIORITY else 999
                if new_rank < existing_rank:
                    ec_to_source[ec] = src

        for chain_id in chain_ids:
            if ec_to_source or chain_id not in chain_ec_source:
                chain_ec_source[chain_id] = ec_to_source

        time.sleep(0.03)

    return chain_ec_source

def _select_priority_ec(
    ec_numbers: list[str],
    ec_to_source: dict[str, str],
) -> tuple[str, str]:
    """
    Given the EC numbers for a row and the source map from the API, return
    (prio_ec_number, prio_ec_source) according to the project rules:

      1. More than 3 EC numbers → ("-1", "-1")  [handled in caller]
      2. Single EC number       → that EC + its source (or "Unknown")
      3. Multiple EC numbers    → prefer UniProt; if no UniProt present,
                                  fall back to PDB Primary Data; then Unknown.
                                  All EC numbers sharing the winning source
                                  are kept (joined with "; ").
    """
    if not ec_numbers:
        return ("", "")

    # Attach source to each EC (fall back to "Unknown" if API had no data)
    annotated = [(ec, ec_to_source.get(ec, "Unknown")) for ec in ec_numbers]

    # Single EC — straightforward
    if len(annotated) == 1:
        ec, src = annotated[0]
        return (ec, src)

    # Multiple EC — find best source tier present
    winning_source = None
    for preferred in SOURCE_PRIORITY:
        if any(src == preferred for _, src in annotated):
            winning_source = preferred
            break
    if winning_source is None:
        winning_source = annotated[0][1]   # fallback: whatever the first source is

    winning_ecs = sorted(ec for ec, src in annotated if src == winning_source)
    return ("; ".join(winning_ecs), winning_source)

def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    # Parse ec_number into a list per row
    def parse_ec(val: str) -> list[str]:
        return [e.strip() for e in str(val).split(";") if e.strip()]

    df["_ec_list"] = df["ec_number"].apply(parse_ec)

    # Identify rows with >3 EC numbers — these go to the bottom with -1
    mask_too_many = df["_ec_list"].apply(len) > 3
    df_normal     = df[~mask_too_many].copy()
    df_too_many   = df[mask_too_many].copy()

    # Query API for all unique (pdb_id, chain) pairs in normal rows
    # Cache at PDB entry level to avoid redundant calls
    pdb_cache: dict[str, dict[str, dict[str, str]]] = {}

    unique_pdb_ids = df_normal["pdb_id"].str.upper().unique().tolist()
    print(f"Querying RCSB for {len(unique_pdb_ids)} unique PDB entries...")

    for pdb_id in tqdm(unique_pdb_ids, desc="RCSB provenance queries", unit="entry"):
        pdb_cache[pdb_id] = _fetch_provenance_for_entry(pdb_id)
        time.sleep(0.05)

    prio_ec_numbers = []
    prio_ec_sources = []

    for _, row in df_normal.iterrows():
        pdb_id   = str(row["pdb_id"]).strip().upper()
        chain_id = str(row["chain"]).strip().upper()
        ec_list  = row["_ec_list"]

        ec_to_source = pdb_cache.get(pdb_id, {}).get(chain_id, {})
        prio_ec, prio_src = _select_priority_ec(ec_list, ec_to_source)
        prio_ec_numbers.append(prio_ec)
        prio_ec_sources.append(prio_src)

    df_normal["prio_ec_number"] = prio_ec_numbers
    df_normal["prio_ec_source"] = prio_ec_sources

    df_too_many["prio_ec_number"] = "-1"
    df_too_many["prio_ec_source"] = "-1"

    df_out = pd.concat([df_normal, df_too_many], ignore_index=True)

    df_out = df_out.drop(columns=["_ec_list"])
    ec_idx = df_out.columns.get_loc("ec_number")
    cols   = list(df_out.columns)
    cols.remove("prio_ec_number")
    cols.remove("prio_ec_source")
    cols.insert(ec_idx + 1, "prio_ec_number")
    cols.insert(ec_idx + 2, "prio_ec_source")
    df_out = df_out[cols]

    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone.")
    print(f"  Total rows        : {len(df_out)}")
    print(f"  Normal rows       : {len(df_normal)}")
    print(f"  Flagged (>3 EC)   : {len(df_too_many)}")
    print(f"  Output saved to   : {OUTPUT_CSV}")

    source_counts = df_normal["prio_ec_source"].value_counts()
    print("\nPriority source distribution (normal rows):")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()

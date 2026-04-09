# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

QUERY = """
query TestQuery($pdb_id: String!) {
  entry(entry_id: $pdb_id) {
    rcsb_id
    struct {
      title
    }
    assemblies {
      rcsb_id
      pdbx_struct_assembly {
        oligomeric_count
        oligomeric_details
      }
    }
  }
}
"""


def query_single(pdb_id: str) -> list[dict]:
    """
    Query the RCSB API for a single PDB ID.
    Returns a list of row dicts (one per assembly), or a single error row.
    """
    pdb_id = pdb_id.strip().upper()

    try:
        resp = requests.post(
            RCSB_GRAPHQL_URL,
            json={"query": QUERY, "variables": {"pdb_id": pdb_id}},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.ConnectionError:
        print(f"  [{pdb_id}] ERROR: Could not connect.")
        return [_error_row(pdb_id, "Connection error")]
    except requests.exceptions.Timeout:
        print(f"  [{pdb_id}] ERROR: Request timed out.")
        return [_error_row(pdb_id, "Timeout")]
    except Exception as exc:
        print(f"  [{pdb_id}] ERROR: {exc}")
        return [_error_row(pdb_id, str(exc))]

    entry = (data.get("data") or {}).get("entry")

    if entry is None:
        errors = data.get("errors")
        msg = json.dumps(errors) if errors else "Entry not found"
        print(f"  [{pdb_id}] Not found: {msg}")
        return [_error_row(pdb_id, msg)]

    title = (entry.get("struct") or {}).get("title", "")
    assemblies = entry.get("assemblies") or []

    if not assemblies:
        # Entry exists but has no assembly data
        print(f"  [{pdb_id}] Found (no assemblies listed)")
        return [{
            "pdb_id":              pdb_id,
            "title":               title,
            "assembly_id":         "",
            "oligomeric_details":  "",
            "oligomeric_count":    "",
            "error":               "",
        }]

    rows = []
    for asm in assemblies:
        asm_data = asm.get("pdbx_struct_assembly") or {}
        rows.append({
            "pdb_id":              pdb_id,
            "title":               title,
            "assembly_id":         asm.get("rcsb_id", ""),
            "oligomeric_details":  asm_data.get("oligomeric_details", ""),
            "oligomeric_count":    asm_data.get("oligomeric_count", ""),
            "error":               "",
        })

    print(f"  [{pdb_id}] OK — {len(rows)} assembly row(s)")
    return rows


def _error_row(pdb_id: str, message: str) -> dict:
    return {
        "pdb_id":             pdb_id,
        "title":              "",
        "assembly_id":        "",
        "oligomeric_details": "",
        "oligomeric_count":   "",
        "error":              message,
    }


def batch_query(pdb_ids: list[str], output_csv: str) -> None:
    """
    Query all PDB IDs and write results to a CSV file.
    """
    fieldnames = [
        "pdb_id",
        "title",
        "assembly_id",
        "oligomeric_details",
        "oligomeric_count",
        "error",
    ]

    all_rows = []
    print(f"Querying {len(pdb_ids)} PDB ID(s)...\n")

    for pdb_id in pdb_ids:
        rows = query_single(pdb_id)
        all_rows.extend(rows)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} row(s) written to: {output_csv}")


# ---------------------------------------------------------------------------
# Edit PDB_IDS and OUTPUT_CSV, then run:  python batch_rcsb_query.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    PDB_IDS = [
        "13PK", 
        "1AH7", 
        "1AKM",
        "1AL6", 
        "1ALK", 
        "1B3R", 
        "1BG0", 
        "1BML", 
        "1BOL", 
        "1BRM", 
        "1BWD", 
        "1BWP", 
        "1D4A", 
        "1D4C", 
        "1DII", 
        "1DNK", 
        "1EF8", 
        "1GET", 
        "1GPA", 
        "1GRC", 
        "1HFS", 
        "1HXQ", 
        "1LXA", 
        "1MAS", 
        "1MBB", 
        "1MHY", 
        "1MPY", 
        "1OFG", 
        "1PFK", 
        "1RA2", 
        "1SMN", 
        "1ULA", 
        "1UOK", 
        "1UOX", 
        "1VAO", 
        "1VNC", 
        "1XVA", 
        "1ZIO", 
        "2BBK", 
        "2JCW", 
        "2THI", 
        "3ECA"
    ]

    OUTPUT_CSV = DATA_DIR / "Amitai RCSB query"/ "rcsb_results.csv"

    batch_query(PDB_IDS, OUTPUT_CSV)

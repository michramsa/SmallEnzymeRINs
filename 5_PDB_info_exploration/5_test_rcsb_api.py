# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"


def test_api(pdb_id: str) -> None:
    """
    Fire a minimal query at the RCSB API and print exactly what comes back,
    so you can see whether the entry is found and what the raw structure
    of the response looks like.
    """
    pdb_id = pdb_id.upper()

    query = """
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

    print(f"Querying RCSB for: {pdb_id}")
    print(f"Endpoint: {RCSB_GRAPHQL_URL}\n")

    try:
        resp = requests.post(
            RCSB_GRAPHQL_URL,
            json={"query": query, "variables": {"pdb_id": pdb_id}},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        print(f"HTTP status code : {resp.status_code}")
        print(f"Response headers : {dict(resp.headers)}\n")

        # Print the raw response text before any parsing
        print("--- Raw response ---")
        print(resp.text)
        print("--------------------\n")

        # Now try to parse it
        data = resp.json()
        entry = (data.get("data") or {}).get("entry")

        if entry is None:
            print("RESULT: 'entry' is None — PDB ID not found or API returned no data.")
            print("Check the 'errors' key below if present:")
            print(json.dumps(data.get("errors"), indent=2))
        else:
            print(f"RESULT: Entry found!")
            print(f"  rcsb_id  : {entry.get('rcsb_id')}")
            print(f"  title    : {(entry.get('struct') or {}).get('title')}")
            assemblies = entry.get("assemblies") or []
            print(f"  assemblies ({len(assemblies)} total):")
            for asm in assemblies:
                details = (asm.get("pdbx_struct_assembly") or {}).get("oligomeric_details")
                count   = (asm.get("pdbx_struct_assembly") or {}).get("oligomeric_count")
                print(f"    {asm.get('rcsb_id')} | {details} | count: {count}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect. Check your internet connection.")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out.")
    except Exception as exc:
        print(f"ERROR: Unexpected exception: {exc}")


# ---------------------------------------------------------------------------
# Edit TEST_PDB_ID and run:  python test_rcsb_api.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    TEST_PDB_ID = "1AH7"   # <- swap in any PDB ID you want to test

    test_api(TEST_PDB_ID)

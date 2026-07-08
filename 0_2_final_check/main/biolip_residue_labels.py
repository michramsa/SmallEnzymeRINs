"""
0_6b_biolip_residue_labels_ver1.py
====================================
Parse ``biolip_matched_rows.csv`` (produced by 0_6_biolip_lookup_ver1.py)
into a flat per-residue label lookup CSV.

Output
------
biolip_residue_labels.csv
    One row per unique (pdb_id, chain, seq_num) triple that appears in at
    least one BioLiP binding-residue or catalytic-residue column.

    Columns
    -------
    pdb_id     : lower-case PDB ID (as stored in biolip_matched_rows.csv)
    chain      : receptor chain ID  (e.g. "A")
    seq_num    : integer residue number from PDB author numbering (col 08)
    label      : one of  "catalytic" | "binding" | "dual"

    Label hierarchy
    ---------------
    - A residue that appears in *both* catalytic and binding columns
      (across any BioLiP rows for that pdb/chain) → "dual"
    - A residue that appears only in catalytic columns → "catalytic"
    - A residue that appears only in binding columns  → "binding"
    - Residues absent from BioLiP are simply not present in this file;
      the downstream consumer should treat absence as "none".

Token format assumption
-----------------------
BioLiP col 08 tokens are ``[one-letter AA][integer]``, e.g. ``W543``.
The amino-acid letter prefix is stripped; only the integer is kept as
seq_num so it can be compared directly against ``auth_seq_id`` values
in CIF files and Arpeggio JSON node IDs.

Tokens that do not match the expected pattern are logged and skipped.

Inputs (edit in main())
-----------------------
MATCHED_CSV : Path to biolip_matched_rows.csv
OUT_DIR     : Directory to write biolip_residue_labels.csv
"""

import re
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imports import *

# ---------------------------------------------------------------------------
# Token parsing
# ---------------------------------------------------------------------------

# Matches tokens like  W543  E544  H42  Y5426  (optional trailing letter ignored)
_TOKEN_RE = re.compile(r'^[A-Za-z](\d+)[A-Za-z]?$')

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def _parse_token(token: str) -> int | None:
    """
    Extract the integer residue number from a BioLiP col-08 token.

    Returns the integer on success, or None if the token is malformed.
    Tokens with a trailing insertion-code letter (e.g. ``W543A``) are
    currently skipped (return None) because the CIF node-ID scheme used
    downstream does not yet handle insertion codes.
    """
    m = _TOKEN_RE.match(token.strip())
    if m is None:
        return None
    # Reject tokens with trailing letters (insertion codes)
    raw = token.strip()
    if len(raw) > len(m.group(0)):  # shouldn't happen given the RE, but guard
        return None
    # Check for trailing letter after the digits
    digits_end = 1 + len(m.group(1))   # 1 for the leading AA letter
    if digits_end < len(raw):
        log.warning("Skipping token with insertion code: %s", raw)
        return None
    return int(m.group(1))


def _parse_residue_field(field_value: str) -> list[int]:
    """
    Split a space-separated BioLiP residue field and return a list of
    integer seq_nums.  Returns an empty list for null / placeholder values.
    """
    if not isinstance(field_value, str):
        return []
    stripped = field_value.strip()
    if stripped in ("", "-", "nan", "NaN"):
        return []
    seq_nums = []
    for tok in stripped.split():
        num = _parse_token(tok)
        if num is not None:
            seq_nums.append(num)
        else:
            log.warning("Unrecognised token (skipped): %r", tok)
    return seq_nums


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_residue_labels(matched_csv: Path) -> pd.DataFrame:
    """
    Read ``biolip_matched_rows.csv`` and return a DataFrame with one row per
    unique (pdb_id, chain, seq_num), labelled "binding", "catalytic", or "dual".

    Parameters
    ----------
    matched_csv : Path
        Path to biolip_matched_rows.csv produced by 0_6_biolip_lookup_ver1.py.

    Returns
    -------
    pd.DataFrame with columns: pdb_id, chain, seq_num, label
    """
    log.info("Loading matched rows from: %s", matched_csv)
    df = pd.read_csv(matched_csv, dtype=str)
    log.info("  Rows loaded: %d", len(df))

    # Sets of (pdb_id, chain, seq_num) for each annotation type
    binding_residues:   set[tuple[str, str, int]] = set()
    catalytic_residues: set[tuple[str, str, int]] = set()
    skipped_rows = 0

    for _, row in df.iterrows():
        pdb_id = str(row.get("pdb_id", "")).strip().lower()
        chain  = str(row.get("receptor_chain", "")).strip()

        if not pdb_id or not chain:
            skipped_rows += 1
            continue

        # --- binding residues (col 08) ---
        for seq_num in _parse_residue_field(str(row.get("binding_residues_pdb", ""))):
            binding_residues.add((pdb_id, chain, seq_num))

        # --- catalytic residues (col 10) ---
        for seq_num in _parse_residue_field(str(row.get("catalytic_residues_pdb", ""))):
            catalytic_residues.add((pdb_id, chain, seq_num))

    if skipped_rows:
        log.warning("Skipped %d rows with missing pdb_id or chain.", skipped_rows)

    log.info("  Unique binding residues   : %d", len(binding_residues))
    log.info("  Unique catalytic residues : %d", len(catalytic_residues))
    log.info("  Dual (in both)            : %d", len(binding_residues & catalytic_residues))

    # --- assign labels ---
    all_keys = binding_residues | catalytic_residues
    records = []
    for key in sorted(all_keys):
        pdb_id, chain, seq_num = key
        in_binding   = key in binding_residues
        in_catalytic = key in catalytic_residues

        if in_binding and in_catalytic:
            label = "dual"
        elif in_catalytic:
            label = "catalytic"
        else:
            label = "binding"

        records.append({
            "pdb_id":  pdb_id,
            "chain":   chain,
            "seq_num": seq_num,
            "label":   label,
        })

    label_df = pd.DataFrame(records, columns=["pdb_id", "chain", "seq_num", "label"])

    log.info("  Total labeled residues    : %d", len(label_df))
    log.info("  Label breakdown:")
    for lbl, cnt in label_df["label"].value_counts().items():
        log.info("    %-10s : %d", lbl, cnt)

    return label_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # ── Edit these paths ──────────────────────────────────────────────────────
    MATCHED_CSV = DATA_DIR_3 / "test" / "biolip_lookup" / "biolip_matched_rows.csv"
    OUT_DIR     = DATA_DIR_3 / "test" / "biolip_lookup"
    # ─────────────────────────────────────────────────────────────────────────

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "biolip_residue_labels.csv"

    label_df = build_residue_labels(MATCHED_CSV)

    label_df.to_csv(out_path, index=False)
    log.info("Wrote %d rows to: %s", len(label_df), out_path)

    # ── Quick sanity-check print ──────────────────────────────────────────────
    print("\n=== biolip_residue_labels.csv — first 15 rows ===")
    print(label_df.head(15).to_string(index=False))
    print(f"\nTotal rows : {len(label_df):,}")
    print(f"Label counts:\n{label_df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()

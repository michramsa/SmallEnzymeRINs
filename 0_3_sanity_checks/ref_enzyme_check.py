# 0_4_ref_enzyme_check_ver1.py
# ============================================================
# Query the RCSB PDB GraphQL API to determine which proteins
# in the reference set have EC number annotations (i.e., are
# enzymes).
#
# PDB IDs are sourced by scanning a directory for files named
# PDBID_clean.json (e.g. "1A2B_clean.json" -> PDB ID "1A2B"),
# consistent with the Arpeggio pipeline output convention used
# throughout this project.
#
# Outputs (written to output_dir)
# --------------------------------
#   ref_ec_lookup.csv        – one row per PDB ID:
#                              pdb_id, ec_number, is_enzyme, api_error
#   ref_enzyme_summary.txt   – plain-text counts / percentages
#   ref_ec_distribution.png  – bar chart of EC main classes
#                              (enzymes only, styled to match project)
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *

from figure_style_ver2 import (
    apply_style, save_fig,
    COLOR_CYCLE, FIG_SIZE, FONT, LAYOUT, COLORS,
)

# ---------------------------------------------------------------------------
# EC class metadata (mirrors 0_4_enzyme_distribution_ver2.py)
# ---------------------------------------------------------------------------

EC_CLASSES = {
    1: "EC 1\nOxidoreductases",
    2: "EC 2\nTransferases",
    3: "EC 3\nHydrolases",
    4: "EC 4\nLyases",
    5: "EC 5\nIsomerases",
    6: "EC 6\nLigases",
    7: "EC 7\nTranslocases",
}

EC_COLORS = {
    1: COLOR_CYCLE[0],
    2: COLOR_CYCLE[1],
    3: COLOR_CYCLE[2],
    4: COLOR_CYCLE[3],
    5: COLOR_CYCLE[4],
    6: COLOR_CYCLE[5],
    7: COLOR_CYCLE[6],
}

# ---------------------------------------------------------------------------
# PDB ID extraction from PDBID_clean.json filenames
# ---------------------------------------------------------------------------

def collect_pdb_ids_from_json_dir(json_dir: Path) -> list[str]:
    """
    Scan json_dir for files matching the pattern PDBID_clean.json and
    return a sorted list of unique, uppercase PDB IDs.

    The PDB ID is taken as everything before the first underscore in the
    filename stem — e.g. "1A2B_clean.json" yields "1A2B".

    Parameters
    ----------
    json_dir : Path
        Directory containing Arpeggio output files named PDBID_clean.json.

    Returns
    -------
    list of str  Sorted unique uppercase PDB IDs.
    """
    json_dir = Path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    pdb_ids: set[str] = set()
    skipped: list[str] = []

    for json_path in json_dir.glob("*_clean.json"):
        stem = json_path.stem                   # e.g. "1A2B_clean"
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1] == "clean":
            pdb_id = parts[0].upper()
            pdb_ids.add(pdb_id)
        else:
            skipped.append(json_path.name)

    if skipped:
        print(f"WARNING: {len(skipped)} file(s) did not match the "
              f"PDBID_clean.json pattern and were skipped:")
        for name in sorted(skipped):
            print(f"  {name}")

    sorted_ids = sorted(pdb_ids)
    print(f"Found {len(sorted_ids)} unique PDB IDs in: {json_dir}")
    return sorted_ids


# ---------------------------------------------------------------------------
# RCSB GraphQL query
# ---------------------------------------------------------------------------

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

# Query template: given a list of entry IDs, return the EC numbers of all
# polymer entities. rcsb_enzyme_class_combined is populated from PDB
# annotations and is independent of BioLiP.
_QUERY_TEMPLATE = """
{{
  entries(entry_ids: [{id_list}]) {{
    rcsb_id
    polymer_entities {{
      rcsb_polymer_entity {{
        rcsb_enzyme_class_combined {{
          ec
        }}
      }}
    }}
  }}
}}
"""


def _build_query(pdb_ids: list[str]) -> str:
    """Format a batch of PDB IDs into a GraphQL query string."""
    id_list = ", ".join(f'"{pid.upper()}"' for pid in pdb_ids)
    return _QUERY_TEMPLATE.format(id_list=id_list)


def query_ec_batch(
    pdb_ids: list[str],
    retries: int = 3,
    backoff: float = 2.0,
) -> dict[str, list[str] | None]:
    """
    Query RCSB GraphQL for EC numbers for a batch of PDB IDs.

    Returns
    -------
    dict mapping pdb_id (uppercase) -> list of EC number strings found
    across all polymer entities.  An empty list means the entry was
    returned by the API but had no EC annotation.  A value of None
    indicates an API / network error for the whole batch.
    """
    query = _build_query(pdb_ids)
    results: dict[str, list[str]] = {}

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": query},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                print(f"  GraphQL errors (partial): {data['errors']}")

            entries = data.get("data", {}).get("entries") or []
            for entry in entries:
                pid = entry["rcsb_id"].upper()
                ec_numbers: list[str] = []
                for entity in (entry.get("polymer_entities") or []):
                    ec_combined = (
                        entity
                        .get("rcsb_polymer_entity", {})
                        .get("rcsb_enzyme_class_combined") or []
                    )
                    for ec_entry in ec_combined:
                        ec_val = ec_entry.get("ec")
                        if ec_val:
                            ec_numbers.append(ec_val)
                results[pid] = ec_numbers

            return results

        except Exception as exc:
            print(f"  Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(backoff ** attempt)

    # All retries exhausted — mark every ID in this batch as errored
    return {pid.upper(): None for pid in pdb_ids}


def fetch_ec_for_ids(
    pdb_ids: list[str],
    batch_size: int = 50,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch EC numbers for all PDB IDs, batching requests to the RCSB API.

    Parameters
    ----------
    pdb_ids       : list of PDB ID strings (any case)
    batch_size    : number of IDs per GraphQL request
    sleep_between : seconds to wait between batch requests

    Returns
    -------
    pd.DataFrame with columns:
        pdb_id    – uppercase PDB ID
        ec_number – comma-separated EC numbers (empty string if none)
        is_enzyme – True if at least one EC number was found
        api_error – True if the API call failed for this entry
    """
    unique_ids = list({pid.upper() for pid in pdb_ids})
    n_total    = len(unique_ids)
    print(f"\nQuerying RCSB PDB API for {n_total} unique PDB IDs "
          f"(batch size = {batch_size})...")

    all_results: dict[str, list[str] | None] = {}

    batches = [unique_ids[i:i + batch_size]
               for i in range(0, n_total, batch_size)]

    for idx, batch in enumerate(batches, start=1):
        print(f"  Batch {idx}/{len(batches)}  ({len(batch)} IDs)...")
        batch_results = query_ec_batch(batch)
        all_results.update(batch_results)
        if idx < len(batches):
            time.sleep(sleep_between)

    # Build output DataFrame
    rows = []
    for pid in unique_ids:
        ec_list = all_results.get(pid)
        if ec_list is None:
            # API / network error — could not determine enzyme status
            rows.append({
                "pdb_id":    pid,
                "ec_number": "",
                "is_enzyme": False,
                "api_error": True,
            })
        else:
            # Deduplicate EC numbers while preserving order
            seen: set[str] = set()
            unique_ec: list[str] = []
            for ec in ec_list:
                if ec not in seen:
                    seen.add(ec)
                    unique_ec.append(ec)
            rows.append({
                "pdb_id":    pid,
                "ec_number": ",".join(unique_ec),
                "is_enzyme": bool(unique_ec),
                "api_error": False,
            })

    return pd.DataFrame(rows).sort_values("pdb_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_and_save_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print enzyme / non-enzyme counts and save to a text file."""
    n_total  = len(df)
    n_enzyme = df["is_enzyme"].sum()
    n_non    = (~df["is_enzyme"] & ~df["api_error"]).sum()
    n_error  = df["api_error"].sum()

    lines = [
        "Reference Set — Enzyme Content Summary",
        "=" * 42,
        f"Total PDB IDs queried : {n_total}",
        f"Enzymes (have EC)     : {n_enzyme}  ({100 * n_enzyme / n_total:.1f}%)",
        f"Non-enzymes (no EC)   : {n_non}  ({100 * n_non / n_total:.1f}%)",
        f"API errors            : {n_error}  ({100 * n_error / n_total:.1f}%)",
    ]

    if n_error > 0:
        lines.append("\nPDB IDs with API errors (enzyme status unknown):")
        for pid in sorted(df.loc[df["api_error"], "pdb_id"]):
            lines.append(f"  {pid}")

    summary = "\n".join(lines)
    print("\n" + summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary + "\n")
    print(f"\nSummary saved to: {output_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_ec_main_classes(ec_string: str) -> list[int]:
    """
    Extract all valid EC main class integers (1–7) from a
    comma-separated EC string such as "1.2.3.4,3.5.6.7".
    Returns an empty list if the string is empty or unparseable.
    """
    classes = []
    for token in ec_string.split(","):
        token = token.strip()
        if not token:
            continue
        first = token.split(".")[0]
        try:
            cls = int(first)
            if 1 <= cls <= 7:
                classes.append(cls)
        except ValueError:
            pass
    return classes


def plot_ref_ec_distribution(
    df: pd.DataFrame,
    output_fig: Path,
    title: str = "Reference Set — Distribution of EC Main Classes",
) -> None:
    """
    Bar chart of EC main-class counts for the reference set enzymes.
    Styled to match 0_4_enzyme_distribution_ver2.py.

    Only rows where is_enzyme=True and api_error=False are counted.
    Each enzyme can contribute to multiple EC classes if it carries
    multiple annotations; the "n = " annotation reflects the number
    of unique enzyme proteins, not the bar sum.
    """
    apply_style()

    enzyme_df = df[df["is_enzyme"] & ~df["api_error"]].copy()
    n_enzymes = len(enzyme_df)

    # Expand multi-EC entries so each class gets counted separately
    expanded_classes: list[int] = []
    for ec_str in enzyme_df["ec_number"]:
        expanded_classes.extend(_get_ec_main_classes(ec_str))

    counts = pd.Series(0, index=range(1, 8))
    for cls in expanded_classes:
        counts[cls] += 1

    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    bars = ax.bar(
        counts.index,
        counts.values,
        color=[EC_COLORS[i] for i in counts.index],
        edgecolor=COLORS["white"],
        linewidth=LAYOUT["spine_linewidth"],
        width=0.65,
        zorder=3,
    )

    # Value labels on top of each bar
    max_count = max(counts.values) if max(counts.values) > 0 else 1
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.015,
                str(val),
                ha="center", va="bottom",
                fontsize=FONT["size_tick"],
                fontweight="bold",
                color=COLORS["dark_gray"],
            )

    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(
        [EC_CLASSES[i] for i in range(1, 8)],
        fontsize=FONT["size_tick"],
    )
    ax.set_ylabel("Number of Entries", fontsize=FONT["size_label"])
    ax.set_xlabel("Enzyme Commission (EC) Class", fontsize=FONT["size_label"])

    if title:
        ax.set_title(
            title,
            fontsize=FONT["size_title"],
            fontweight=FONT["weight_title"],
        )

    # "n = " annotation — enzyme protein count, not bar sum
    ax.text(
        0.99, 0.97,
        f"n = {n_enzymes:,} enzyme proteins",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT["size_legend"],
        color=COLORS["mid_gray"],
        style="italic",
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"\nFigure saved to: {output_fig}")
    print("\nEC Class Counts:")
    for ec, count in counts.items():
        label = EC_CLASSES[ec].replace("\n", " - ")
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    # ── Input ───────────────────────────────────────────────────────────────
    # Directory containing PDBID_clean.json files (Arpeggio output).
    # PDB IDs are extracted from filenames; file contents are not read.
    json_dir   = DATA_DIR_3 / "test" / "JSONs",

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir = DATA_DIR_3 / "test" / "test_enzyme_check_no_mc",

    # ── API settings ────────────────────────────────────────────────────────
    # Number of PDB IDs per GraphQL request. 50 is a safe default.
    batch_size    = 50,

    # Seconds to sleep between batches (polite to the RCSB API).
    sleep_between = 0.5,
):
    json_dir   = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect PDB IDs from PDBID_clean.json filenames ──────────────────
    pdb_ids = collect_pdb_ids_from_json_dir(json_dir)

    if not pdb_ids:
        print("No PDB IDs found — check that json_dir contains *_clean.json files.")
        return

    # ── 2. Query RCSB PDB API ────────────────────────────────────────────────
    ec_df = fetch_ec_for_ids(
        pdb_ids,
        batch_size=batch_size,
        sleep_between=sleep_between,
    )

    # ── 3. Save lookup table ─────────────────────────────────────────────────
    lookup_csv = output_dir / "test_ec_lookup.csv"
    ec_df.to_csv(lookup_csv, index=False)
    print(f"\nEC lookup table saved to: {lookup_csv}")

    # ── 4. Print and save summary ────────────────────────────────────────────
    summary_path = output_dir / "test_enzyme_summary.txt"
    print_and_save_summary(ec_df, summary_path)

    # ── 5. Plot EC class distribution (enzymes only) ─────────────────────────
    output_fig = output_dir / "test_ec_distribution.png"
    plot_ref_ec_distribution(
        df=ec_df,
        output_fig=output_fig,
        title="Reference Set — Distribution of EC Main Classes",
    )


if __name__ == "__main__":
    main(batch_size    = 50,sleep_between = 0.5)

# for subdirectories to find the imports.py file
import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from imports import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings

from figure_style_ver2 import apply_style, save_fig, COLOR_CYCLE, FIG_SIZE, FONT, LAYOUT, DPI, COLORS

# ---------------------------------------------------------------------------
# Standard amino acids — used to filter HETATM / non-protein residues
# when counting chain length from structure files.
# ---------------------------------------------------------------------------

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    # Selenocysteine / pyrrolysine — common in PDB/CIF files
    "SEC", "PYL",
}

# ---------------------------------------------------------------------------
# EC class labels and colors
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
    1: COLOR_CYCLE[0],   # blue
    2: COLOR_CYCLE[1],   # orange
    3: COLOR_CYCLE[2],   # green
    4: COLOR_CYCLE[3],   # purple
    5: COLOR_CYCLE[4],   # teal
    6: COLOR_CYCLE[5],   # red
    7: COLOR_CYCLE[6],   # yellow
}


# ===========================================================================
# EC DISTRIBUTION HELPERS
# ===========================================================================

def _get_ec_counts(df: pd.DataFrame, ec_col: str) -> pd.Series:
    """
    Extract the first digit of each EC number and return a count Series
    indexed 1–7, with 0 for any missing classes.

    Entries whose EC number does not parse to a valid integer in 1–7
    (e.g. partial EC numbers like "3.-.-.-" or missing values) are
    dropped from the counts but reported to stdout.

    Parameters
    ----------
    df     : pd.DataFrame  Must contain ec_col.
    ec_col : str           Column holding full EC numbers (e.g. "3.5.4.13").

    Returns
    -------
    pd.Series  Integer counts indexed 1–7.
    """
    ec_class = (
        df[ec_col]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
    )
    ec_class = pd.to_numeric(ec_class, errors="coerce")
    valid_mask = ec_class.notna() & ec_class.between(1, 7)

    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        cols = ["pdb_id", ec_col] if "pdb_id" in df.columns else [ec_col]
        dropped_df = df.loc[~valid_mask, cols]
        print(f"NOTE: {n_dropped} entry/entries have unparseable EC numbers "
              f"and are excluded from bar counts:")
        for _, row in dropped_df.iterrows():
            print(f"  {row.to_dict()}")

    ec_class = ec_class[valid_mask].astype(int)
    counts = ec_class.value_counts().sort_index()
    all_classes = pd.Series(0, index=range(1, 8))
    return (all_classes + counts).fillna(0).astype(int)


def plot_ec_distribution(
    df: pd.DataFrame,
    ec_col: str,
    output_fig: Path,
    title: str = "Distribution of Enzyme Commission (EC) Main Classes",
    n_total: int = None,
) -> None:
    """
    Plot a bar chart of EC main-class distribution and save to disk.
    Styling follows figure_style_ver2.py conventions.

    Parameters
    ----------
    df         : pd.DataFrame  Must contain ec_col.
    ec_col     : str           Column holding EC numbers (e.g. "prio_ec_number").
    output_fig : Path          Full output path including filename.
    title      : str           Figure title. Pass None to suppress.
    n_total    : int, optional
        The true number of proteins to display in the "n = " annotation.
        Defaults to len(df) when not provided. Pass this explicitly when
        the DataFrame has already been deduplicated upstream.
    """
    apply_style()

    counts = _get_ec_counts(df, ec_col)

    if n_total is None:
        n_total = len(df)

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

    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.015,
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
        ax.set_title(title, fontsize=FONT["size_title"], fontweight=FONT["weight_title"])

    ax.text(
        0.99, 0.97,
        f"n = {n_total:,} proteins",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT["size_legend"],
        color=COLORS["mid_gray"],
        style="italic",
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"Figure saved to: {output_fig}")
    print(f"\nEC Class Counts ({ec_col}):")
    for ec, count in counts.items():
        label = EC_CLASSES[ec].replace("\n", " - ")
        print(f"  {label}: {count}")


# ===========================================================================
# SIZE DISTRIBUTION HELPERS
# ===========================================================================

def _count_residues_from_cif(cif_path: Path) -> tuple[int | None, int | None, str | None]:
    """
    Count the number of standard amino acid residues and protein chains in a
    CIF file using gemmi (fast C++ parser) if available, otherwise BioPython.

    Only ATOM records are considered; HETATM residues (ligands, waters) are
    excluded. n_chains counts only chains that contain at least one standard
    amino acid residue (i.e. protein chains only).

    Returns
    -------
    (n_residues, n_chains, error_msg)
    n_residues and n_chains are None on failure; error_msg is None on success.
    """
    # --- try gemmi first (much faster for large CIF files) ---
    try:
        import gemmi  # type: ignore
        st = gemmi.read_structure(str(cif_path))
        count = 0
        protein_chains = 0
        for model in st:
            for chain in model:
                chain_res = sum(
                    1 for res in chain
                    if res.entity_type == gemmi.EntityType.Polymer
                    and res.name in STANDARD_AA
                )
                if chain_res > 0:
                    count += chain_res
                    protein_chains += 1
        if count > 0:
            return count, protein_chains, None
        return None, None, "gemmi parsed but found 0 standard residues"
    except ImportError:
        pass  # gemmi not installed — fall back to BioPython
    except Exception as exc:
        return None, None, f"gemmi: {exc}"

    # --- BioPython fallback ---
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        count = 0
        protein_chains = 0
        # Use first model only for chain counting (avoids double-counting NMR models)
        model = next(iter(structure))
        for chain in model:
            chain_res = sum(
                1 for res in chain
                if res.get_resname().strip() in STANDARD_AA
                and res.get_id()[0] == " "
            )
            if chain_res > 0:
                count += chain_res
                protein_chains += 1
        if count > 0:
            return count, protein_chains, None
        return None, None, "BioPython parsed but found 0 standard residues"
    except Exception as exc:
        return None, None, f"BioPython CIF: {exc}"


def _count_residues_from_pdb(pdb_path: Path) -> tuple[int | None, int | None, str | None]:
    """
    Count standard amino acid residues and protein chains in a PDB file
    using BioPython.

    Only ATOM records are considered (residue id[0] == " "). n_chains counts
    only chains that contain at least one standard amino acid residue.
    Deduplication across NMR models is handled by using the first model only
    for chain counting, consistent with the CIF parser.

    Returns
    -------
    (n_residues, n_chains, error_msg)
    n_residues and n_chains are None on failure; error_msg is None on success.
    """
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        # Use first model only to avoid double-counting NMR ensembles
        model = next(iter(structure))
        seen = set()
        protein_chains = 0
        for chain in model:
            chain_residues = [
                res for res in chain
                if res.get_resname().strip() in STANDARD_AA
                and res.get_id()[0] == " "
            ]
            if chain_residues:
                protein_chains += 1
                for res in chain_residues:
                    seen.add((chain.id, res.get_id()))
        if seen:
            return len(seen), protein_chains, None
        return None, None, "BioPython parsed but found 0 standard residues"
    except Exception as exc:
        return None, None, f"BioPython PDB: {exc}"


def count_residues_from_structure_dir(
    structure_dir: Path,
    glob_pattern: str = "*.cif",
    csv_path: Path = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk *structure_dir* and count the number of standard amino acid residues
    in every structure file matching *glob_pattern*.

    Works for CIF files (``*.cif``) and PDB files (``*.pdb``).  Pass the
    directory of Arpeggio-cleaned CIF/PDB files to get cleaned counts; or
    pass a directory of raw CIF/PDB files for the full-length counts.

    Parameters
    ----------
    structure_dir : Path
        Directory containing structure files. Non-recursive (top-level only).
    glob_pattern : str
        Glob pattern to match files, e.g. ``"*.cif"`` or ``"*.pdb"``.
        Default is ``"*.cif"``.
    csv_path : Path, optional
        If provided, the results DataFrame is saved to this path as a CSV.
        The directory is created if it does not exist. Both successful counts
        and errors are recorded — failed files have NaN in ``n_residues`` and
        a description in the ``error`` column.
    verbose : bool
        Print progress to stdout. Default True.

    Returns
    -------
    pd.DataFrame with columns:
        ``pdb_id``       — uppercase PDB ID extracted from the filename stem
                           (first 4 characters); full stem used as fallback
        ``stem``         — full filename without extension
        ``n_residues``   — residue count (NaN for files that failed to parse)
        ``n_chains``     — number of protein chains (NaN for files that failed to parse)
        ``error``        — error description (NaN for successful parses)
        ``source_file``  — full path string (for auditing)
    """
    structure_dir = Path(structure_dir)
    files = sorted(structure_dir.glob(glob_pattern))

    if not files:
        print(f"WARNING: no files matching '{glob_pattern}' in {structure_dir}")
        return pd.DataFrame(columns=["pdb_id", "stem", "n_residues", "error", "source_file"])

    if verbose:
        print(f"Counting residues in {len(files)} {glob_pattern} file(s) "
              f"from {structure_dir} ...")

    is_cif = glob_pattern.lower().endswith(".cif") or ".cif" in glob_pattern.lower()

    rows = []
    failed = 0
    for i, fpath in enumerate(files, 1):
        if is_cif:
            n, n_chains, err = _count_residues_from_cif(fpath)
        else:
            n, n_chains, err = _count_residues_from_pdb(fpath)

        if n is None:
            failed += 1

        # Extract PDB ID: first 4 chars of stem, uppercased.
        # Falls back to the full stem if it is shorter than 4 characters.
        stem = fpath.stem
        pdb_id = stem[:4].upper() if len(stem) >= 4 else stem.upper()

        rows.append({
            "pdb_id":      pdb_id,
            "stem":        stem,
            "n_residues":  n,
            "n_chains":    n_chains,
            "error":       err,
            "source_file": str(fpath),
        })

        if verbose and i % 100 == 0:
            print(f"  ... processed {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df["n_residues"] = pd.to_numeric(df["n_residues"], errors="coerce")
    df["n_chains"]   = pd.to_numeric(df["n_chains"],   errors="coerce")

    if verbose:
        valid = df["n_residues"].notna().sum()
        print(f"  Done — {valid}/{len(files)} parsed successfully "
              f"({failed} failed).")
        if valid > 0:
            print(f"  Residue count range: "
                  f"{df['n_residues'].min():.0f} – {df['n_residues'].max():.0f}")
        if failed > 0:
            print(f"  Failed files:")
            for _, row in df[df["n_residues"].isna()].iterrows():
                print(f"    {row['stem']}: {row['error']}")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Results saved to: {csv_path}")

    return df


def plot_size_distribution(
    residue_counts: pd.Series | pd.DataFrame,
    output_fig: Path,
    label: str = "Proteins",
    title: str = "Protein Size Distribution",
    color: str = None,
    bin_width: int = 50,
    x_max: int = None,
) -> None:
    """
    Plot a histogram of protein size (number of residues) and save to disk.

    Parameters
    ----------
    residue_counts : pd.Series or pd.DataFrame
        Either a Series of residue counts, or a DataFrame with an
        ``n_residues`` column (as returned by count_residues_from_structure_dir).
        NaN values are automatically dropped.
    output_fig : Path
        Full output path including filename.
    label : str
        Dataset label used in the n= annotation. Default "Proteins".
    title : str
        Figure title. Pass None to suppress.
    color : str, optional
        Bar fill colour (hex or named). Defaults to COLOR_CYCLE[0] (blue).
    bin_width : int
        Width of each histogram bin in residues. Default 50.
    x_max : int, optional
        Upper limit of the x-axis. Automatically determined from data if None.
    """
    apply_style()

    if isinstance(residue_counts, pd.DataFrame):
        if "n_residues" not in residue_counts.columns:
            raise ValueError("DataFrame must contain an 'n_residues' column.")
        counts = residue_counts["n_residues"].dropna()
    else:
        counts = residue_counts.dropna()

    counts = counts.astype(float)

    if counts.empty:
        print(f"WARNING: no valid residue counts — skipping {output_fig}")
        return

    if color is None:
        color = COLOR_CYCLE[0]

    # Determine bin edges
    data_max = counts.max()
    if x_max is None:
        # Round up to next clean bin boundary
        x_max = int(((data_max // bin_width) + 2) * bin_width)
    bins = range(0, x_max + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=FIG_SIZE["full_width"])

    ax.hist(
        counts,
        bins=list(bins),
        color=color,
        edgecolor=COLORS["white"],
        linewidth=LAYOUT["spine_linewidth"],
        zorder=3,
    )

    # Summary statistics lines
    median_val = counts.median()
    mean_val   = counts.mean()

    ax.axvline(
        median_val,
        color=COLORS["dark_gray"],
        linewidth=LAYOUT["line_width"],
        linestyle="--",
        zorder=4,
        label=f"Median = {median_val:.0f}",
    )
    ax.axvline(
        mean_val,
        color=COLORS["mid_gray"],
        linewidth=LAYOUT["line_width"],
        linestyle=":",
        zorder=4,
        label=f"Mean = {mean_val:.0f}",
    )

    ax.legend(fontsize=FONT["size_legend"], loc="upper right")

    ax.set_xlabel("Number of Residues", fontsize=FONT["size_label"])
    ax.set_ylabel("Number of Proteins", fontsize=FONT["size_label"])

    if title:
        ax.set_title(title, fontsize=FONT["size_title"], fontweight=FONT["weight_title"])

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    n_total = len(counts)
    ax.text(
        0.99, 0.97,
        f"n = {n_total:,} {label}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=FONT["size_legend"],
        color=COLORS["mid_gray"],
        style="italic",
    )

    fig.tight_layout(pad=LAYOUT["tight_pad"])
    output_fig = Path(output_fig)
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_fig)
    plt.close(fig)

    print(f"Figure saved to: {output_fig}")
    print(f"\nSize Distribution Summary ({label}):")
    print(f"  n        = {n_total:,}")
    print(f"  min      = {counts.min():.0f} residues")
    print(f"  median   = {median_val:.0f} residues")
    print(f"  mean     = {mean_val:.1f} residues")
    print(f"  max      = {counts.max():.0f} residues")
    print(f"  std      = {counts.std():.1f} residues")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configurable paths  (update these as your directory structure changes)
    # ------------------------------------------------------------------

    FIGURES_DIR = FIGS_DIR / "enzyme_dist_no_mc"

    # --- Size distribution inputs ---
    # Point at whichever directory of structure files you want to count.
    # Arpeggio-cleaned CIFs are preferred because non-protein atoms have
    # already been stripped; raw CIF/PDB directories also work.

    # Enzyme test set — Arpeggio-cleaned CIFs
    ENZYME_CIF_DIR  = DATA_DIR_3 / "test" / "CIFs_cleaned"

    # Reference (PISCES) set — adjust to match your layout
    REF_CIF_DIR     = DATA_DIR_3 / "ref" / "CIFs_cleaned" / "structures"

    # Glob pattern: "*.cif" for CIF files, "*.pdb" for PDB files
    STRUCT_GLOB     = "*.cif"

    # ------------------------------------------------------------------
    # Size distribution — enzyme test set
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("SIZE DISTRIBUTION — enzyme test set")
    print("=" * 60)

    df_enzyme_sizes = count_residues_from_structure_dir(
        structure_dir=ENZYME_CIF_DIR,
        glob_pattern=STRUCT_GLOB,
        csv_path=FIGURES_DIR / "no_mc_size_dist_enzyme_test_set.csv",
    )

    plot_size_distribution(
        residue_counts=df_enzyme_sizes,
        output_fig=FIGURES_DIR / "no_mc_size_dist_enzyme_test_set.png",
        label="enzymes",
        title="Protein Size Distribution — Enzyme Test Set",
        color=COLOR_CYCLE[0],
        bin_width=10,
    )

    # ------------------------------------------------------------------
    # Size distribution — reference set  (uncomment when REF_CIF_DIR exists)
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("SIZE DISTRIBUTION — reference set")
    print("=" * 60)

    df_ref_sizes = count_residues_from_structure_dir(
        structure_dir=REF_CIF_DIR,
        glob_pattern=STRUCT_GLOB,
        csv_path=FIGURES_DIR / "no_mc_size_dist_ref_set.csv",
    )

    plot_size_distribution(
        residue_counts=df_ref_sizes,
        output_fig=FIGURES_DIR / "no_mc_size_dist_ref_set.png",
        label="reference proteins",
        title="Protein Size Distribution — Reference Set",
        color=COLOR_CYCLE[2],
        bin_width=10,
    )

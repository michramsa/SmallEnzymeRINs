
import csv
from pathlib import Path
from collections import Counter
import gemmi

CIF_DIR = Path("/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/enzymes_CIFs")
OUTPUT_CSV = Path("/home/epi2melabs/AcrNetworks-clean-v2/3_data/enzymes/cif_inspection.csv")


def inspect_cif(cif_path):
    """
    Parse a CIF file and return a dict with:
      - experimental_method (str): e.g. 'X-RAY DIFFRACTION', 'SOLUTION NMR'
      - resolution (float or None)
      - altloc_atom_count (int): number of atoms with a non-empty alt-loc
      - altloc_residue_count (int): number of residues that contain alt-locs
      - altloc_labels (str): comma-separated set of alt-loc labels seen (e.g. "A,B")
      - total_atoms (int)
      - error (str or None)
    """
    info = {
        "file": cif_path.name,
        "experimental_method": None,
        "resolution": None,
        "altloc_atom_count": 0,
        "altloc_residue_count": 0,
        "altloc_labels": "",
        "total_atoms": 0,
        "has_altlocs": False,
        "error": None,
    }

    try:
        structure = gemmi.read_structure(str(cif_path))

        # --- experimental method ---
        # Read directly from the CIF doc. This is version-agnostic across gemmi
        # releases (the Metadata object's attributes change between versions).
        methods = []
        try:
            doc = gemmi.cif.read(str(cif_path))
            block = doc.sole_block()
            exptl = block.find_values("_exptl.method")
            if exptl:
                methods = [v.strip().strip("'\"") for v in exptl]
        except Exception:
            pass  # fall through to UNKNOWN

        info["experimental_method"] = " | ".join(methods) if methods else "UNKNOWN"

        # --- resolution ---
        if structure.resolution and structure.resolution > 0:
            info["resolution"] = round(structure.resolution, 2)

        # --- alt-loc scan ---
        altloc_atoms = 0
        altloc_residues = set()
        altloc_labels = set()
        total_atoms = 0

        # Only iterate over model 1 to avoid double-counting NMR ensembles
        if len(structure) > 0:
            model = structure[0]
            for chain in model:
                for residue in chain:
                    residue_has_altloc = False
                    for atom in residue:
                        total_atoms += 1
                        if atom.altloc and atom.altloc != "\0":
                            altloc_atoms += 1
                            altloc_labels.add(atom.altloc)
                            residue_has_altloc = True
                    if residue_has_altloc:
                        altloc_residues.add((chain.name, residue.seqid.num, residue.name))

        info["altloc_atom_count"] = altloc_atoms
        info["altloc_residue_count"] = len(altloc_residues)
        info["altloc_labels"] = ",".join(sorted(altloc_labels))
        info["total_atoms"] = total_atoms
        info["has_altlocs"] = altloc_atoms > 0

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"

    return info


def classify_method(method_str):
    """Collapse method strings into broad categories for the summary."""
    if not method_str or method_str == "UNKNOWN":
        return "UNKNOWN"
    m = method_str.upper()
    if "X-RAY" in m or "DIFFRACTION" in m:
        return "X-ray"
    if "NMR" in m:
        return "NMR"
    if "ELECTRON MICROSCOPY" in m or "CRYO" in m or "EM" in m:
        return "Cryo-EM"
    if "NEUTRON" in m:
        return "Neutron"
    return "Other"


def main():
    if not CIF_DIR.exists():
        print(f"Directory not found: {CIF_DIR}")
        return

    cif_files = sorted(CIF_DIR.rglob("*.cif"))
    print(f"Found {len(cif_files)} CIF files in {CIF_DIR}")

    if not cif_files:
        return

    results = []
    for i, cif_path in enumerate(cif_files, 1):
        print(f"  [{i}/{len(cif_files)}] {cif_path.name}", end="")
        result = inspect_cif(cif_path)
        result["relative_path"] = str(cif_path.relative_to(CIF_DIR))
        results.append(result)

        if result["error"]:
            print(f"  ERROR: {result['error']}")
        else:
            altloc_tag = f"alt-locs={result['altloc_atom_count']}" if result["has_altlocs"] else "no alt-locs"
            print(f"  [{classify_method(result['experimental_method'])}] {altloc_tag}")

    # --- write CSV ---
    fieldnames = [
        "relative_path",
        "file",
        "experimental_method",
        "method_category",
        "resolution",
        "total_atoms",
        "has_altlocs",
        "altloc_atom_count",
        "altloc_residue_count",
        "altloc_labels",
        "error",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r["method_category"] = classify_method(r["experimental_method"])
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nCSV written to: {OUTPUT_CSV}")

    # --- console summary ---
    total = len(results)
    errored = sum(1 for r in results if r["error"])
    with_altlocs = sum(1 for r in results if r["has_altlocs"])
    method_counts = Counter(classify_method(r["experimental_method"]) for r in results if not r["error"])

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total files:        {total}")
    print(f"  Parsed OK:          {total - errored}")
    print(f"  Errors:             {errored}")
    print(f"  Contain alt-locs:   {with_altlocs}  ({100 * with_altlocs / total:.1f}%)")
    print()
    print("  By experimental method:")
    for method, count in method_counts.most_common():
        print(f"    {method:<12} {count}")

    # Breakdown of alt-locs by method
    print()
    print("  Alt-loc prevalence by method:")
    method_altloc = {}
    for r in results:
        if r["error"]:
            continue
        cat = classify_method(r["experimental_method"])
        if cat not in method_altloc:
            method_altloc[cat] = [0, 0]  # [with_altlocs, total]
        method_altloc[cat][1] += 1
        if r["has_altlocs"]:
            method_altloc[cat][0] += 1
    for method, (n_alt, n_total) in sorted(method_altloc.items()):
        print(f"    {method:<12} {n_alt}/{n_total}  ({100 * n_alt / n_total:.0f}%)")


if __name__ == "__main__":
    main()

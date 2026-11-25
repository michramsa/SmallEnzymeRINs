"""Average network metrics across all graphs (one row per protein) in a CSV."""

import pandas as pd

DEFAULT_COLUMNS = [
    "nodes",
    "edges",
    "density",
    "diameter",
    "avg_clustering",
    "char_path_length",
]


def compute_averages(csv_path, columns=None):
    """Return a Series of column means (NaNs ignored) and the number of graphs."""
    columns = columns or DEFAULT_COLUMNS
    df = pd.read_csv(csv_path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in {csv_path}: {missing}")

    # Coerce to numeric so stray strings become NaN instead of crashing
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(skipna=True), numeric.count(), len(df)


def main(csv_path, columns=None, output_csv=None):
    means, counts, n_graphs = compute_averages(csv_path, columns)

    print(f"File: {csv_path}")
    print(f"Graphs in file: {n_graphs}\n")
    print(f"{'metric':<20}{'mean':>15}{'n_used':>10}")
    print("-" * 45)
    for name in means.index:
        print(f"{name:<20}{means[name]:>15.4f}{counts[name]:>10d}")

    if output_csv:
        out = pd.DataFrame({"metric": means.index, "mean": means.values,
                            "n_used": counts.values})
        out.to_csv(output_csv, index=False)
        print(f"\nSaved averages to {output_csv}")


if __name__ == "__main__":
    # ---- Edit these values ----
    csv_path = "...network_metrics.csv"
    columns = DEFAULT_COLUMNS
    output_csv = "...metric_averages.csv"
    # ---------------------------

    main(csv_path, columns, output_csv)

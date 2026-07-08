"""
inspect_jackknife_stepsizes.py
-------------------------------
Reports the np.linspace step size (and full candidate array) for each
feature axis used in jackknife_experiment.py's build_grid(), WITHOUT
loading data or running the jackknife loop.

Reuses build_grid() directly from jackknife_experiment.py so the values
shown here are guaranteed to match what main() would actually generate.

All bounds below mirror the defaults in jackknife_experiment.main().
Edit them here (or override via CLI-style edits in __main__) if you want
to preview a different set of bounds before committing to a full run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jackknife_experiment import build_grid


def report_stepsizes(
    # StdCloseness
    closeness_min=-0.281, closeness_max=1.764, closeness_n=10,
    # StdBetweenness
    betweenness_min=-0.465, betweenness_max=2.237, betweenness_n=10,
    # RCA_zscore
    rca_min=-1.015, rca_max=2.123, rca_n=10,
    # StdPageRank
    pagerank_min=-0.663, pagerank_max=1.447, pagerank_n=10,
    # RSA (shared across all metrics)
    rsa_min=0.00, rsa_max=0.90, rsa_n=10,
):
    metric_configs = [
        ("closeness",   closeness_min,   closeness_max,   closeness_n),
        ("betweenness", betweenness_min, betweenness_max, betweenness_n),
        ("rca",         rca_min,         rca_max,         rca_n),
        ("pagerank",    pagerank_min,    pagerank_max,    pagerank_n),
    ]

    print("=" * 70)
    print("Linspace step sizes per metric axis (centrality) + shared RSA axis")
    print("=" * 70)

    for metric_name, c_min, c_max, c_n in metric_configs:
        # build_grid also needs rsa bounds, but we only care about c_vals
        # here; rsa_vals will be identical across all four calls.
        _, c_vals, rsa_vals = build_grid(c_min, c_max, c_n, rsa_min, rsa_max, rsa_n)

        c_step = (c_max - c_min) / (c_n - 1)
        print(f"\n[{metric_name}]")
        print(f"  range        : [{c_min}, {c_max}]  (n={c_n})")
        print(f"  step size    : {c_step:.6f}")
        print(f"  candidates   : {c_vals.tolist()}")

    # RSA axis is the same regardless of which metric loop it came from
    rsa_step = (rsa_max - rsa_min) / (rsa_n - 1)
    print(f"\n[rsa] (shared across all metrics)")
    print(f"  range        : [{rsa_min}, {rsa_max}]  (n={rsa_n})")
    print(f"  step size    : {rsa_step:.6f}")
    print(f"  candidates   : {rsa_vals.tolist()}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Edit any bounds/n here to preview different candidate grids before
    # committing to a full jackknife_experiment.main() run.
    report_stepsizes(
        closeness_min=-0.281,   closeness_max=1.764,   closeness_n=10,
        betweenness_min=-0.465, betweenness_max=2.237, betweenness_n=10,
        rca_min=-1.015,         rca_max=2.123,          rca_n=10,
        pagerank_min=-0.663,    pagerank_max=1.447,     pagerank_n=10,
        rsa_min=0.00,           rsa_max=0.90,           rsa_n=10,
    )

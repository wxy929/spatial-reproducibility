"""One-factor analyses and five PCA/KMeans seeds used in the Supplement."""
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

from run_real_data import RESULTS, cases, fit_pair
from scores import prepare_pair, build_state, score_state


def parameter_grid(dataset):
    neighbors = {"dlpfc": [25, 50, 100, 150, 200, 250, 300],
                 "cancer": [50, 75, 100, 150, 200]}[dataset]
    return {
        "k_ot": neighbors, "k_neighbor": neighbors,
        "ot_pcs": [10, 30, 50], "spot_pcs": [10, 30, 50],
        "feature_pcs": [10, 30, 50], "neighbor_pcs": [10, 30, 50],
        "domain_pcs": [10, 30, 50], "domain_k": [5, 7, 9],
    }


def run(case):
    left, right, base = fit_pair(case)
    xy_a, xy_b = left.obsm["spatial"], right.obsm["spatial"]
    a, b = base["pca_a"], base["pca_b"]
    defaults = dict(k_neighbor=case["k"], spot_pcs=30, feature_pcs=10,
                    neighbor_pcs=50, domain_pcs=50, domain_k=7)
    rows = []
    for factor, values in parameter_grid(case["dataset"]).items():
        for value in values:
            settings = defaults.copy()
            state = base
            if factor == "k_ot":
                if value != case["k"]:
                    state = build_state(a, b, xy_a, xy_b, value)
            elif factor == "ot_pcs":
                if value != 50:
                    state = build_state(a, b, xy_a, xy_b, case["k"], ot_pcs=value)
            else:
                settings[factor] = value
            result = score_state(state, xy_a, xy_b, **settings)
            rows.append(dict(pair=case["name"], factor=factor, value=value, **result))
    seed_rows = []
    if case["dataset"] == "dlpfc":
        for seed in range(5):
            a, b = prepare_pair(left, right, seed=seed)
            state = build_state(a, b, xy_a, xy_b, case["k"])
            result = score_state(state, xy_a, xy_b, **defaults, seed=seed)
            seed_rows.append(dict(pair=case["name"], seed=seed, **result))
    return rows, seed_rows


if __name__ == "__main__":
    dataset = sys.argv[1].lower()
    rows, seed_rows = [], []
    with threadpool_limits(limits=2):
        for case in cases():
            if case["dataset"] == dataset and case["main"]:
                print(case["name"], flush=True)
                pair_rows, pair_seeds = run(case)
                rows.extend(pair_rows)
                seed_rows.extend(pair_seeds)
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"{dataset}_sensitivity.csv", index=False)
    if dataset == "dlpfc":
        pd.DataFrame(seed_rows).to_csv(RESULTS / "dlpfc_seeds.csv", index=False)

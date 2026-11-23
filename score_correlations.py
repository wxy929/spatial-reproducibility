"""Descriptive correlations among the 66 DLPFC pairwise score estimates."""
from itertools import combinations

import numpy as np
import pandas as pd

from run_real_data import RESULTS, cases


def correlations(table):
    metrics = ["Spot", "Feature", "Neighbor", "Domain"]
    groups = {"all": table, "within_donor": table[table.donor_a == table.donor_b],
              "cross_donor": table[table.donor_a != table.donor_b]}
    rows = []
    for name, group in groups.items():
        pearson = group[metrics].corr(method="pearson")
        spearman = group[metrics].corr(method="spearman")
        for a, b in combinations(metrics, 2):
            rows.append(dict(group=name, n_pairs=len(group), score_x=a, score_y=b,
                             pearson_r=pearson.loc[a, b], spearman_rho=spearman.loc[a, b]))
    return pd.DataFrame(rows)


def load_dlpfc_scores():
    donors = pd.DataFrame([dict(case=case["name"], donor_a=case["donor_a"], donor_b=case["donor_b"])
                           for case in cases() if case["dataset"] == "dlpfc"])
    path = RESULTS / "dlpfc_scores.csv"
    if not path.exists():
        raise ValueError("Run python run_real_data.py dlpfc first to calculate all 66 DLPFC pairs.")
    table = pd.read_csv(path)
    if table.case.duplicated().any() or set(table.case) != set(donors.case):
        raise ValueError("Expected exactly 66 unique DLPFC pairs; run python run_real_data.py dlpfc.")
    if not np.isfinite(table[["Spot", "Feature", "Neighbor", "Domain"]].to_numpy()).all():
        raise ValueError("DLPFC scores contain missing or non-finite values.")
    return table.merge(donors, on="case")


if __name__ == "__main__":
    correlations(load_dlpfc_scores()).to_csv(
        RESULTS / "dlpfc_score_correlations.csv", index=False)

"""Calculate the paper's real-tissue comparisons from the bundled inputs."""

from itertools import combinations
from pathlib import Path
import sys

import anndata as ad
import pandas as pd
from threadpoolctl import threadpool_limits

from scores import prepare_pair, build_state, score_state


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
DLPFC_SLICES = ["151507", "151508", "151509", "151510", "151669", "151670",
                "151671", "151672", "151673", "151674", "151675", "151676"]


def cases():
    records = []
    main_pairs = {(DLPFC_SLICES[start + i], DLPFC_SLICES[start + i + 1])
                  for start in (0, 4, 8) for i in range(3)}
    for a, b in combinations(DLPFC_SLICES, 2):
        records.append(dict(name=f"dlpfc_{a}_{b}", dataset="dlpfc", left=f"dlpfc_{a}",
                            right=f"dlpfc_{b}", k=200, domain_k=7, main=(a, b) in main_pairs,
                            donor_a=DLPFC_SLICES.index(a) // 4 + 1,
                            donor_b=DLPFC_SLICES.index(b) // 4 + 1,
                            slice_a=a, slice_b=b))
    for a, b in [("V2", "V1"), ("V3", "V1"), ("V4", "V1"),
                 ("V3", "V2"), ("V4", "V2"), ("V3", "V4")]:
        records.append(dict(name=f"skin_{a}_{b}", dataset="skin", left=f"skin_{a}",
                            right=f"skin_{b}", k=25, domain_k=7, main=True))
    for a, b in [("R1_D1", "R1_D2"), ("R1_D1", "R1_T"), ("R1_D2", "R1_T"),
                 ("R1_D2", "R2_D2"), ("R1_D2", "R2_T"),
                 ("R1_T", "R2_D2"), ("R1_T", "R2_T")]:
        records.append(dict(name=f"cancer_{a}_{b}", dataset="cancer", left=f"cancer_{a}",
                            right=f"cancer_{b}", k=100, domain_k=7, main=True))
    return records


def load_pair(case):
    left = ad.read_h5ad(DATA / f"{case['left']}.h5ad")
    right = ad.read_h5ad(DATA / f"{case['right']}.h5ad")
    # Dense float32 input preserves the normalization arithmetic.
    left.X = left.X.toarray()
    right.X = right.X.toarray()
    return left, right


def fit_pair(case):
    left, right = load_pair(case)
    pca_a, pca_b = prepare_pair(left, right, dense_order="C" if case["dataset"] == "skin" else None)
    xy_a, xy_b = left.obsm["spatial"], right.obsm["spatial"]
    state = build_state(pca_a, pca_b, xy_a, xy_b, k=case["k"])
    return left, right, state


def run(case):
    left, right, state = fit_pair(case)
    values = score_state(state, left.obsm["spatial"], right.obsm["spatial"],
                         k_neighbor=case["k"], domain_k=case["domain_k"])
    print(case["name"], values, flush=True)
    return dict(case=case["name"], **values)


def save_scores(rows, dataset):
    path = RESULTS / f"{dataset}_scores.csv"
    table = pd.DataFrame(rows)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    table.drop_duplicates("case", keep="last").sort_values("case").to_csv(path, index=False)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    selection = sys.argv[1] if len(sys.argv) > 1 else "main"
    selected = [case for case in cases()
                if selection in ("all", case["name"], case["dataset"])
                or (selection == "main" and case["main"])]
    if not selected:
        raise SystemExit(f"Unknown selection: {selection}")
    with threadpool_limits(limits=2):
        for dataset in ("dlpfc", "skin", "cancer"):
            rows = [run(case) for case in selected if case["dataset"] == dataset]
            if rows:
                save_scores(rows, dataset)

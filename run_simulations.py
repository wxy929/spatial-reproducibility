"""Run the fixed simulations: python run_simulations.py [case | depth | locations]."""

from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from simulation_perturbations import (DATA, cancer_cells, capture_grids, feature_relocation,
                                     local_coordinate_shuffle, neighbor_relocation)
from scores import (build_state, covariance, mapped_covariance_cosines,
                    physical_neighbors, prepare_pair, row_cosines, score_state)


RESULTS = Path(__file__).resolve().parent / "results"


def cases():
    return {
        "neighbor": dict(k=6, domain_k=2, iterations=50000),
        "feature": dict(k=6, domain_k=2, iterations=50000),
        "capture_cancer": dict(k=100, domain_k=7, iterations=5000),
        "capture_wmb": dict(k=100, domain_k=7, iterations=5000),
        "capture_wmb_shuffle": dict(k=100, domain_k=7, iterations=5000),
    }


def load_pair(name):
    if name.startswith("capture_wmb"):
        cells = ad.read_h5ad(DATA / "wmb_roi.h5ad")
        if name == "capture_wmb_shuffle":
            cells = local_coordinate_shuffle(cells)
        return capture_grids(cells, cells.uns["capture_origin"])
    cells = cancer_cells()
    if name == "capture_cancer":
        xy = cells.obsm["spatial"]
        keep = (xy[:, 0] >= 0) & (xy[:, 0] < 3800) & (xy[:, 1] >= 0) & (xy[:, 1] < 3000)
        return capture_grids(cells[keep].copy(), [0, 0])
    a, b = neighbor_relocation(cells)
    if name == "feature":
        return feature_relocation(cells, a)
    return a, b


def score_pair(a, b, config):
    xy_a, xy_b = a.obsm["spatial"], b.obsm["spatial"]
    pca_a, pca_b = prepare_pair(a, b)
    state = build_state(pca_a, pca_b, xy_a, xy_b, config["k"], config["iterations"])
    values = score_state(state, xy_a, xy_b, config["k"], domain_k=config["domain_k"])
    return values, state


def save_neighbor_scores(a, b, state):
    xy = a.obsm["spatial"]
    neighbors = physical_neighbors(xy, 6)
    covariance_a = covariance(state["smooth_a"], neighbors)
    covariance_b = covariance(state["smooth_b"], neighbors)
    forward = mapped_covariance_cosines(covariance_a, covariance_b, state["map_ab"])
    reverse = mapped_covariance_cosines(covariance_b, covariance_a, state["map_ba"])
    changed = a.obs.profile_id.to_numpy() != b.obs.profile_id.to_numpy()
    fraction = changed[neighbors].mean(axis=1)
    status = np.where(changed, "exchanged_center",
                      np.where(fraction > 0, "unchanged_exposed", "unchanged_remote"))
    local = pd.DataFrame(dict(
        spot=a.obs_names, x=xy[:, 0], y=xy[:, 1], center_status=status,
        exchanged_neighbor_fraction=fraction,
        Spot_A_to_B=row_cosines(state["pca_a"][:, :30], state["pca_b"][state["map_ab"], :30]),
        Neighbor=(forward + reverse) / 2,
    ))
    local.to_csv(RESULTS / "neighbor_local.csv", index=False)
    local.groupby("center_status").Neighbor.agg(["count", "mean", "median", "min", "max"]).to_csv(
        RESULTS / "neighbor_exposure.csv")


def neighbor_correspondences(a, b, state, balanced_scores):
    rows = [dict(correspondence="balanced_ot", **balanced_scores)]
    identity = np.arange(a.n_obs)
    true_ab = pd.Index(b.obs.profile_id).get_indexer(a.obs.profile_id)
    true_ba = pd.Index(a.obs.profile_id).get_indexer(b.obs.profile_id)
    for name, map_ab, map_ba in (("fixed_position", identity, identity),
                                ("true_profile_id", true_ab, true_ba)):
        mapped = dict(state, map_ab=map_ab, map_ba=map_ba, matched_a=identity, matched_b=map_ab,
                      mass_a=np.full(a.n_obs, 1 / a.n_obs), mass_b=np.full(b.n_obs, 1 / b.n_obs))
        values = score_state(mapped, a.obsm["spatial"], b.obsm["spatial"], 6, domain_k=2)
        rows.append(dict(correspondence=name, **values))
    pd.DataFrame(rows).to_csv(RESULTS / "neighbor_correspondences.csv", index=False)


def run_cases(names):
    rows = []
    for name in names:
        config = cases()[name]
        a, b = load_pair(name)
        values, state = score_pair(a, b, config)
        rows.append(dict(case=name, **values))
        if name == "neighbor":
            save_neighbor_scores(a, b, state)
            neighbor_correspondences(a, b, state, values)
        print(name, values, flush=True)
    output = RESULTS / "simulation_scores.csv"
    results = pd.DataFrame(rows)
    if output.exists():
        results = pd.concat([pd.read_csv(output), results], ignore_index=True)
    results.drop_duplicates("case", keep="last").sort_values("case").to_csv(output, index=False)


def run_depth():
    a, b = load_pair("neighbor")
    config = dict(k=6, domain_k=2, iterations=200000)
    back_to_a = pd.Index(b.obs.profile_id).get_indexer(a.obs.profile_id)
    xa, xb = a.X.astype(np.int64), b.X.astype(np.int64)
    rows = []
    for relocated, right in ((False, a), (True, b)):
        values, _ = score_pair(a, right, config)
        rows.append(dict(relocated=relocated, condition="original", replicate=0, **values))
    for replicate in range(20):
        seed_a, seed_b = 20261025 + 2 * replicate, 20261026 + 2 * replicate
        half_a = np.random.default_rng(seed_a).binomial(xa, .5)
        half_b = np.random.default_rng(seed_b).binomial(xb, .5)
        for condition, left_counts in (("both_half", half_a), ("b_only_half", xa)):
            for relocated, right_counts in ((False, half_b[back_to_a]), (True, half_b)):
                left, right = a.copy(), b.copy()
                left.X, right.X = left_counts.astype(np.float32), right_counts.astype(np.float32)
                values, _ = score_pair(left, right, config)
                rows.append(dict(relocated=relocated, condition=condition, replicate=replicate + 1,
                                 seed_a=seed_a if condition == "both_half" else -1, seed_b=seed_b, **values))
        print("Depth replicate", replicate + 1, flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "depth_scores.csv", index=False)


def run_locations():
    frozen = np.load(DATA / "area_shift_inputs.npz")
    cases_table = pd.read_csv(DATA / "area_shift_cases.csv")
    xy = frozen["coordinates"]
    obs, var = pd.DataFrame(index=frozen["coordinate_ids"].astype(str)), pd.DataFrame(index=frozen["genes"])
    rows = []
    for row in cases_table.to_dict("records"):
        a = ad.AnnData(frozen[f"{row['case_id']}_A"].astype(np.float32), obs=obs.copy(), var=var.copy())
        b = ad.AnnData(frozen[f"{row['case_id']}_B"].astype(np.float32), obs=obs.copy(), var=var.copy())
        a.obsm["spatial"], b.obsm["spatial"] = xy.copy(), xy.copy()
        values, _ = score_pair(a, b, cases()[row["simulation"]])
        rows.append(dict(**row, **values))
        print(row["case_id"], values, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "location_scores.csv", index=False)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(cases())
    with threadpool_limits(limits=1):
        if names == ["depth"]:
            run_depth()
        elif names == ["locations"]:
            run_locations()
        else:
            run_cases(names)

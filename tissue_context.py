"""Annotation comparisons, Skin programs, and Cancer marker summaries."""

import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from threadpoolctl import threadpool_limits

from run_real_data import DATA, RESULTS, cases, fit_pair
from scores import covariance, mapped_covariance_cosines, physical_neighbors, row_cosines, weighted_mean


SKIN_PROGRAMS = {
    "AD epidermal activation": ["S100A7", "S100A8", "S100A9", "KRT16"],
    "Basal keratinocytes": ["KRT5", "KRT14", "KRT15"],
    "Granular/barrier differentiation": ["KRT2", "IVL", "SBSN", "CALML5", "DSC1", "LOR", "FLG", "SPINK5"],
    "Dermal fibroblasts": ["DCN", "COL1A1", "COL1A2"],
    "Sebaceous lipid/transport": ["ALOX15B", "APOC1", "FABP7", "FADS1", "FADS2", "FASN", "PPARG", "RARRES1"],
}
CANCER_MARKERS = ["CEACAM6", "FASN", "KRT15", "ACTA2", "POSTN"]
ANNOTATION_GROUPS = {
    "D2": ["DCIS_2"], "T": ["Invasive_Tumor", "Prolif_Invasive_Tumor"],
    "Myo_ACTA2": ["Myoepi_ACTA2+"], "Myo_KRT15": ["Myoepi_KRT15+"],
    "Stromal": ["Stromal"], "Macrophages": ["Macrophages_1", "Macrophages_2"],
    "Endothelial": ["Endothelial"], "Lymphoid": ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells"],
}
CANCER_BOUNDS = {
    "cancer_R1_D2": (5500, 7500, 2200, 4000), "cancer_R2_D2": (5500, 7500, 0, 2000),
    "cancer_R1_T": (1500, 3500, 400, 2200), "cancer_R2_T": (600, 2000, 0, 2200),
}


def annotation_similarity(left, right, neighbors_a, neighbors_b, state, categories):
    proportions = []
    for data, neighbors in [(left, neighbors_a), (right, neighbors_b)]:
        labels = data.obs.annotation.to_numpy().astype(str)
        p = np.column_stack([(labels[neighbors] == label).mean(axis=1) for label in categories])
        proportions.append(p.astype(np.float32).astype(float))
    pa, pb = proportions
    directional = []
    for a, b, mapping, mass in [(pa, pb, state["map_ab"], state["mass_a"]),
                               (pb, pa, state["map_ba"], state["mass_b"])]:
        matched = b[mapping]
        midpoint = (a + matched) / 2
        divergence = .5 * np.sum(rel_entr(a, midpoint) + rel_entr(matched, midpoint), axis=1) / np.log(2)
        directional.append(weighted_mean(1 - np.clip(divergence, 0, 1), mass))
    return np.mean(directional)


def skin_programs():
    slices = [ad.read_h5ad(DATA / f"skin_{sid}.h5ad") for sid in ("V1", "V2", "V3", "V4")]
    expression = pd.concat([data.obsm["program_expression"] for data in slices], ignore_index=True)
    scale = expression.std(ddof=0).replace(0, 1)
    standardized = (expression - expression.mean()) / scale
    rows = []
    start = 0
    for sid, data in zip(("V1", "V2", "V3", "V4"), slices):
        row = dict(slice=sid)
        for program, genes in SKIN_PROGRAMS.items():
            row[program] = standardized.iloc[start:start + data.n_obs][genes].mean(axis=1).mean()
        rows.append(row)
        start += data.n_obs
    return pd.DataFrame(rows)


def cancer_pair_context(case, left, right, state, neighbors_a, neighbors_b):
    cov_a = covariance(state["smooth_a"], neighbors_a)
    cov_b = covariance(state["smooth_b"], neighbors_b)
    neighbor_a = mapped_covariance_cosines(cov_a, cov_b, state["map_ab"])
    neighbor_b = mapped_covariance_cosines(cov_b, cov_a, state["map_ba"])
    ia, ib = state["matched_a"], state["matched_b"]
    matched_spot = row_cosines(state["pca_a"][ia, :30], state["pca_b"][ib, :30])
    frames, maps = {}, {}
    for side, other, local_neighbor in [
        ("a", "b", neighbor_a), ("b", "a", neighbor_b)
    ]:
        name = case["left" if side == "a" else "right"]
        partner = case["right" if side == "a" else "left"]
        mapping = state[f"map_{side}{other}"]
        spot = row_cosines(state[f"pca_{side}"][:, :30], state[f"pca_{other}"][mapping, :30])
        data = left if side == "a" else right
        frames[name] = pd.DataFrame(dict(slice=name, spot_id=data.obs_names, Spot=spot, Neighbor=local_neighbor))
        maps[name] = (partner, mapping)
    agreement = left.obs.annotation.to_numpy()[ia] == right.obs.annotation.to_numpy()[ib]
    matches = pd.DataFrame(dict(Spot=matched_spot, agreement=agreement,
                                decile=pd.qcut(matched_spot, 10, labels=False) + 1))
    deciles = matches.groupby("decile").agg(n_pairs=("Spot", "size"), spot_mean=("Spot", "mean"),
                                            annotation_agreement=("agreement", "mean")).reset_index()
    deciles.insert(0, "case", case["name"])
    return frames, maps, deciles


def cancer_context(frames, maps):
    inputs = {name: ad.read_h5ad(DATA / f"{name}.h5ad") for name in CANCER_BOUNDS}
    totals = {name: np.asarray(data.X.sum(axis=1)).ravel().astype(float) for name, data in inputs.items()}
    pooled = np.concatenate(list(totals.values()))
    target_total = np.median(pooled[pooled > 0])
    for name, data in inputs.items():
        frame = frames[name]
        labels = data.obs.annotation.to_numpy()
        xy = data.obsm["spatial"]
        neighbors = physical_neighbors(xy, 100)
        target = np.isin(labels, ANNOTATION_GROUPS["D2" if name.endswith("D2") else "T"])
        fraction = target[neighbors].mean(axis=1)
        frame["is_target"] = target
        frame["target_fraction"] = fraction
        frame["context"] = np.select([fraction <= .25, fraction <= .75, fraction <= .90],
                                      ["Surroundings", "Mixed neighborhood", "Transition"], default="Target-dominated")
        radius = np.linalg.norm(xy[neighbors[:, -1]] - xy, axis=1)
        xmin, xmax, ymin, ymax = CANCER_BOUNDS[name]
        boundary_distance = np.minimum.reduce([xy[:, 0] - xmin, xmax - xy[:, 0], xy[:, 1] - ymin, ymax - xy[:, 1]])
        frame["crop_clear"] = boundary_distance >= radius
        grouped_labels = np.full(data.n_obs, "Other", dtype=object)
        for group, members in ANNOTATION_GROUPS.items():
            grouped_labels[np.isin(labels, members)] = group
        frame["annotation_group"] = grouped_labels
        counts = data[:, CANCER_MARKERS].X.toarray().astype(float)
        expression = np.log1p(counts * target_total / np.maximum(totals[name], 1)[:, None])
        for j, gene in enumerate(CANCER_MARKERS):
            frame[f"expr_{gene}"] = expression[:, j]
    for name, frame in frames.items():
        partner, mapping = maps[name]
        frame["paired_crop_clear"] = frame.crop_clear.to_numpy() & frames[partner].crop_clear.to_numpy()[mapping]
    cells = pd.concat(frames.values(), ignore_index=True)
    selected = cells.loc[cells.is_target & cells.paired_crop_clear
                         & cells.context.isin(["Mixed neighborhood", "Target-dominated"])]
    summary = selected.groupby(["slice", "context"]).agg(
        Spot=("Spot", "mean"), Neighbor=("Neighbor", "mean"), n=("Spot", "size"))
    return cells, summary.reset_index()


def run(selection):
    if selection not in ("all", "skin", "dlpfc", "cancer"):
        raise SystemExit(f"Unknown selection: {selection}")
    if selection in ("all", "skin"):
        skin_programs().to_csv(RESULTS / "skin_programs.csv", index=False)
    annotations, deciles, local, maps = [], [], {}, {}
    for case in cases():
        is_dlpfc = case["dataset"] == "dlpfc" and case["main"]
        is_cancer = case["name"] in ("cancer_R1_D2_R2_D2", "cancer_R1_T_R2_T")
        if (is_dlpfc or is_cancer) and selection in ("all", case["dataset"]):
            left, right, state = fit_pair(case)
            neighbors_a = physical_neighbors(left.obsm["spatial"], case["k"])
            neighbors_b = physical_neighbors(right.obsm["spatial"], case["k"])
            if is_dlpfc:
                value = annotation_similarity(left, right, neighbors_a, neighbors_b, state,
                                              [str(i) for i in range(7)])
                annotations.append(dict(case=case["name"], annotation_similarity=value))
            else:
                frames, mapping, table = cancer_pair_context(case, left, right, state, neighbors_a, neighbors_b)
                local.update(frames)
                maps.update(mapping)
                deciles.append(table)
            del left, right, state
    if selection in ("all", "dlpfc"):
        pd.DataFrame(annotations).to_csv(RESULTS / "dlpfc_annotation_scores.csv", index=False)
    if selection in ("all", "cancer"):
        pd.concat(deciles, ignore_index=True).to_csv(RESULTS / "cancer_annotation_deciles.csv", index=False)
        cells, summary = cancer_context(local, maps)
        cells.to_csv(RESULTS / "cancer_local_context.csv.gz", index=False)
        summary.to_csv(RESULTS / "cancer_context_scores.csv", index=False)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    selection = sys.argv[1] if len(sys.argv) > 1 else "all"
    with threadpool_limits(limits=2):
        run(selection)

"""Construct the paper's fixed profile, gene-block, and capture-grid perturbations."""

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


DATA = Path(__file__).resolve().parent / "data"


def cancer_cells():
    with h5py.File(DATA / "cancer_r1_counts.h5") as handle:
        matrix = handle["matrix"]
        counts = csc_matrix((matrix["data"][:], matrix["indices"][:],
                             matrix["indptr"][:]), shape=tuple(matrix["shape"][:])).T.tocsr()
        ids = matrix["barcodes"].asstr()[:]
        genes = matrix["features/name"].asstr()[:]
    metadata = pd.read_csv(DATA / "cancer_r1_cells.csv.gz", dtype={"cell_id": str}).set_index("cell_id")
    cells = ad.AnnData(counts, obs=metadata.loc[ids], var=pd.DataFrame(index=genes))
    cells.obsm["spatial"] = cells.obs[["x_centroid", "y_centroid"]].to_numpy(float)
    return cells


def match_positions(left, right):
    left = (left - left.mean(axis=0)) / left.std(axis=0)
    right = (right - right.mean(axis=0)) / right.std(axis=0)
    return linear_sum_assignment(cdist(left, right, metric="sqeuclidean"))


def nearest(cells, indices, center, count):
    distance = np.linalg.norm(cells.obsm["spatial"][indices] - center, axis=1)
    order = np.lexsort((cells.obs_names[indices].astype(int), distance))
    return indices[order[:count]]


def square(xy, center, side):
    return (np.abs(xy - center) <= side / 2).all(axis=1)


def neighbor_relocation(cells):
    xy = cells.obsm["spatial"]
    tumor = cells.obs.original_label.isin(["Invasive_Tumor", "Prolif_Invasive_Tumor"]).to_numpy()
    dcis = cells.obs.original_label.eq("DCIS_1").to_numpy()
    base = np.flatnonzero(square(xy, [3190, 1890], 300) & (tumor | dcis))
    base = base[np.argsort(cells.obs_names[base].astype(int))]
    donor = np.flatnonzero(square(xy, [3705, 2965], 100) & dcis)
    donor = donor[np.argsort(cells.obs_names[donor].astype(int))]
    source_tumor = nearest(cells, base[square(xy[base], [3200, 1985], 100) & tumor[base]], [3200, 1985], 55)
    target = nearest(cells, base[square(xy[base], [3090, 1790], 100) & tumor[base]], [3090, 1790], 56)
    donor_rows, source_rows = match_positions(xy[donor], xy[source_tumor])
    profiles = base.copy()
    lookup = pd.Index(base)
    profiles[lookup.get_indexer(source_tumor[source_rows])] = donor[donor_rows]
    source = np.r_[base[dcis[base]], source_tumor]
    source = source[np.argsort(cells.obs_names[source].astype(int))]
    left, right = match_positions(xy[source], xy[target])
    source_rows, target_rows = lookup.get_indexer(source[left]), lookup.get_indexer(target[right])
    a = ad.AnnData(cells.X[profiles].toarray().astype(np.float32),
                   obs=cells.obs.iloc[base].copy(), var=cells.var.copy())
    a.obsm["spatial"] = xy[base].copy()
    a.obs["profile_id"] = cells.obs_names[profiles]
    a.obs["profile_label"] = cells.obs.original_label.iloc[profiles].to_numpy()
    b = a.copy()
    b.X[source_rows], b.X[target_rows] = a.X[target_rows], a.X[source_rows]
    profile_b = a.obs.profile_id.to_numpy(copy=True)
    profile_b[source_rows], profile_b[target_rows] = profile_b[target_rows], profile_b[source_rows]
    b.obs["profile_id"] = profile_b
    b.obs["profile_label"] = cells.obs.loc[profile_b, "original_label"].to_numpy()
    return a, b


def feature_relocation(cells, scaffold):
    geometry = pd.read_csv(DATA / "feature_blocks.csv", dtype=str)
    source = scaffold.obs_names.get_indexer(geometry.source_cell_id)
    target = scaffold.obs_names.get_indexer(geometry.recipient_cell_id)
    assignments = pd.read_csv(DATA / "feature_donors.csv", dtype=str)
    donors = cells.obs_names.get_indexer(assignments.donor_cell_id)
    a = scaffold.copy()
    profile_ids = scaffold.obs.profile_id.to_numpy(copy=True)
    donor_counts = cells.X[donors].toarray()
    for donor_row, assignment in enumerate(assignments.itertuples()):
        destination = scaffold.obs_names.get_loc(assignment.source_coordinate_cell_id)
        if assignment.donor_origin == "native21_moved":
            vacated = scaffold.obs_names.get_loc(assignment.donor_cell_id)
            a.X[vacated] = scaffold.X[destination]
            profile_ids[vacated] = scaffold.obs.profile_id.iloc[destination]
        a.X[destination] = donor_counts[donor_row]
        profile_ids[destination] = assignment.donor_cell_id
    a.obs["profile_id"] = profile_ids
    a.obs["profile_label"] = cells.obs.loc[profile_ids, "original_label"].to_numpy()
    b = a.copy()
    genes = a.var_names.get_indexer(["MKI67", "TOP2A", "CENPF"])
    b.X[np.ix_(source, genes)] = a.X[np.ix_(target, genes)]
    b.X[np.ix_(target, genes)] = a.X[np.ix_(source, genes)]
    return a, b


def capture_grids(cells, origin):
    """Sum raw counts in 55-um disks on two translated triangular lattices."""
    grid_x, grid_y = np.meshgrid(np.arange(37), np.arange(34))
    row_pitch = 100 * np.sqrt(3) / 2
    grids = []
    for shift in (0, 27.5):
        positions = np.column_stack((origin[0] + shift + 27.5 + grid_x.ravel() * 100 + (grid_y.ravel() % 2) * 50,
                                     origin[1] + shift + 27.5 + grid_y.ravel() * row_pitch))
        distance, index = NearestNeighbors(n_neighbors=1).fit(positions).kneighbors(cells.obsm["spatial"])
        captured = distance[:, 0] <= 27.5 + 1e-10
        rows = np.flatnonzero(captured)
        assignment = csr_matrix((np.ones(len(rows), dtype=np.int8), (index[captured, 0], rows)),
                                shape=(len(positions), cells.n_obs))
        counts = (assignment @ cells.X).tocsr().astype(np.float32)
        grid = ad.AnnData(counts, var=cells.var.copy())
        grid.obsm["spatial"] = positions
        grids.append(grid)
    positive = (np.asarray(grids[0].X.sum(axis=1)).ravel() > 0) & (np.asarray(grids[1].X.sum(axis=1)).ravel() > 0)
    return tuple(grid[positive].copy() for grid in grids)


def local_coordinate_shuffle(cells):
    shuffled = cells.copy()
    xy = cells.obsm["spatial"]
    tiles = np.floor((xy - cells.uns["capture_origin"]) / 250).astype(int)
    tile_id = tiles[:, 1] * (tiles[:, 0].max() + 1) + tiles[:, 0]
    permutation = np.arange(cells.n_obs)
    rng = np.random.default_rng(0)
    for tile in np.unique(tile_id):
        members = np.flatnonzero(tile_id == tile)
        n_move = int(np.floor(.6 * len(members)))
        if n_move >= 2:
            chosen = rng.choice(members, n_move, replace=False)
            permutation[chosen] = np.roll(chosen, 1)
    shuffled.obsm["spatial"] = xy[permutation]
    return shuffled

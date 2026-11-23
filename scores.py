"""Pairwise preprocessing, balanced transport, and four reproducibility scores."""

import anndata as ad
import numpy as np
import ot
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def prepare_pair(a, b, seed=0, dense_order=None):
    """Fit a joint 50-component PCA to the shared genes of two AnnData slices."""
    genes = a.var_names.intersection(b.var_names)
    joint = ad.concat([a[:, genes], b[:, genes]], index_unique="-")
    # Dense layout affects normalization sums and nearly tied matches.
    if dense_order is not None:
        joint.X = np.asarray(joint.X, order=dense_order)
    sc.pp.normalize_total(joint)
    sc.pp.log1p(joint)
    sc.pp.highly_variable_genes(joint, flavor="seurat", n_top_genes=5000)
    sc.pp.pca(joint, n_comps=50, mask_var="highly_variable", random_state=seed)
    pca = np.asarray(joint.obsm["X_pca"], dtype=float)
    return pca[:a.n_obs], pca[a.n_obs:]


def spatial_smoothing(xy, k):
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    distances, indices = distances[:, 1:], indices[:, 1:]
    q25, q50 = np.quantile(distances, [0.25, 0.5])
    rate = np.where(distances <= q25, 0.5, np.where(distances <= q50, 1.0, 2.0))
    weights = np.exp(-rate * distances / distances.max(axis=1, keepdims=True))
    n = len(xy)
    rows = np.repeat(np.arange(n), k)
    smoother = csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(n, n))
    smoother += csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    return smoother.multiply(1 / np.asarray(smoother.sum(axis=1)))


def physical_neighbors(xy, k):
    """Exclude each center and order the returned candidates by distance and index."""
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    neighbors = np.empty((len(xy), k), dtype=np.int32)
    for center in range(len(xy)):
        keep = indices[center] != center
        ids, ds = indices[center, keep], distances[center, keep]
        neighbors[center] = ids[np.lexsort((ids, ds))][:k]
    return neighbors


def transport(left, right, iterations=5000):
    cost = ot.dist(left, right, metric="sqeuclidean")
    maximum = cost.max()
    if maximum > 0:
        cost /= maximum
    na, nb = len(left), len(right)
    coupling = ot.sinkhorn(
        np.full(na, 1 / na), np.full(nb, 1 / nb), cost, 0.01,
        numItermax=iterations, stopThr=1e-9,
    )
    if na >= nb:
        maxima = coupling.max(axis=1, keepdims=True)
        ia, ib = np.where((coupling == maxima) & (maxima > 0))
    else:
        maxima = coupling.max(axis=0, keepdims=True)
        ib, ia = np.where(((coupling == maxima) & (maxima > 0)).T)
    return dict(map_ab=coupling.argmax(axis=1), map_ba=coupling.argmax(axis=0),
                mass_a=coupling.sum(axis=1), mass_b=coupling.sum(axis=0),
                matched_a=ia, matched_b=ib)


def build_state(pca_a, pca_b, xy_a, xy_b, k, iterations=5000, ot_pcs=50):
    pca_a, pca_b = np.asarray(pca_a, dtype=float), np.asarray(pca_b, dtype=float)
    smooth_a = np.asarray(spatial_smoothing(xy_a, k) @ pca_a)
    smooth_b = np.asarray(spatial_smoothing(xy_b, k) @ pca_b)
    matches = transport(smooth_a[:, :ot_pcs], smooth_b[:, :ot_pcs], iterations)
    return dict(pca_a=pca_a, pca_b=pca_b, smooth_a=smooth_a, smooth_b=smooth_b,
                **matches)


def row_cosines(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.full(len(a), np.nan)
    np.divide(np.sum(a * b, axis=1), denominator, out=values, where=denominator > 0)
    return values


def covariance(features, neighbors, d=50):
    values = np.asarray(features[:, :d], dtype=float)
    cov = np.empty((len(values), d, d), dtype=np.float32)
    for start in range(0, len(values), 256):
        local = values[neighbors[start:start + 256]]
        local -= local.mean(axis=1, keepdims=True)
        cov[start:start + 256] = np.einsum("bkd,bke->bde", local, local, optimize=True) / neighbors.shape[1]
    return cov


def mapped_covariance_cosines(a, b, mapping):
    values = np.full(len(a), np.nan)
    for start in range(0, len(a), 256):
        left = a[start:start + 256].astype(float)
        right = b[mapping[start:start + 256]].astype(float)
        numerator = np.einsum("bij,bij->b", left, right)
        denominator = np.sqrt(np.einsum("bij,bij->b", left, left) * np.einsum("bij,bij->b", right, right))
        np.divide(numerator, denominator, out=values[start:start + 256], where=denominator > 0)
    return np.clip(values, -1, 1)


def weighted_mean(values, mass):
    keep = np.isfinite(values) & (mass > 0)
    return np.sum(values[keep] * mass[keep]) / mass[keep].sum()


def score_state(state, xy_a, xy_b, k_neighbor, spot_pcs=30, feature_pcs=10,
                neighbor_pcs=50, domain_pcs=50, domain_k=7, seed=0):
    sa, sb = state["smooth_a"], state["smooth_b"]
    ia, ib = state["matched_a"], state["matched_b"]
    spot = row_cosines(state["pca_a"][ia, :spot_pcs], state["pca_b"][ib, :spot_pcs])
    feature = row_cosines(sa[ia, :feature_pcs].T, sb[ib, :feature_pcs].T)
    ca = covariance(sa, physical_neighbors(xy_a, k_neighbor), neighbor_pcs)
    cb = covariance(sb, physical_neighbors(xy_b, k_neighbor), neighbor_pcs)
    nab = weighted_mean(mapped_covariance_cosines(ca, cb, state["map_ab"]), state["mass_a"])
    nba = weighted_mean(mapped_covariance_cosines(cb, ca, state["map_ba"]), state["mass_b"])
    labels = KMeans(n_clusters=domain_k, n_init=50, random_state=seed).fit_predict(
        np.vstack([sa[:, :domain_pcs], sb[:, :domain_pcs]]))
    domain = normalized_mutual_info_score(labels[:len(sa)][ia], labels[len(sa):][ib])
    return dict(Spot=float(np.nanmean(spot)), Feature=float(np.mean(feature)),
                Neighbor=float((nab + nba) / 2), Domain=float(domain))

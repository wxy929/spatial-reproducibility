"""Mean-expression correlations and gene-wise Moran's I."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from scores import physical_neighbors


def moran_i(values, coordinates, k):
    neighbors = physical_neighbors(coordinates, k)
    n = len(values)
    graph = sparse.csr_matrix((np.full(n * k, 1 / k),
                              (np.repeat(np.arange(n), k), neighbors.ravel())),
                             shape=(n, n))
    centered = values - values.mean(axis=0)
    denominator = np.sum(centered ** 2, axis=0)
    numerator = np.sum(centered * (graph @ centered), axis=0)
    return np.divide(n / graph.sum() * numerator, denominator,
                     out=np.full(values.shape[1], np.nan), where=denominator > 0)


def compare(left, right, k):
    # Keep every input gene, including constant columns, before normalization.
    joint = ad.concat([left, right], index_unique="-")
    # Column-contiguous reductions keep near-tied Moran ranks reproducible.
    if not sparse.issparse(joint.X):
        joint.X = np.asfortranarray(joint.X)
    sc.pp.normalize_total(joint)
    sc.pp.log1p(joint)
    values = joint.X.toarray() if sparse.issparse(joint.X) else joint.X
    a, b = np.asarray(values, dtype=float)[:len(left)], np.asarray(values, dtype=float)[len(left):]
    ia = moran_i(a, left.obsm["spatial"], k)
    ib = moran_i(b, right.obsm["spatial"], k)
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    genes = pd.DataFrame(dict(gene=joint.var_names, mean_log1p_A=ma,
                             mean_log1p_B=mb, moran_A=ia, moran_B=ib, delta_moran=ib - ia))
    valid = np.isfinite(ia) & np.isfinite(ib)
    summary = dict(mean_expression_pearson=pearsonr(ma, mb).statistic,
                   mean_expression_spearman=spearmanr(ma, mb).statistic,
                   moran_pearson=pearsonr(ia[valid], ib[valid]).statistic,
                   moran_spearman=spearmanr(ia[valid], ib[valid]).statistic,
                   mean_absolute_delta_moran=np.abs(ib[valid] - ia[valid]).mean())
    return genes, summary


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    from run_simulations import cases, load_pair

    output = Path(__file__).resolve().parent / "results"
    output.mkdir(exist_ok=True)
    summaries, gene_tables = [], []
    with threadpool_limits(limits=2):
        for name in ("neighbor", "feature", "capture_cancer", "capture_wmb"):
            left, right = load_pair(name)
            genes, summary = compare(left, right, cases()[name]["k"])
            genes.insert(0, "case", name)
            gene_tables.append(genes)
            summaries.append(dict(case=name, **summary))
    pd.concat(gene_tables, ignore_index=True).to_csv(output / "moran_genes.csv", index=False, na_rep="NA")
    pd.DataFrame(summaries).to_csv(output / "conventional_metrics.csv", index=False)

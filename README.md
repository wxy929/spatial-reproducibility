# Spatial Transcriptomic Reproducibility

Code and analysis inputs for *Multilevel assessment distinguishes molecular and neighborhood agreement in spatial transcriptomics* (submitted manuscript).

## Project overview

The method takes expression matrices and spatial coordinates from two tissue slices. It selects shared genes, jointly normalizes and log-transforms expression, and fits a common PCA representation. Within-slice spatial smoothing adds local context; balanced optimal transport then establishes matches across slices. Four scores summarize different aspects of agreement:

- **Spot:** cosine similarity of unsmoothed expression profiles at matched spots.
- **Feature:** similarity of smoothed principal-component patterns across matches.
- **Neighbor:** similarity of local expression covariance around matched spots.
- **Domain:** agreement between jointly inferred cluster labels at matches.

## Installation

Use Python 3.10.18. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

```bash
python run_simulations.py neighbor
```

This runs the 540-position Neighbor simulation and writes `results/simulation_scores.csv` and the corresponding summaries. Reference scores are Spot 0.977730, Feature 0.960201, Neighbor 0.838264, and
Domain 1.000000.

## Reproducing analyses

| Analysis | Command |
| --- | --- |
| Five simulation comparisons | `python run_simulations.py` |
| One simulation | `python run_simulations.py feature` |
| Sampling-depth controls | `python run_simulations.py depth` |
| Robustness to perturbation location (10 cases each) | `python run_simulations.py locations` |
| Main real-data comparisons (9 DLPFC, 6 Skin, 7 Cancer) | `python run_real_data.py` |
| All 66 DLPFC pairs | `python run_real_data.py dlpfc` |
| Skin / Cancer pairs | `python run_real_data.py skin` / `python run_real_data.py cancer` |
| One real-data pair | `python run_real_data.py dlpfc_151675_151676` |
| Spatial permutation baseline (100 shuffles) | `python run_baselines.py neighbor` |
| Parameter sensitivity and DLPFC seed analysis | `python run_sensitivity.py dlpfc` |
| Cancer parameter sensitivity | `python run_sensitivity.py cancer` |
| Mean-expression correlations and Moran's I | `python conventional_metrics.py` |
| Tissue-context summaries | `python tissue_context.py` |

Simulation names are `neighbor`, `feature`, `capture_cancer`, `capture_wmb`, and `capture_wmb_shuffle`. Baselines accept the first four names or an exact main real-data pair name, such as `cancer_R1_D2_R2_D2`. Tissue-context analyses also accept `skin`, `dlpfc`, or `cancer` separately.

To calculate score correlations, first generate the complete DLPFC table:

```bash
python run_real_data.py dlpfc
python score_correlations.py
```

Cancer transport calculations require substantial memory; use at least 32 GB RAM.

## Data and results

`data/` contains processed expression matrices, spatial coordinates, annotations, and the fixed inputs needed to construct the simulations. All expression-matrix gene columns are included. `input_sources.csv` identifies the processed tissue matrices. Original resources are [DLPFC / spatialLIBD](https://research.libd.org/spatialLIBD/), [Cancer R1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780153), [Cancer R2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780154), [Skin](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206391), and [WMB](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-2.html).

`results/paper_scores.csv` contains the 84 reference comparisons. Scripts write new tables to `results/`. Real-data scores use separate `dlpfc_scores.csv`, `skin_scores.csv`, and `cancer_scores.csv` files. Rerunning a subset updates only those comparisons, including in `simulation_scores.csv`. The correlation script requires all 66 unique DLPFC pairs. PCA and transport matrices are not saved. Cancer context outputs comprise a compressed spot-level table for the marker maps and a small score-summary table for the tissue-context comparison.

## Citation and license

Please cite *Multilevel assessment distinguishes molecular and neighborhood agreement in spatial transcriptomics* (submitted manuscript), and the original data sources used in your analysis. Publication details will be added when available.

The original code in this repository is licensed under the [MIT License](LICENSE). Third-party datasets and software dependencies are subject to their respective licenses and terms of use.

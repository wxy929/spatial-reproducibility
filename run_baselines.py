"""The paper's 100 A–randomized-B spatial permutations."""
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_real_data
import run_simulations
from scores import prepare_pair, build_state, score_state


def run(name):
    simulations = {key: value for key, value in run_simulations.cases().items()
                   if key != "capture_wmb_shuffle"}
    if name in simulations:
        case = simulations[name]
        left, right = run_simulations.load_pair(name)
        seed = 20260917 if name == "neighbor" else 20260922
        iterations = 200000 if name in ("neighbor", "feature") else 5000
    else:
        case = next(c for c in run_real_data.cases() if c["name"] == name and c["main"])
        left, right = run_real_data.load_pair(case)
        seed = 20260920 if case["dataset"] == "dlpfc" else 20260922
        iterations = 5000
    xy_a, xy_b = left.obsm["spatial"], right.obsm["spatial"]
    a, b = prepare_pair(left, right, dense_order="C" if case.get("dataset") == "skin" else None)
    rows = []
    for permutation in range(101):
        current_seed = seed + permutation - 1
        order = np.arange(len(b)) if permutation == 0 else np.random.default_rng(current_seed).permutation(len(b))
        state = build_state(a, b[order], xy_a, xy_b, case["k"], iterations=iterations)
        values = score_state(state, xy_a, xy_b, k_neighbor=case["k"], domain_k=case["domain_k"])
        rows.append(dict(permutation=permutation, seed=-1 if permutation == 0 else current_seed, **values))
        print(name, permutation, flush=True)
    output = run_real_data.RESULTS
    output.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output / f"{name}_shuffle.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run(sys.argv[1])

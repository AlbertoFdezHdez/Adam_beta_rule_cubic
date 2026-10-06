"""Reconstruct representative protocol ablations from saved gradient sketches.

All beta decisions precede the validation-loss lookup. The full exploratory
timing and budget audit summaries are supplied alongside the compact sketches.
No network training, GPU, or external data are needed.
"""
import json
from pathlib import Path

import numpy as np

from reproduce_selection import aggregate

BASE = Path(__file__).resolve().parent


def select_beta(path, step, budget, grid):
    with np.load(path) as data:
        positions = {int(t): i for i, t in enumerate(data["steps"])}
        group_positions = {int(t): i for i, t in enumerate(
            data["group_steps"] if "group_steps" in data else data["steps"])}
        full = data["proj_r_full"].astype(float)
        group = data["proj_r_group"].astype(float)
    endpoint = positions[step]
    G = group.shape[1]
    eta = np.mean(np.sum((group[group_positions[step]] - full[endpoint]) ** 2, axis=1))
    V = budget / (G - 1) * eta
    lags = np.array([20., 50., 100.])
    changes = np.array([np.sum((full[endpoint] - full[positions[step - int(n)]]) ** 2) / n ** 2
                        for n in lags])
    intercept, slope = np.linalg.lstsq(
        np.column_stack([np.ones(3), 1 / lags ** 2]), changes, rcond=None)[0]
    D = max(0., float(intercept)) + max(0., float(slope)) / 10000
    H_min, H_max = 1 / (1 - min(grid)), 1 / (1 - max(grid))
    H = min(H_max, max(H_min, float(V / (4 * D)) ** (1 / 3))) if D > 0 else H_max if V > 0 else H_min
    beta = min(grid, key=lambda b: abs(b - (1 - 1 / H)))
    return dict(V=float(V), D=D, memory=H, beta=beta)


def main():
    inputs = json.loads((BASE / "inputs.json").read_text(encoding="utf-8"))
    timing = json.loads((BASE / "timing_audit.json").read_text(encoding="utf-8"))
    budget = json.loads((BASE / "probe_budget_audit.json").read_text(encoding="utf-8"))
    rows, summary = [], []
    fixed_gaps = []
    for case in inputs["cases"]:
        losses = {float(b): value for b, value in case["seed1_grid_losses"].items()}
        fixed_gaps.append(100 * (losses[inputs["baseline"]] / min(losses.values()) - 1))
    summary.append(dict(protocol="fixed", **aggregate(fixed_gaps)))
    protocols = [(16,t) for t in range(150,501,25)] + [(m,200) for m in [2,4,8]]
    for m, step in protocols:
        gaps = []
        for case in inputs["cases"]:
            decision = select_beta(BASE / "robustness_probes" / (case["workload"] + "_mb" + str(m) + ".npz"),
                                   step, m, inputs["grid"])
            # Only now access candidate validation losses.
            losses = {float(b): value for b, value in case["seed1_grid_losses"].items()}
            gap = 100 * (losses[decision["beta"]] / min(losses.values()) - 1)
            gaps.append(gap)
            rows.append(dict(workload=case["workload"], budget=m, step=step,
                             gap_pct=gap, **decision))
        calculated = aggregate(gaps)
        if m == 16:
            expected = next(r for r in timing["summary"] if r["budget"] == m and r["step"] == step)
        else:
            expected = next(r for r in budget["summary"] if r["budget"] == m and r["factor"] == 4)
        for key, archived_key in [("mean", "mean_gap_pct"), ("maximum", "max_gap_pct"),
                                  ("cvar25", "cvar25_gap_pct")]:
            assert abs(calculated[key] - expected[archived_key]) < 1e-10
        summary.append(dict(protocol="mb" + str(m) + "_step" + str(step), **calculated))
    output = dict(summary=summary, per_workload=rows,
                  scope="Representative table; full exploratory summaries accompany the package.")
    (BASE / "reconstructed_robustness.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

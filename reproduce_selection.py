"""Reconstruct selection and manuscript tables from compact, anonymized inputs.

Run: python reproduce_selection.py
Dependency: Python 3.10+ and NumPy. No training frameworks, GPU, or downloads.
All input gradients were measured before the selection decision.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent


def aggregate(gaps):
    values = sorted(gaps, reverse=True)
    return {"mean": float(np.mean(values)), "maximum": values[0],
            "cvar25": float(np.mean(values[:math.ceil(len(values) / 4)]))}


def stationary_root(V, D):
    if D == 0:
        return math.inf if V > 0 else 1.0
    if V == 0:
        return 1.0
    low, high = 1.0, max(2.0, 2 * (V / D) ** (1 / 3))
    for _ in range(100):
        midpoint = (low + high) / 2
        if D * (midpoint - 1) * (2 * midpoint - 1) ** 2 < V:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def constrained_memory(V, D, grid):
    lower, upper = 1 / (1 - min(grid)), 1 / (1 - max(grid))
    if D == 0:
        return upper if V > 0 else lower
    return min(upper, max(lower, (V / (4 * D)) ** (1 / 3)))


def finite_history_score(V, D, beta, steps=200):
    ages = np.arange(steps, dtype=float)
    weights = (1 - beta) * beta ** ages / (1 - beta ** steps)
    return float(V * np.dot(weights, weights) + D * np.dot(weights, ages) ** 2)


def main():
    inputs = json.loads((BASE / "inputs.json").read_text(encoding="utf-8"))
    grid = inputs["grid"]
    fixed = inputs["baseline"]
    nearest = lambda value: min(grid, key=lambda beta: abs(beta - value))
    rows = []
    diagnostics = {name: [] for name in ["fixed", "rule", "stationary_root", "leading_grid", "stationary_grid", "finite_history_grid",
                                        "constant_target", "endpoint_only", "intercept_only"]}
    for case in inputs["cases"]:
        with np.load(BASE / "probes" / (case["workload"] + ".npz")) as data:
            positions = {int(t): i for i, t in enumerate(data["steps"])}
            full = data["proj_r_full"].astype(float)
            group = data["proj_r_group"].astype(float)
        endpoint = positions[200]
        eta = float(np.mean(np.sum((group[endpoint] - full[endpoint]) ** 2, axis=1)))
        center = group[endpoint].mean(axis=0)
        pooling = float(np.sum((center - full[endpoint]) ** 2))
        m = inputs["effective_batches_per_checkpoint"]
        G = group.shape[1]
        V = m / (G - 1) * eta
        lags = np.array([20, 50, 100])
        y = np.array([np.sum((full[endpoint] - full[positions[200 - n]]) ** 2) / n ** 2 for n in lags])
        intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(3), 1 / lags ** 2]), y, rcond=None)[0]
        intercept = max(0.0, float(intercept))
        slope = max(0.0, float(slope))
        D = intercept + slope / 10000
        H = constrained_memory(V, D, grid)
        beta = nearest(1 - 1 / H)
        variants = {
            "fixed": fixed, "rule": beta,
            "stationary_root": nearest(1 - 1 / stationary_root(V, D)),
            "leading_grid": min(grid, key=lambda b: V * (1 - b) / 2 + D / (1 - b) ** 2),
            "stationary_grid": min(grid, key=lambda b: V / (2 / (1 - b) - 1) + D * (1 / (1 - b) - 1) ** 2),
            "finite_history_grid": min(grid, key=lambda b: finite_history_score(V, D, b)),
            "constant_target": nearest(1-(m*100**2/8)**(-1/3)),
            "endpoint_only": nearest(1 - 1 / constrained_memory(V, slope / 10000, grid)),
            "intercept_only": nearest(1 - 1 / constrained_memory(V, intercept, grid)),
        }
        losses = {float(b): value for b, value in case["seed1_grid_losses"].items()}
        oracle = min(losses.values())
        gap = lambda b: 100 * (losses[b] / oracle - 1)
        for name, b in variants.items():
            diagnostics[name].append(gap(b))
        paired = case["common_seed_losses"]
        paired_fixed = float(np.mean([item["fixed"] for item in paired]))
        paired_rule = float(np.mean([item["selected"] for item in paired]))
        assert beta == case["expected_beta"], (case["workload"], beta)
        assert variants["finite_history_grid"] == beta, (case["workload"], variants)
        rows.append(dict(workload=case["workload"], beta=beta, V=V, D=D,
                         rule_loss=losses[beta], fixed_loss=losses[fixed],
                         oracle_loss=oracle, rule_gap=gap(beta), fixed_gap=gap(fixed),
                         H=H, continuous_beta=1-1/H, pooling_pct=100 * pooling / eta,
                         finite_lag_pct=100 * slope / 10000 / D,
                         common_seed_loss_improvement_pct=100 * (paired_fixed - paired_rule) / paired_fixed))
    summary = {name: aggregate(values) for name, values in diagnostics.items()}
    reductions = {key: 100 * (summary["fixed"][key] - summary["rule"][key]) / summary["fixed"][key]
                  for key in ["mean", "maximum", "cvar25"]}
    with (BASE / "reconstructed_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(summary=summary, gap_reduction_pct=reductions,
                  fixed_grid_summary={str(b):aggregate([100*(float(c['seed1_grid_losses'][str(b)])/
                      min(c['seed1_grid_losses'].values())-1) for c in inputs['cases']]) for b in grid},
                  common_seed_mean_loss_improvement_pct=float(np.mean([r["common_seed_loss_improvement_pct"] for r in rows])),
                  improved=sum(r["rule_loss"] < r["fixed_loss"] for r in rows),
                  tied=sum(r["rule_loss"] == r["fixed_loss"] for r in rows),
                  worsened=sum(r["rule_loss"] > r["fixed_loss"] for r in rows))
    (BASE / "reconstructed_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

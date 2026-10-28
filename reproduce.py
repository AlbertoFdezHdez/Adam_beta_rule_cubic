"""Reconstruct the ESANN manuscript tables on a CPU from compact saved data.

Run: python reproduce.py --check
The selector reads only gradient-direction sketches. Candidate losses are used
after all decisions have been made, exclusively to evaluate those decisions.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def estimate_memory(pooled, groups, lags, earlier_pooled, probes, grid):
    """Select memory from one checkpoint and its earlier direction sketches.

    pooled: p-vector from all probes at the decision checkpoint.
    groups: G-by-p array from disjoint equal groups at the same checkpoint.
    earlier_pooled: K-by-p array, ordered like the K positive distinct lags.
    D_fit and C_fit are signed ordinary least-squares coefficients.
    """
    pooled, groups = np.asarray(pooled, float), np.asarray(groups, float)
    earlier_pooled, lags = np.asarray(earlier_pooled, float), np.asarray(lags, float)
    grid = sorted(float(b) for b in grid)
    if pooled.ndim != 1 or groups.ndim != 2 or groups.shape[1] != len(pooled):
        raise ValueError('Expected a pooled p-vector and G-by-p group matrix.')
    G = groups.shape[0]
    if G < 2 or probes % G or probes < G:
        raise ValueError('At least two equal groups must partition the probes.')
    if len(lags) < 2 or np.any(lags <= 0) or len(set(lags)) != len(lags):
        raise ValueError('At least two distinct positive lags are required.')
    if earlier_pooled.shape != (len(lags), len(pooled)):
        raise ValueError('Earlier directions must have shape K-by-p.')
    if not grid or min(grid) < 0 or max(grid) >= 1:
        raise ValueError('Candidate memories must lie in [0,1).')
    if not all(np.isfinite(a).all() for a in [pooled, groups, earlier_pooled]):
        raise ValueError('All direction sketches must be finite.')

    V = float(probes / (G*(G-1)) * np.sum((groups-pooled)**2))
    y = np.sum((earlier_pooled-pooled)**2, axis=1) / lags**2
    design = np.column_stack([np.ones(len(lags)), lags**-2])
    D_fit, C_fit = map(float, np.linalg.lstsq(design, y, rcond=None)[0])
    L = float(max(lags))
    D_L = D_fit + C_fit / L**2
    if D_L <= 0:
        beta = max(grid)
        H = 1/(1-beta)
        beta_continuous = None
    else:
        H = max(1., (V/(4*D_L))**(1/3))
        beta_continuous = 1-1/H
        beta = min(grid, key=lambda b:(abs(b-beta_continuous), b))
    return dict(V=V, D_fit=D_fit, C_fit=C_fit, D_L=D_L, H=H,
                beta_continuous=beta_continuous, beta=beta)


def aggregate(gaps):
    values = sorted(float(x) for x in gaps)
    return dict(mean=float(np.mean(values)), maximum=max(values),
                cvar25=float(np.mean(values[-math.ceil(len(values)/4):])))


def verify_manifest():
    manifest = json.loads((ROOT/'SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum verification failed: {name}')


def check_expected(actual, expected, location='results'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'{location}: dictionary keys differ')
        for key, value in expected.items():
            check_expected(actual[key], value, f'{location}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{location}: list lengths differ')
        for i, (a, b) in enumerate(zip(actual, expected)):
            check_expected(a, b, f'{location}[{i}]')
    elif isinstance(expected, (float, int)):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f'{location}: {actual!r} != {expected!r}')
    elif actual != expected:
        raise AssertionError(f'{location}: {actual!r} != {expected!r}')


def reconstruct():
    inputs = json.loads((ROOT/'data/inputs.json').read_text(encoding='utf-8'))
    grid, protocol = inputs['grid'], inputs['protocol']
    protocols = [('main', 16, 200), ('timing_150', 16, 150),
                 ('timing_250', 16, 250), ('probes_8', 8, 200)]
    decisions = {name:[] for name, _, _ in protocols}
    for case in inputs['cases']:
        for name, m, t in protocols:
            path = ROOT/'data/probes'/f"{case['workload']}_mb{m}.npz"
            with np.load(path, allow_pickle=False) as data:
                pos = {int(s):i for i,s in enumerate(data['steps'])}
                gpos = {int(s):i for i,s in enumerate(data['group_steps'])}
                lags = protocol['lags']
                result = estimate_memory(data['proj_r_full'][pos[t]],
                    data['proj_r_group'][gpos[t]], lags,
                    data['proj_r_full'][[pos[t-n] for n in lags]], m, grid)
            decisions[name].append(dict(workload=case['workload'], network=case['network'],
                                        dataset=case['dataset'], **result))

    # Only now evaluate the frozen decisions against the saved candidate losses.
    cases = {c['workload']:c for c in inputs['cases']}
    for rows in decisions.values():
        for row in rows:
            case = cases[row['workload']]
            losses = {float(b):float(v) for b,v in case['validation_losses'].items()}
            oracle = min(losses.values())
            row.update(loss=losses[row['beta']], oracle_loss=oracle,
                       gap=100*(losses[row['beta']]/oracle-1))
    fixed = {str(b):aggregate([100*(c['validation_losses'][str(b)] /
                    min(c['validation_losses'].values())-1) for c in inputs['cases']]) for b in grid}
    main = aggregate([r['gap'] for r in decisions['main']])
    reductions = {b:{k:100*(1-main[k]/fixed[b][k]) for k in main}
                  for b in ['0.94377','0.96838']}
    summary = dict(rule=main, fixed=fixed, relative_gap_reductions_percent=reductions,
        sensitivity={name:aggregate([r['gap'] for r in rows]) for name,rows in decisions.items()},
        gradient_cost=dict(pilot_updates=200, probe_evaluations=64, total_extra=264,
                           percent_for_10000_updates=2.64))
    return dict(per_task=decisions, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Also compare every result with the reference JSON.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    verify_manifest()
    results = reconstruct()
    if args.check:
        expected = json.loads((ROOT/'expected_results.json').read_text(encoding='utf-8'))
        check_expected(results, expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'reconstructed_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for name, rows in results['per_task'].items():
        with (args.output_dir/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Protocol                 Mean gap (%)   Maximum (%)   CVaR25 (%)')
    for name, values in results['summary']['sensitivity'].items():
        print(f"{name:24s} {values['mean']:12.4f} {values['maximum']:13.4f} {values['cvar25']:12.4f}")
    if args.check:
        print('All checksums, decisions and numerical reference results match.')
    print(f'Results written to {args.output_dir}')


if __name__ == '__main__':
    main()

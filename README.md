# Early Memory Selection for Balanced Adam

Code and compact data accompanying the six-page **ESANN 2027 manuscript** by
Alberto Fernández-Hernández, Cristian Pérez-Corral, Jose I. Mestre,
Manuel F. Dolz, and Enrique S. Quintana-Ortí.

A local bias–variance tracking model yields a rule for choosing one fixed shared
Adam memory, `beta1 = beta2`, from early gradient measurements. This package
reconstructs the reported selections and comparisons on eleven vision and
language tasks using saved measurements, on a CPU, without retraining networks.

## Reproduce the article

Python 3.10 or later and NumPy are sufficient. Tested with Python 3.12 and NumPy 2.3.5.

```bash
python -m pip install -r requirements.txt
python reproduce.py --check
```

The command verifies file checksums, reconstructs all decisions, and checks the
results against `expected_results.json`. It writes per-task CSV files and a full
JSON report in `results/`. These generated files are ignored by Git.

## Rule and measurement protocol

The pilot uses shared `beta = 0.95` for 200 updates. At updates 100, 150, 180 and
200, each measurement uses 16 fresh effective-batch gradients in four groups.
For pooled direction `z` and group directions `z_j`, compute

```text
V = m / (G * (G - 1)) * sum_j ||z_j - z||^2
y_n = ||z_200 - z_(200-n)||^2 / n^2,    n in {20, 50, 100}
(D_fit, C_fit) = ordinary least-squares fit of y_n = D_fit + C_fit / n^2
D_L = D_fit + C_fit / 100^2
H = max(1, (V / (4 * D_L))^(1/3))       when D_L > 0
beta = 1 - 1 / H
```

The two fitted coefficients enter directly with their signs. For `D_L <= 0`,
the fitted score selects the largest allowed memory. Evaluation rounds the
continuous beta to the nearest candidate, choosing the smaller value in a tie:

```text
0.68377, 0.82217, 0.90000, 0.94377, 0.96838, 0.98222,
0.99000, 0.99438, 0.99684, 0.99822, 0.99900
```

The function `estimate_memory` can also be used with separately collected pooled
and group direction sketches. Selection reads gradient measurements only;
candidate validation losses enter afterwards to evaluate the frozen decisions.

## Results reproduced

Relative gap is `100 * (selected_loss - oracle_loss) / oracle_loss`, where the
oracle is the minimum saved loss over the candidate grid for that task. Losses
are minimum recorded validation losses from completed fixed-memory runs with
initialization seed 1. Each task has equal weight. CVaR25 averages the three
largest gaps among eleven tasks.

| Shared memory or rule | Mean gap (%) | Maximum gap (%) | CVaR25 (%) |
|---|---:|---:|---:|
| Fixed 0.90000 | 1.4292 | 3.8770 | 3.3543 |
| Fixed 0.94377 | 0.6236 | 2.4853 | 2.0409 |
| Best fixed 0.96838 | 0.5461 | 2.1862 | 1.4304 |
| **Rule: 16 probes, update 200** | **0.3431** | **1.0613** | **1.0391** |

Mean-gap reductions are **45.0%** against 0.94377 (the grid point nearest to
0.95) and **37.2%** against the benchmark's best fixed memory. These percentages
measure reductions in relative gap, rather than percentage reductions in loss.

The package also reproduces every sensitivity comparison reported in the article:

| Probes | Decision update | Mean gap (%) | Maximum gap (%) | CVaR25 (%) |
|---:|---:|---:|---:|---:|
| 16 | 150 | 0.3695 | 1.3339 | 1.1359 |
| **16** | **200** | **0.3431** | **1.0613** | **1.0391** |
| 16 | 250 | 0.7551 | 4.4034 | 2.2662 |
| 8 | 200 | 0.5535 | 2.3036 | 1.5662 |

## Files and data format

- `reproduce.py`: selection, evaluation, and verification in one program.
- `data/inputs.json`: common protocol, candidate losses, and task training budgets.
- `data/probes/`: 22 compact NumPy files, one per task and probe budget (8 or 16).
- `expected_results.json`: numerical reference for every reported decision and metric.
- `SHA256.json`: integrity hashes of the distributed files.

Each NumPy file contains `steps`, `proj_r_full`, `group_steps`, and `proj_r_group`.
The pooled sketches have shape `(len(steps), 1024)`; group sketches have shape
`(len(group_steps), 4, 1024)`. Both are float64. The explicit step arrays identify
the rows; all measurements needed for the reported comparisons are retained.

Each sketch sums signed coordinates into 1024 slots, with each coordinate divided
by the square root of its parameter tensor's size. Signs are +1 or -1 with equal
probability; slots are uniform; the mapping (seed 12345) is fixed across checkpoints.
Probes use the training effective batch size and gradient clipping. Pilot random
states and model buffers are restored after measurement.

The manuscript supplies architectures, datasets, and training recipes. This
repository evaluates saved runs: each chosen memory is used from initialization,
with the pilot discarded. The measurements were collected in a dense campaign;
these files retain the observations needed to reproduce the article's offline
results. Prospective sparse-only collection may follow a different random stream.

Calibration counts 200 pilot updates plus 64 extra gradient evaluations: a 2.64%
gradient-equivalent surplus for a subsequent 10000-update training. Wall-clock
cost additionally depends on data loading, transfers, sketching, and restart.

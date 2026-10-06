# Early Memory Selection for Balanced Adam

A cubic noise–lag rule selects one fixed shared memory parameter for Adam,
with beta1 = beta2, from early gradient measurements.

This repository contains the reproducibility package: saved gradient-direction
sketches, validation-loss summaries, and CPU programs for reconstructing the
selection results and sensitivity analyses across eleven vision and language
workloads. Reconstruction uses saved measurements and requires no model training.

## Quick start

Use Python 3.10 or later, then run:

```bash
python -m pip install -r requirements.txt
python reproduce_selection.py
python reproduce_robustness.py
python reproduce_dense_audit.py
python strength_numerical.py
python strength_trajectory.py --inputs validation_curves.json --output .
```

## Method and data

The main protocol uses a 200-update pilot with shared beta 0.95 and sixteen
fresh effective-batch gradients at each of four checkpoints. The memory is
computed as `(V / (4 * D)) ** (1 / 3)`, restricted to the candidate interval,
converted to beta, and rounded to the candidate grid.

- `probes/`: main measurement sketches.
- `robustness_probes/`: timing and probe-budget sensitivity inputs.
- `inputs.json`: candidate grid and validation-loss summaries.
- `validation_curves.json`: saved validation curves.
- `SHA256.json`: hashes of all original package files.
- `README.txt`: detailed protocol, input formats, interpretation, and cost.

The compact data reproduce the reported protocols at the original float64
precision. Arbitrary new measurement layouts requiring omitted dense vectors
need the full original archive. Training architectures and recipes are
documented in the accompanying manuscript; this package reconstructs results
from existing measurements rather than rerunning those trainings.

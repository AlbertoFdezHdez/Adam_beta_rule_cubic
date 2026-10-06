Early Memory Selection for Balanced Adam: offline reproducibility package

Protocol
The main rule measures 16 fresh effective-batch gradients in four equal groups
at pilot updates 100, 150, 180, and 200. The pilot has shared beta 0.95. Pooled
and group direction sketches determine the variability proxy; three temporal
lags determine the change proxy. Memory is (V/(4D))^(1/3), restricted to the
candidate interval before nearest-beta rounding. One selected shared beta is
evaluated using a full training from a fresh initialization.

Scope
The package reconstructs every selected beta, per-workload validation loss,
relative gap, aggregate metric, measurement decomposition, and common-seed
comparison. Additional sketches reproduce all fifteen decision times from
150 to 500 (spacing 25), and probe budgets 2, 4, 8, and 16 at update 200.
Execution uses the supplied measurements and Python with NumPy on a CPU.
The manuscript itself supplies model, dataset, and training recipes.

Requirements and commands
Python 3.10 or later and NumPy >=1.24 and <3.
Install: python -m pip install -r requirements.txt
Main tables: python reproduce_selection.py
Timing and budget tables: python reproduce_robustness.py
Dense timing and measurement-grid audits: python reproduce_dense_audit.py
Task tradeoffs and decision stability: python strength_numerical.py
Alternative summaries of the saved curves:
python strength_trajectory.py --inputs validation_curves.json --output .

Outputs
reconstructed_cases.csv: every selected beta, V, D, memory, loss, gap and share.
reconstructed_summary.json: aggregate gaps and direct common-seed comparisons.
It also reproduces every fixed-grid comparator and the constant-target,
endpoint-only, and intercept-only coefficient diagnostics from the appendix.
reconstructed_robustness.json: all audited timing decisions and budget outcomes.
strength_numerical.json: task-level attribution against both strong fixed
values, workload-removal sensitivity, timing frequencies, and arithmetic
coefficient-ratio margins. The same program produces the additional tables.
The selection reconstruction also verifies agreement of the exact
finite-history grid score at history length 200 with every main decision.
validation_curves.json contains 155 anonymous validation curves, including
all 121 seed-1 grid candidates and the available common-seed pairs.
strength_trajectory.json and its table reproduce minimum, final, and mean
of the last three validation losses, including every fixed comparator.
Expected main mean gaps: fixed 0.623566500749381%, rule 0.36946903173513074%.
Relative mean-gap reduction: 40.74905703062698%.
Main maximum gap: 1.333888408480921%; main CVaR25: 1.1358507576355887%.
Common-training-seed mean direct loss improvement: -0.10493233770267536%.
These metrics describe complementary aspects of selection utility. The pilot
chooses each configuration once; training seeds repeat that fixed configuration.

Inputs
inputs.json supplies the grid, baseline, candidate losses with matched seed 1,
and common-seed fixed/selected losses. Loss is the minimum recorded validation
loss in each full run. The finite-grid best loss is only an evaluation reference.
Early gradient sketches supply the coefficient calculation and beta selection.

probes/ supplies the four main observations as arrays steps, proj_r_full,
and proj_r_group. Each direction sketch has 1024 coordinates. There are four
group directions, each normalizing four samples; the pooled direction uses all
sixteen. The same fixed tensor-balanced signed-bucket projection is used.
robustness_probes/ adds checkpoints needed for the timing/budget audits.
The compact mb16 files retain raw pooled sketches at the representative
timing checkpoints, and group_steps labels the retained group sketches.
dense_steps and dense_V retain the original variability measurements.
secants stores the exact float64 squared secants at updates 100..500
for lags 20,50,100. mesh_secants stores all lags 20..100 at update 200.
These are computed with exactly the original vector operations, without
rounding or changing precision; the dense audit consumes them directly.
Retained group sketches independently verify the variability scalars,
and raw pooled sketches verify all available squared secants. The
compact archive reproduces every reported protocol; the full dense
vectors are omitted, so arbitrary new measurement layouts require
the original full archive.
It reconstructs all 401 decision times from 100 to 500 and the seven regular
measurement grids with shortest/longest lags held at 20/100. Coefficients and
choices precede candidate-loss evaluation. It reproduces 339/351 mean-gap wins
and 351/351 CVaR wins against .94377 over updates 150..500, and 288/351 mean-gap
wins against the hindsight-selected fixed .96838. These are within-pilot
sensitivity frequencies, not independent training trials or confidence levels.
Budgets 2,4,8,16 have groups 2,4,4,4, respectively. Their variability multiplier
is budget/(groups-1) for every examined budget.

timing_audit.json and probe_budget_audit.json retain the wider audit records.
reproduce_robustness.py directly recomputes the manuscript's timing and budget
claims from the supplied sketches and checks agreement with the audit.
The coefficient convention and rounding are common to all workloads.
Sensitivity comparisons are exploratory analyses of the same saved benchmark
suite, exposing how its decisions respond to the observation protocol.
SHA256.json identifies every distributed file.

Cost and exact replay
The sketches were extracted from a dense collection campaign. A prospective
four-checkpoint collection needs 64 extra effective-batch gradients. Its total
gradient-equivalent surplus, including a discarded 200-update pilot, is
264/T, or 2.64% for a subsequent 10000-update training. This is an operation
count for compact deployment. Wall-clock accounting additionally includes
data loading, transfers, sketch construction, and restart overhead.
Sparse-only sampling draws different examples unless it reproduces the dense
probe stream advancement. The supplied sketches reproduce the reported
offline decisions exactly, separately from exact raw-stream regeneration.

All primary candidates use seed 1. Paired repeated training uses seeds 1--3
where present, and seed 1 for Food-101. The distributed assets are anonymized
gradient-direction sketches, validation-loss summaries, and CPU reconstruction
programs.

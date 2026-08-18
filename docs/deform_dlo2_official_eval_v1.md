# DEFORM DLO2 Official Evaluation V1

Every loaded checkpoint is reconstructed with
`official-deform-dlo-initialization-v1`; the official evaluator refuses a parent
protocol with a different initialization contract. See
`docs/deform_dlo2_initialization_amendment_v1.md`.

This stage is the final, one-shot evaluation of a method selected without using
the official DEFORM evaluation partition. It can run only after the fresh DLO2
source reproduction, checkpoint-posterior transfer gate, and all-56 training
refit have all completed successfully.

## Fixed comparison

The candidate is the exact posterior operator, checkpoint weights, variance
floor, and validation-fitted variance scale selected on fresh DLO2 source data.
The comparison baseline is the source-validation-selected single checkpoint
from the same all-56 refit. Action-aware persistence is reported as a context
baseline. This isolates the posterior contribution while keeping training data,
optimization schedule, and evaluation inputs identical.

All 14 DLO2 evaluation trajectories are hashed, loaded in sorted order, and
evaluated once. The stage performs no target checkpoint selection, posterior
weight fitting, uncertainty calibration, case replacement, or retry. A failure
after the evaluation partition is opened is sealed as a failed one-shot run and
does not authorize a replacement run.

## Claim boundary

The primary benchmark metric is mean coordinate-wise L1 error in metres. A
positive result requires all three frozen gates:

1. candidate mean strictly below the published DEFORM DLO2 value of 9.7 mm;
2. at least 1% improvement over the identical-training single checkpoint;
3. candidate wins on at least 8 of 14 trajectories.

Early, middle, and late errors are descriptive diagnostics. Coordinate-marginal
coverage, interval width, Gaussian NLL, and coordinate NEES reuse the source
validation variance scale unchanged. They are calibration diagnostics, not
simultaneous trajectory-level coverage guarantees.

## Execution

```bash
python scripts/remote/run_deform_dlo2_official.py \
  --protocol configs/sota/deform_dlo2_official_eval_v1.json \
  --alltrain-protocol configs/sota/deform_dlo2_alltrain_refit_v1.json \
  --source-protocol configs/sota/deform_dlo2_fresh_v1.json \
  --alltrain-result /path/to/alltrain_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /new/empty/official-output \
  --device cuda:0
```

The command is intentionally unavailable until the upstream source gates have
produced the exact required artifacts.

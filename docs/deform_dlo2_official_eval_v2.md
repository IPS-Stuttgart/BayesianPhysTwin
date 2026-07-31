# DEFORM DLO2 one-shot official evaluation v2

## Target-free correction

Version 2 supersedes the unopened v1 evaluation protocol. The DEFORM paper
reports average L1 loss over a 500-step horizon, so its DLO2 value of 0.97 in
units of `10^-2 m` is 9.7 mm mean coordinate-wise L1. That metric agrees with
the Bayesian-PhysTwin evaluator.

The released upstream loader does not, however, evaluate every file exactly
once. With Python seed 0, it first draws 56 training paths with replacement and
then draws 14 evaluation paths with replacement. Applied to a canonical sorted
14-file list, the evaluation indices are:

```text
1, 7, 9, 7, 11, 7, 13, 8, 8, 6, 8, 5, 8, 4
```

Only nine indices are unique. The upstream script obtains its path population
from `glob.glob` without defining an order, so this canonical draw is a
source-code compatibility view rather than a claim to reconstruct the unknown
filesystem order of the paper run.

This audit used only the public paper and source at the already locked upstream
commit. No DLO2 training trajectory, source outcome, filename, evaluation
filename, or evaluation value was accessed.
The machine-readable record is
`results/sota/deform_dlo2_official_eval_v2/reference_operator_audit.json`.

## Frozen reporting operators

The one-shot run evaluates all 14 sorted evaluation files exactly once and
reports two aggregates:

1. `all_unique`: the complete 14-trajectory mean and paired Bayesian-versus-
   single-checkpoint comparison;
2. `published_reference_compatibility`: the fixed with-replacement index draw
   above, applied to the same sealed per-case records.

The 9.7 mm published-reference gate must pass under both aggregates. The
Bayesian method must additionally improve by at least one percent over its
identically trained single checkpoint and win at least 8 of 14 unique
trajectories. This makes the final claim more conservative than either using
the duplicated draw alone or comparing the all-unique scalar to the paper
number without qualification.

The candidate operator is selected before this target opens from three fixed
point functionals: parameter mean, posterior predictive mean, and coordinate-
wise posterior predictive median. The median is included because coordinate-
wise L1 is the benchmark loss for which it is Bayes optimal. It changes only
the reported point estimate; posterior members and source-calibrated
uncertainty remain unchanged.

Model selection, checkpoint weights, variance calibration, and every case are
unchanged. There is no target selection, retry, replacement, or calibration.
The old v1 protocol remains in version control as superseded provenance and
must not be used to open the official evaluation partition.

## Execution

```bash
python scripts/remote/run_deform_dlo2_official.py \
  --protocol configs/sota/deform_dlo2_official_eval_v2.json \
  --alltrain-protocol configs/sota/deform_dlo2_alltrain_refit_v2.json \
  --source-protocol configs/sota/deform_dlo2_fresh_v2.json \
  --alltrain-result /path/to/alltrain_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /new/empty/official-output \
  --device cuda:0
```

The runner remains unavailable until every upstream source, posterior, and
all-training-data gate produces its exact required artifact.

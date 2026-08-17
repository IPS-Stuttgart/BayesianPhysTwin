# DEFORM DLO checkpoint-belief exploratory study

This study asks whether a validation-selected average of checkpoints from the
frozen DEFORM DLO1 reproduction improves its point estimate before any official
evaluation is opened.

## Information boundary

- The arm bank and thresholds are fixed in
  `configs/sota/deform_dlo_checkpoint_belief_exploratory_v1.json`.
- Only the eight DLO1 validation trajectories select an arm.
- The DLO1 source-test split is already opened by the source-reproduction gate.
  Its checkpoint-belief result is therefore exploratory and cannot confirm
  transfer or state of the art.
- Only the selected validation arm is evaluated on that source-test split.
- The official DEFORM evaluation tree remains protected by the source runner's
  audit hook and is not read.
- A failed one-percent validation gate returns the exact selected single
  checkpoint, rather than an approximately equivalent reconstruction.

The registered arms average two or more sequential checkpoint state
dictionaries. Floating tensors are accumulated in double precision and restored
to their original dtype. Discrete state must agree exactly across checkpoints;
otherwise the arm is invalid.

## Advancement rule

Checkpoint belief advances only if:

1. the frozen DEFORM source reproduction first passes its parity and persistence
   gates;
2. one registered averaged arm improves validation mean coordinate L1 by at
   least one percent over the selected single checkpoint; and
3. that already-open DLO1 source test improves by at least one percent with at
   least five of eight paired wins.

Passing those gates authorizes only a fresh DLO2 reproduction using the same arm
bank and thresholds. DLO1 tuning may not be carried into that confirmation, and
no state-of-the-art claim is permitted before an identical-information official
evaluation.

## Command

```bash
python scripts/remote/run_deform_dlo_checkpoint_belief.py \
  --protocol configs/sota/deform_dlo_checkpoint_belief_exploratory_v1.json \
  --source-protocol configs/sota/deform_dlo_source_v1.json \
  --source-result /path/to/source_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/checkpoint-belief-output \
  --device cuda:0
```

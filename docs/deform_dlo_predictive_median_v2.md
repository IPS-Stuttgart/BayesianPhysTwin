# DEFORM checkpoint posterior v2

## Motivation

DEFORM's published DLO benchmark uses mean coordinate-wise L1 error. The
Bayes-optimal point estimate for L1 loss is a median, whereas the frozen v1
checkpoint posterior tested only parameter and posterior predictive means.
Version 2 adds one preregistered operator:

```text
predictive_median = coordinate-wise weighted median of checkpoint rollouts
```

For a cumulative posterior weight exactly equal to one half, the operator uses
the midpoint of the adjacent values. Consequently, two equally weighted
members return their midpoint, while odd uniform banks return the ordinary
coordinate-wise median.

This is a point-functional correction, not a new posterior or a target-tuned
ensemble. Checkpoint members, validation-derived weights, the variance around
the posterior mean, the variance floor, and validation-only scale calibration
remain unchanged.

Like a predictive mean, a coordinate-wise median need not itself be one exact
simulator rollout. It is reported only as the decision-theoretic L1 point
summary; physical feasibility remains represented by the member rollouts, and
uncertainty remains the source-calibrated checkpoint posterior.

## Locked chain

The executable chain is:

1. `deform_dlo_longrun_posterior_v2.json` selects among parameter mean,
   predictive mean, and predictive median using DLO1 validation only. It must
   improve validation L1 by at least 1%, then transfer by at least 1% with five
   of eight wins on the already-open exploratory DLO1 source split.
2. `deform_dlo2_fresh_v2.json` repeats the same bank from scratch on unopened
   DLO2 training data. Its wrapper requires the checksummed successful DLO1
   posterior result and selection seal before opening DLO2. Exact selected-
   single fallback is retained.
3. `deform_dlo2_alltrain_refit_v2.json` copies the selected operator and weights
   without reselection and retrains on all 56 DLO2 training trajectories.
4. `deform_dlo2_official_eval_v2.json` opens all 14 evaluation trajectories
   once only after every preceding gate passes.

The committed v1 configurations remain provenance but are rejected by the v2
posterior validators. No DLO2 source or evaluation value was inspected while
adding this operator.

## Claim boundary

Passing DLO1 is exploratory evidence only. Passing fresh DLO2 source gates
authorizes the one-shot evaluation but is not itself a state-of-the-art claim.
The final claim requires beating the published 9.7 mm reference under both the
all-unique and canonical compatibility aggregates, plus at least 1% gain and
8/14 wins against the identically trained selected single checkpoint.

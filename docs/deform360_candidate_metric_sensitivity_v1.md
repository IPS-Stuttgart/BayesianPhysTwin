# Deform360 candidate-metric sensitivity v1

## Purpose

This source-only audit asks whether the already-open Deform360-27 development
gain of the frozen pairwise-consensus online belief update depends on one
particular metric convention.

It does not reproduce the official Deform360 evaluator. The exact public
split, future-frame policy, coordinate convention, units, masking, and
aggregation contract remain incomplete. Every output is therefore labelled
`candidate_convention_sensitivity_only`.

## Fixed evidence

The evaluator consumes:

- the checksum-bound 27-case output of
  `deform360-open27-raw-alltracker-pairwise-gate-v1-development`;
- the matching already-open independent-source outcomes;
- the physical prior, persistence control, frozen pairwise-consensus RBF arm,
  and the earlier support-gated diagnostic arm.

For each case it verifies the saved report and archive hashes, validates the
source prediction/outcome chain, and requires the physical-prior and
persistence trajectories to be byte-identical across both artifacts.

## Population and metrics

All assimilation-centre identities are permanently excluded. A point is
scored only when the source visibility and validity masks are true and both
prediction and target are finite. The population and future frames remain
fixed while the audit reports:

- coordinate MSE and RMSE;
- mean and root-mean-square Euclidean identity error;
- one-sided, reverse, and symmetric Chamfer;
- Euclidean and squared nearest-neighbour reductions;
- frame-pooled, episode-balanced, and physical-object-balanced aggregation.

Every metric carries an explicit `m` or `m^2` unit.

## Prospective decision rule

The metric-robustness gate passes only when the unchanged pairwise arm
improves over both the physical prior and persistence for all of:

1. mean Euclidean identity error;
2. one-sided prediction-to-target Euclidean Chamfer;
3. symmetric Euclidean Chamfer;

under all three aggregation conventions.

Passing this gate justifies an independently preregistered fresh-object run.
It does not authorize tuning, relabelling the open result as confirmation, or
claiming official Deform360 parity.

## Reproduction

```bash
bpt-audit-deform360-candidate-metrics \
  /path/to/deform360-raw-alltracker-pairwise-gate-v2-development \
  /path/to/deform360-dense-reusable-panel-v1/independent-source-v1 \
  result.json
```

The result JSON includes all input hashes, per-case support, per-case and
per-object means, aggregation sensitivity, paired object-cluster bootstrap
intervals, and the exact next-step decision.

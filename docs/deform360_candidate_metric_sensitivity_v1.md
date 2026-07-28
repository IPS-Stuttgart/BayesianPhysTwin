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

## Open-source result

The exact evaluator commit
`e2f8d827bfd60df79eeffee511a5df7e2d53ea21` passes the metric-robustness
gate on 27 already-open episodes from five physical objects. Lower is better;
values are physical-object-balanced means in millimetres.

| Arm | Identity mean Euclidean | One-sided Chamfer | Symmetric Chamfer |
|---|---:|---:|---:|
| Physical prior | 16.160 | 8.873 | 8.999 |
| Persistence | 14.533 | 7.889 | 8.190 |
| Frozen pairwise update | **11.542** | **6.588** | **6.795** |

Relative to the physical prior, the pairwise update improves these metrics by
28.58%, 25.76%, and 24.49%. Relative to persistence, it improves them by
20.58%, 16.50%, and 17.02%. All five object means improve for every headline
metric. The corresponding physical-object cluster intervals for differences
against persistence are `[-4.566, -1.406]`, `[-2.097, -0.523]`, and
`[-2.201, -0.587]` mm.

The generated report is
`results/sota/deform360_candidate_metric_sensitivity_v1/source_audit.json`.
Its file SHA-256 is
`1dd3b525daa22e925569ea056f0893f9205e98549e31f532c30e8a5658c86eab`;
its canonical internal result digest is
`476fb48f8ba228340676b7b1e6440b9d828b20e8d906a1f8a479a67de43f5245`.

This is stronger evidence that the open-development gain is not a metric
artifact. It remains weaker than an official comparison because the evaluated
population and benchmark contract do not match an author-confirmed Deform360
table protocol.

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

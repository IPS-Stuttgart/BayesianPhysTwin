# Source-only discrepancy candidate tournament v1

## Purpose

BayesianPhysTwin contains several increasingly expressive discrepancy-belief
families. Implementing a family and passing numerical tests does not establish
that its additional state, covariance, or runtime improves a physical query.
This tournament selects at most one candidate on already opened source groups
before a separate prospective experiment is frozen.

The principal reference is a registered simple candidate such as held last
residual. The physical predictor remains a distinct fallback candidate. Every
candidate must be evaluated on the same units, horizons, physical fallback,
proper score, interval policy, and target-blind prediction barrier.

The evaluator is candidate-agnostic. A dynamic endpoint model, structured
low-rank field, graph-modal dynamic field, or later method may participate by
emitting the common JSON contract. The selector does not import or privilege a
particular candidate implementation.

## Information boundary

The input must declare all of the following:

- `split` is exactly `source-only`;
- candidate predictions were sealed before scoring;
- candidate generation did not use the scored outcomes;
- future observations were not used;
- confirmation payloads remain closed; and
- no candidate or statistical group may be replaced after scoring.

The statistical group must be a physical object or independent acquisition
session, not a frame, point, track, camera, or taxel. All point losses, proper
scores, coverage values, and widths are averaged within a group before groups
receive equal weight.

## Matched records

For every scored unit, the artifact must contain one record for every registered
candidate. Records bind:

- candidate, unit, group, and horizon identities;
- raw point loss and raw proper score;
- the common physical-fallback point loss and proper score;
- acceptance and deployed values;
- optional interval coverage and complete interval width.

A rejected candidate must deploy the exact physical-fallback values. The
registered physical-fallback candidate must itself be rejected and reproduce its
raw fallback values exactly. Candidate-dependent fallback scores, incomplete
candidate rosters, mixed interval availability, duplicate units, or altered
horizons fail closed.

Candidate metadata binds the exact source revision, configuration digest, and
prediction-artifact digest in addition to family, state dimension, parameter
count, measured runtime, and retained covariance bytes. The input also binds one
common evaluator revision, scoring-policy digest, scored-unit roster, physical
fallback artifact, prediction barrier, point-loss definition, proper-score
definition, and interval semantics. Complexity quantities are tie-break metadata,
not accuracy evidence.

## Selection rule

A challenger is eligible only when all frozen gates pass:

1. minimum equal-group point-loss improvement over the reference;
2. bounded worst-group relative regression;
3. bounded harmful accepted-update count relative to physical fallback;
4. non-regressing equal-group proper score;
5. optional nonpositive upper endpoint of a paired group-bootstrap interval; and
6. optional equal-group interval coverage within the registered shortfall.

The reference and physical fallback remain registered baselines and cannot win as
challengers. Eligible challengers are ordered by:

1. lower equal-group mean proper score;
2. lower equal-group mean point loss;
3. lower equal-group mean interval width when intervals are registered; and
4. lower state dimension, parameter count, runtime, covariance bytes, then ID.

The complete selection is repeated while holding out each independent group. A
positive source gate requires held-group non-regression and, when configured,
the same provisional winner in every fold. If either cross-fitted condition
fails, the final selected candidate is the registered reference even when a
challenger won the complete source set. The provisional full-source winner is
retained separately for diagnosis. If no challenger satisfies the source rule,
the report likewise retains the registered reference. Either outcome is a valid
source result.

## Command

```bash
bpt diagnostic run select-discrepancy-candidate \
  source-tournament.json \
  source-tournament-report.json
```

The command returns exit code `0` when one challenger advances and `3` when the
valid result retains the reference. Input reading rejects duplicate JSON keys,
non-finite constants, changing files, non-ordinary files, and oversized files.
Publication is atomic and does not replace an existing report unless
`--overwrite` is explicit.

The report contains content identities for the input evidence and report,
per-candidate gate diagnostics, paired intervals, the complete leave-one-group-out
trace, the provisional full-source winner, the final retained or advanced
candidate, and the decision. `claim_authorized` is always false.

## Relationship to candidate branches

Candidate branches should export records through a separate frozen runner rather
than making the tournament import branch-specific APIs. In particular, dynamic,
structured, and graph-modal discrepancy implementations can be compared from
immutable prediction artifacts even while their implementation pull requests
remain independent.

A selected candidate still requires a new protocol that freezes its exact code,
basis or component family, process treatment, guard, calibration groups, target
seal, and downstream physical-query evaluator before any unopened target is
scored.

## Scientific boundary

A passing tournament is source-only model selection. It does not establish raw
covariance calibration, fresh-object or fresh-session transfer, Causal4D
intervention benefit, deployment safety, or state of the art. A failed tournament
must not be rescued by tuning on the same scored source outcomes, and neither
outcome authorizes confirmation access.

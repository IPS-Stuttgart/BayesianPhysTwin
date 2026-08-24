# Cross-action placebo-separation certificate v1

## Scientific question

A source-derived correction may beat same-action continuation without representing
physical state or material parameters. The stronger diagnostic asks whether one
frozen physical-transport candidate also beats controls that retain comparable
residual statistics but deliberately destroy the hypothesized transport
mechanism.

This certificate supplements, but does not replace, the prospective
`CrossActionTransportResultV1` study added by BayesianPhysTwin PR #756.

## Registered controls

The protocol can require any nonempty subset of four controls:

| Control | Construction | Failure localized |
| --- | --- | --- |
| `wrong_action` | Propagate the source-inferred candidate under a preregistered incorrect action identity. | The result depends only weakly on the commanded physical intervention. |
| `wrong_object` | Apply a source candidate from a matched but different physical object/session. | The correction is generic persistence rather than object-specific physical inference. |
| `phase_shifted` | Apply a frozen nonzero temporal shift to the source action or prefix. | The result is insensitive to action-response timing. |
| `identity_permuted` | Permute registered point or graph identities while preserving marginal residual statistics. | The result does not require the claimed material or graph correspondence. |

The exact physical candidate, donor assignment, incorrect-action mapping, phase
shift, identity permutation, matching variables, and random seeds are each bound
by an immutable construction-artifact identity before target access. A control is
invalid when target outcomes select any of these choices.

## Prediction and score separation

`SealedCrossActionPlaceboPredictionV1` is published before target access. Every
prediction binds:

- the placebo protocol and complete object/session/action identity;
- one exact parent `SealedTransportPredictionV1` identity from the primary
  cross-action experiment;
- the registered arm-construction artifact;
- the prediction artifact, one complete prediction-batch identity, and source
  commit;
- the parent physical selection or exact-fallback disposition; and
- explicit target-closed attestations.

`CrossActionPlaceboScoreRowV1` is produced only after authorized target access. It
binds one sealed prediction, one target outcome, one target-access attestation,
one frozen scorer, and the proper score. The protocol content-binds the
`lower_is_better` orientation used by the contrast; other orientations are
rejected rather than silently reversing the decision. Target-side method or
threshold selection is rejected.

## Statistical unit and decision

Every registered arm is scored on every frozen off-diagonal action pair. Scores
are averaged across action pairs *within* each physical object/session. Only the
resulting object/session means enter the paired bootstrap.

For placebo `p`, with a lower-is-better proper score, the session contrast is

```text
C[s,p] = score[s,p] - score[s,physical].
```

Positive values favor physical transport. The conjunctive claim passes only when:

1. the minimum number of independent sessions is present;
2. at least one parent physical candidate was selected rather than every case
   returning exact fallback; and
3. the lower paired-bootstrap confidence limit exceeds the frozen contrast
   margin for **every** registered placebo.

A tie or failure against one placebo returns
`physical_transport_placebo_separation_not_supported`. The result also reports
how often each placebo equals or beats the physical arm, with a Wilson interval;
zero observed placebo wins is not reported as zero population risk.

## Fail-closed invariants

The implementation rejects a report unless:

- every target session contains the complete off-diagonal action-pair and arm
  matrix;
- every arm uses its exact protocol-bound construction identity;
- all arms for one action pair bind the same parent transport prediction,
  outcome, and parent selection/fallback disposition;
- all predictions belong to one sealed batch;
- every score uses one target-access attestation and one scorer;
- an exact-fallback pair selects one byte-identical prediction artifact and one
  identical proper score for all controls; and
- no prediction or score indicates target-informed selection.

## Claim boundary

A positive result supports bounded separation from the exact registered placebo
controls on the frozen cohort. It does not establish a unique data-generating
cause, arbitrary-action or arbitrary-object generalization, real-data uncertainty
calibration, deployment safety, Prob4D provider competence, Causal4D intervention
benefit, or state of the art.

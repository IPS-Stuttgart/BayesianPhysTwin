# Deform360 Adaptive Causal Direct Depth V14 Result

## Decision

The prospective V14 source gate **failed before source-outcome
authorization**. Close V14 without tuning its carrier, event detector,
association, covariance, update, fallback, or gate thresholds.

All twelve source cases produced a valid prediction disposition, but only one
case admitted a causal-response event, below the frozen minimum of six. No
candidate update was applied. Because the source gate is conjunctive, future
source identities and metrics cannot change this decision and remain
unopened.

This is an outcome-blind fresh-source admission result. It is not an accuracy,
calibration, confirmation, or state-of-the-art result.

## Registered Result

| Quantity | Result |
| --- | ---: |
| Locked fresh source objects | 12 |
| Sealed predictions or exact fallbacks | 12/12 |
| Technical failures after sealing | 0 |
| Event-admitted objects | 1/12 |
| Frozen minimum event admissions | 6/12 |
| Applied candidate updates | 0/12 |
| Exact baseline fallbacks | 12/12 |
| Persistence backbones | 9/12 |
| Physical backbones | 3/12 |
| Source outcomes opened | 0 |

The prediction-completeness gate passed. The target-free event-admission gate
failed. Outcome-dependent accuracy, safety, regret, horizon, NEES, coverage,
and interval-width gates were therefore not evaluated.

## Evidence Order

The complete 12-case prediction set was sealed before the source decision was
created. The finalizer then:

1. validated the source lock and mixed admission/physical custody;
2. validated every prediction report, archive, disposition checksum, and
   bit-exact fallback;
3. verified the frame-57 causal cutoff and all information-boundary flags;
4. counted event admissions and candidate applications; and
5. applied the already frozen source-gate cardinalities.

The resulting artifact records
`source_outcome_authorized=false` and
`close_v14_without_source_outcome_reveal`. No future object observation,
future identity, source metric, target artifact, or held-v8 artifact or
process was accessed.

## Technical Amendment

The first rank-3 smoke exposed a shape-contract defect after its permitted
prefix had been sealed: one or two supported entities could not be represented
with the requested three spatial groups, so the admission dataclass raised
instead of returning the registered insufficient-support rejection.

The amendment changes only that failure mode. Sparse support now receives up
to the number of available groups and proceeds to the unchanged support gate,
which rejects it. Adequately supported cases, thresholds, numerical updates,
and advancement gates are unchanged. Rank 3 then sealed an exact persistence
fallback, and the remaining eleven cases completed normally.

## Interpretation

V14 solved several engineering problems:

- outcome-blind fresh-object admission;
- strict source and backend preflight;
- mixed original/reserve custody;
- causal use of tactile, measured actuator, and RGB-D prefix evidence;
- disjoint proposal and validation camera panels;
- correlation-aware covariance with a shared-camera-bias nuisance; and
- exact fallback on every rejected update.

It did not produce a usable observation update. The causal-response detector
admitted only one object, and the downstream update gate rejected that case as
well. The result does not distinguish whether the source actions lacked
sufficient observable nonrigid response or whether the frozen event and
cross-panel tests were too selective. Opening future outcomes would not answer
that question under the registered gate and would only consume fresh data.

Do not tune V14 on these objects. A successor needs a materially different,
prospectively justified source of support, such as an independent modality or
a stronger observed-motion carrier, and a new fresh-object protocol.

## Provenance

- Prediction implementation revision:
  `6c76829ef9b9086bb309226cad03f23409da8fc5`
- Outcome-blind decision implementation revision:
  `652a49a92b45634fca39e7c86c3aac9980414845`
- Method protocol semantic digest:
  `4bb4133a1bbc6b3cab00f5f9e4e86add1f8c4e86fc7faebc39922a0d8a2af68b`
- Source-lock semantic digest:
  `b8ecb85cf6b6e1aa58dc21fadf1cd77b74998e69d57c8811244f59de75a7a5bc`
- Prediction-runtime semantic digest:
  `daccb34fa21636754a1c0e5b1e37c9be92b3eeed2b9e643253adcff29faf33a7`
- Sparse-support amendment digest:
  `ece5c155ba89d6df822b68dab402d99cad91fc9d1bb3c3baa1173c00b114ddcc`
- Source-decision semantic digest:
  `02a9d0729881869cf044d8c3dddeb00e41ece9ed338dbf81bd1d90e565638676`
- Source-decision file SHA-256:
  `bce10114e979cabe9b870474517a2bbc856c2efa66a5c1b65fc41b2219147fbd`
- Exact remote focused verification: 7 tests passed; changed-file Ruff
  clean.

The machine-readable decision is archived at
`results/sota/diagnostics/deform360_causal_response_direct_depth_v14_source/source_decision.json`.

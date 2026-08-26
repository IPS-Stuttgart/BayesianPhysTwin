# Deform360 covariance source scoring v1

## Purpose

This stage implements the already frozen source reducer for the public-data-only
covariance study. It attaches the permitted future suffix to the ten opened
source object-sessions only after the complete 100-record prefix prediction
barrier exists. It cannot read or authorize confirmation data.

## Inputs

The scorer requires three independently content-addressed inputs:

1. the complete source panel receipt, ten unit artifacts, and 100 prediction
   records;
2. one roster-complete source-observation manifest; and
3. the exact observation archives and source-suffix input files named by that
   manifest.

Every observation row binds the selected prediction, unit manifest, reserved
scoring reconstruction and configuration, two disjoint scoring camera
families, and two scoring-plan artifact IDs. The scorer rehashes the observation
archive and every listed suffix input. Provider and scoring input digests must
remain disjoint. The observation manifest and all admitted roots must remain
disjoint from the forbidden confirmation root.

Observed arrays are fixed as `float64` world-frame metres with shape
`(18, N, 3)` and a Boolean validity mask `(18, N)`. Missing rows remain missing;
they are not nearest-filled. A retained scoring-reconstruction failure carries
no invented array or score and forces `source-technical-negative`.

## Frozen score

For every valid 3D identity event, the candidate uses the already horizon-scaled
predictive covariance plus the common `0.005^2 I` observation covariance. The
reference uses only that common observation covariance. The same registered
mean array is used by both arms. An exact-fallback covariance must be all zero
and must reproduce the reference NLL exactly.

Events receive equal weight within each of the early, middle, and late bins.
The three bin means then receive equal weight within the object-session. The
existing frozen gate gives each of the ten source sessions equal weight and
requires the preregistered overall, stratum, support, and point-identity checks.

## Outputs

The write-once output contains:

- `source-scores.json`;
- `source-decision.json`; and
- `source-score-receipt.json`.

The decision is exactly one of `source-positive`, `source-negative`, or
`source-technical-negative`. A scientific negative is a successful, complete
execution artifact. Only `source-positive` authorizes constructing and sealing
prefix-only confirmation predictions; it still does not authorize opening any
confirmation payload or outcome.

## Current boundary

The implementation and adversarial synthetic tests are available, but the
empirical scorer is not authorized for this protocol. The sole registered
source producer attempt, run `33012751418`, terminated with
`source-technical-negative` and sealed `0/100` required prediction records.
Its write-once attempt ledger is consumed. The source scorer, confirmation
prediction construction, and confirmation outcome opening must not run, and the
producer must not be retried or replaced.

The exact result and compact receipt are documented in
`docs/deform360_covariance_source_panel_v1_result.md` and
`results/sota/diagnostics/deform360_covariance_source_panel_v1/`.

This stage does not establish covariance value, calibration, transfer, point
accuracy, benchmark parity, state of the art, Causal4D benefit, or deployment
safety. It changes no frozen mean, covariance donor, scale, observation model,
roster, endpoint, gate, or claim.

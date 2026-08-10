# Deform360 v5 public-data source execution

## What the measurements are

This experiment uses measurements already released by Deform360: calibrated RGB,
tactile arrays, robot state and action, camera calibration, and held-out-view
geometry. These are real-world recordings. The protocol requires no new recording,
robot execution, contact-registration session, or manually supplied approval.

The only authorization in this path is a content-addressed machine decision. A
passing ten-object source gate may authorize one opening of the twelve locked
public confirmation objects. A person cannot waive a failed check, substitute an
object, split a tied risk score, or select a favorable result.

## Additive execution lock

The frozen scientific design remains
`protocols/locks/deform360_official_hub_joint_sparse_prospective_v5.json`. The
additive execution amendment is
`protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json`.
It resolves two implementation details before development suffix scoring:

1. `B0_physical_fallback` is the frozen automatic Warp twin when its frame-zero
   geometry is admissible, and exact persistence otherwise. Both modes remain in
   the denominator and are reported separately.
2. A source fold passes only when its held-out update is accepted, improves by the
   registered margins against B0, B1, and visual-only V1, is covered by its
   source-only interval, is not harmful, and preserves exact B0 fallback.

The amendment binds the exact v5 policy, cohort, public dataset revision, physical
generator components, source evaluator, and CLI by SHA-256. It does not modify the
v5 cohort, endpoint, confirmation decision, or claim boundary.

## Nested source decision

For each development object, the evidence contains a sealed outer-fold forecast
for that held-out object and nine inner cross-fitted training forecasts. No
training forecast may come from another outer fold whose fit used the current
held-out suffix. Every forecast binds its fit-artifact SHA-256 and exact set of
fit object IDs; the parser requires nine fit objects for the outer forecast and
eight for each inner forecast. Those nine records select the largest complete
lower-risk score block whose object coverage lies in `[0.80, 0.98]` and whose
accepted training updates contain no loss more than 2% above B0. Object IDs never
split ties. At 90%
nominal conformal coverage, the nine-object rank is 9, so the additive interval
correction is the maximum training-object nonconformity score.

The held-out development suffix is opened only after its candidate forecast,
uncertainty forecast, and risk score have been sealed. It scores that fold but
cannot change the threshold or interval. At least 8/10 folds and 4/5 in each
stratum must pass. Aggregate 10% gain over B0, 5% over B1, 2% contact increment
over V1, no harmful accepted update, no stratum regression over 2%, and a valid
full-source refit are also required.

If any check fails, `confirmation_access_authorized` is false. No confirmation
payload is opened and the negative source result is complete.

## Prediction-sealed evidence assembly

The execution publishes two immutable artifacts before invoking the gate. First,
all ten outer forecasts and all ninety inner cross-fitted forecasts are collected
into one outcome-free prediction batch:

```bash
python scripts/science/materialize_deform360_joint_sparse_source_evidence_v5.py \
  seal-batch \
  --execution-lock protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json \
  --prediction-seal /path/to/seal-000.json \
  --prediction-seal /path/to/seal-001.json \
  --output /path/to/source-prediction-batch.json
```

The complete command has one `--prediction-seal` argument for each of the 100
nested records. The batch requires the exact 10-by-10 roster, exact outer and
inner fit sets, one implementation revision, and fold-invariant B0 and B1
forecasts. Its information boundary states that no development suffix or
confirmation outcome has been opened.

Only after that non-replacing batch exists may the workflow open the ten
development suffixes and publish one outcome record for each seal. Assembly binds
every loss to the exact method artifact and prediction-batch identity:

```bash
python scripts/science/materialize_deform360_joint_sparse_source_evidence_v5.py \
  assemble \
  --execution-lock protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json \
  --prediction-batch /path/to/source-prediction-batch.json \
  --outcome /path/to/outcome-000.json \
  --outcome /path/to/outcome-001.json \
  --output /path/to/source-evidence.json
```

The complete command has 100 `--outcome` arguments. Manual substitution is
rejected: missing, repeated, foreign-batch, method-artifact, fit-roster, and
content-identity mismatches fail closed.

## Source-gate command

After the prediction batch and assembled source evidence exist:

```bash
python scripts/science/evaluate_deform360_joint_sparse_source_v5.py \
  --execution-lock protocols/locks/deform360_official_hub_joint_sparse_source_execution_v5.json \
  --evidence /path/to/source-evidence.json \
  --output /path/to/source-gate-result.json
```

The commands refuse malformed identities, duplicate or replacement objects,
future-observation use, confirmation access, non-finite values, changed locks, and
silent output replacement.

## Claim boundary

A passing source result authorizes evaluation; it is not evaluation evidence. It
does not establish performance on the confirmation objects, broader unseen-object
generalization, deployment calibration, safety, Causal4D benefit, official
benchmark parity, or state of the art.

# Deform360 v6 source challenger, covariance, and guard gate

> **Retired progression path.** Before any v6 challenger suffix outcome or v6
> target payload was opened, a source-independent audit found that this adapter
> consumed precomputed guard decisions and calibrated interval summaries instead
> of fitting them inside each outer fold. Its archives remain immutable
> provenance, but its public evaluator now fails closed. Use the corrected v6.1
> contract in `docs/deform360_fresh_object_session_source_v6_1.md`.

## Purpose

This stage implements the first executable boundary of the frozen Deform360
fresh object-session v6 protocol. It uses only the ten already-opened v5
development object-session units. It does not select a fresh cohort, inspect a
v5 confirmation payload or outcome, or authorize a v6 target payload.

The base policy is
`protocols/locks/deform360_official_hub_fresh_object_session_v6.json`.
Candidate-specific covariance applicability is fixed before source suffix
scoring by the amendment at
`protocols/amendments/`
`deform360_official_hub_fresh_object_session_v6_source_covariance.json`.

## Why a covariance amendment was required

The two frozen challengers do not have the same raw covariance family.

- The dynamic endpoint model average has one native law-of-total-covariance
  interpretation from its frozen component mixture.
- The joint-sparse visuotactile candidate can expose working-IRLS, observed-
  information, and group-sandwich common-query covariances.

Pretending that the dynamic candidate also has three IRLS-style covariances
would create nonexistent methods. Comparing only one convenient visuotactile
covariance would hide alternatives. The amendment therefore freezes separate
candidate rosters before any source score is used. Raw covariance remains
uncalibrated; source-only scaling and grouped split conformal are still required.

## Prediction-first evidence

For each of the ten source object-session units, one prediction seal records:

- the exact implementation revision;
- the held-out object and episode identity;
- all six matched candidate/covariance variants;
- the exact other-nine fit roster for every challenger;
- prediction, fit, guard, covariance, and interval artifact identities;
- the source-only risk score and frozen guard threshold;
- availability or one explicit retained failure reason; and
- source artifacts plus a target-closed information boundary.

The registered variants are:

| Variant | Policy candidate | Covariance |
| --- | --- | --- |
| `b0_physical_fallback` | physical fallback | reference |
| `b1_last_causal_residual` | last residual | reference |
| `d1_native_model_average` | dynamic endpoint model average v2 | native model average |
| `vt1_working_irls` | joint-sparse visuotactile v5 | working IRLS |
| `vt1_observed_information` | joint-sparse visuotactile v5 | observed information |
| `vt1_group_sandwich` | joint-sparse visuotactile v5 | group sandwich |

Baselines carry no guard score. A challenger is accepted exactly when its sealed
risk score is no larger than its sealed threshold. An unavailable variant has no
scientific artifact or guard value and is scored as exact physical fallback.

All ten seals are then published as one atomic, non-replacing prediction batch.
Only after that batch exists may the already-open source suffixes be scored.
Every outcome binds one exact seal and prediction artifact. Assembly rejects
missing, duplicate, changed, foreign-batch, or replacement records.

## Selection

The assembled evidence is converted into the existing candidate-agnostic
source tournament. Covariance choices are represented as matched challenger
variants, so candidate and covariance are selected jointly without introducing a
second selector.

The generic tournament checks equal-object point loss, Gaussian proper score,
interval coverage and width, harmful accepted updates, exact fallback, and
leave-one-object-session-out behavior. The v6 adapter then applies the additional
frozen requirements:

- a challenger rather than last residual must be selected;
- the same candidate/covariance variant must win at least eight outer folds;
- at least eight held-out units and four of five per stratum must not regress;
- at least eight units and four of five per stratum must be accepted;
- object-session coverage must lie in `[0.80, 0.98]`; and
- mean full interval width must not exceed `1.25` times the calibrated reference.

A passing result authorizes only a full-source refit of the already selected
candidate, covariance, interval, and guard artifacts. It still sets:

```text
fresh_target_selection_authorized = false
fresh_target_payload_access_authorized = false
claim_authorized = false
```

A failed or unstable source result retains last residual and terminates v6 before
fresh-cohort selection.

## Commands

Seal the outcome-free prediction batch:

```bash
python scripts/science/run_deform360_fresh_object_session_source_v6.py \
  seal-batch \
  --policy protocols/locks/deform360_official_hub_fresh_object_session_v6.json \
  --amendment protocols/amendments/\
    deform360_official_hub_fresh_object_session_v6_source_covariance.json \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --prediction-seal /path/to/seal-00.json \
  --prediction-seal /path/to/seal-01.json \
  --output /path/to/source-prediction-batch.json
```

The complete command has exactly ten `--prediction-seal` arguments.

Attach outcomes after the prediction barrier:

```bash
python scripts/science/run_deform360_fresh_object_session_source_v6.py \
  assemble \
  --policy protocols/locks/deform360_official_hub_fresh_object_session_v6.json \
  --amendment protocols/amendments/\
    deform360_official_hub_fresh_object_session_v6_source_covariance.json \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --prediction-batch /path/to/source-prediction-batch.json \
  --outcome /path/to/outcome-00.json \
  --outcome /path/to/outcome-01.json \
  --output /path/to/source-evidence.json
```

The complete command has exactly ten `--outcome` arguments.

Evaluate the source gate:

```bash
python scripts/science/run_deform360_fresh_object_session_source_v6.py \
  evaluate \
  --policy protocols/locks/deform360_official_hub_fresh_object_session_v6.json \
  --evidence /path/to/source-evidence.json \
  --output /path/to/source-result.json
```

Exit code `0` means one challenger/covariance pair advanced to a full-source
refit. Exit code `3` means the valid source result retained last residual and
terminated v6 before fresh target selection.

## Scientific boundary

This stage is source-only model, covariance, interval, and guard selection. It
does not establish fresh-object transfer, physical-query benefit on the v6
cohort, calibrated deployment uncertainty, Causal4D intervention benefit,
deployment safety, benchmark parity, or state of the art. It neither reads nor
reinterprets the v5 terminal outcome.

# Deform360 covariance-only source authorization gate v1

## Purpose

This gate supplies the missing execution boundary between the frozen
covariance-only predictor and the unopened twelve-session confirmation panel.
It evaluates only the ten already-opened Deform360 source object sessions.

The gate does **not** create a new estimator. It binds the existing software
protocol, paper preregistration, exact `last_residual` point mean,
`independent_endpoint_v1` covariance donor, scales `[8, 16, 16]`, and common
5 mm observation standard deviation.

## Information order

The workflow has three immutable artifacts:

1. a prefix-only prediction batch sealed before source-suffix access;
2. a source-score table produced after opening only the already-opened source
   suffix; and
3. a content-addressed source decision.

The prediction batch must contain exactly 100 records: ten outer-fold contexts
for each of the ten source object sessions. Every unit must contain fold indices
`0` through `9`, and the predeclared scoring record is the diagonal record for
that source unit. Every covariance-only record must preserve the exact reference
mean content.

Confirmation payloads, confirmation predictions, confirmation outcomes,
replacement, and target-informed selection are forbidden in both input
artifacts.

## Frozen source rule

The result is `source-positive` only when all of the following hold:

- all 100 prediction records validate;
- all ten source sessions are present without replacement;
- at least `8/10` units and at least `4/5` units in each stratum are supported
  or use exact fallback;
- the equal-object mean candidate-minus-reference Gaussian NLL is negative;
- the sheet and volumetric mean NLL differences are each non-positive;
- every candidate point mean is identical to `last_residual`;
- the point-metric difference is exactly zero; and
- all fallback records are exact.

A scored scientific failure gives `source-negative`. A retained processing
failure gives `source-technical-negative`. Both keep the confirmation panel
closed and are complete outcomes for this source attempt.

A positive source result authorizes only construction and sealing of the twelve
prefix-only confirmation predictions. It does **not** authorize opening
confirmation payloads or outcomes, and it authorizes no paper claim.

## Commands

Seal and validate the prefix-only batch:

```bash
python scripts/science/run_deform360_covariance_only_source_gate_v1.py \
  seal-batch \
  --input source-prediction-batch.raw.json \
  --output source-prediction-batch.sealed.json
```

After the batch is sealed, attach only the already-open source suffix and seal
its ten-row score table:

```bash
python scripts/science/run_deform360_covariance_only_source_gate_v1.py \
  seal-scores \
  --batch source-prediction-batch.sealed.json \
  --input source-scores.raw.json \
  --output source-scores.sealed.json
```

Evaluate the frozen decision:

```bash
python scripts/science/run_deform360_covariance_only_source_gate_v1.py \
  evaluate \
  --batch source-prediction-batch.sealed.json \
  --scores source-scores.sealed.json \
  --output source-decision.json
```

The evaluator exits with status `0` only for `source-positive`; valid negative
and technical-negative decisions use status `3`. Structural or information-
boundary violations fail with an exception. Outputs are created atomically and
are never overwritten.

## Scientific boundary

This gate can establish only whether the frozen candidate passes its
preregistered source authorization rule. It does not establish independent
object transfer, calibrated target uncertainty, improved point prediction,
physical-state identification, provider competence, Causal4D intervention
benefit, deployment safety, or state of the art.

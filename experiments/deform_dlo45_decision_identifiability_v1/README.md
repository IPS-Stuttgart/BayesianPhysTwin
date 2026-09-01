# DEFORM DLO4/DLO5 decision-identifiability evaluation

This experiment is the real-data companion to
`query_decision_certificate_v1`. It evaluates whether a finite downstream
forecasting decision can be certified even though the complete future
centerline residual remains ambiguous inside a registered query quotient.

## Public-data task

The official DEFORM DLO4 and DLO5 files each contain 56 training trajectories
and 14 held evaluation trajectories. Every trajectory has 500 frames and 12
three-dimensional nodes. Nodes `0`, `1`, `-2`, and `-1` are treated as the
registered endpoint-action carrier; the eight internal nodes are predicted.

At each 25-frame horizon, the observation contains five prefix frames and the
future endpoint-action path. The fallback is a damped endpoint-blend kinematic
forecast. Source analog windows define a learned residual correction.

The finite actions are:

1. exact kinematic fallback;
2. half of the minimum-information/Jeffrey residual correction; and
3. the full minimum-information/Jeffrey residual correction.

For each query, source windows are grouped by a source-frozen future-response
signature. Similarity determines only quotient-class masses. The exact
certificate evaluates worst-case finite-action regret over every complete belief
with those class masses and prior support. It does not use the target future.

## Source and target custody

`evaluate.py source` reads only `DLO4/train` and `DLO5/train`. It uses a
39/9/8 deterministic fit/calibration/source-test split to select the neighbor
count, quotient resolution, kernel temperature, and registered regret tolerance.
It writes a content-addressed model and source seal.

`evaluate.py target` verifies that seal before reading the 14 official evaluation
trajectories per DLO. Within each target window, the code first copies the prefix
and future endpoint nodes, chooses every action, and only then slices the held
future internal nodes for scoring. The pickle carrier co-locates these channels,
so this is semantic rather than byte-level separation; the result records that
limitation explicitly.

No target tuning, target retries, case replacement, or raw-prediction commit is
allowed. Hyperparameters and the action portfolio are frozen in `protocol.json`.

## Primary outputs

The target result reports:

- RMSE for fallback, certificate, Jeffrey point choice, kernel point choice,
  nearest-hypothesis point choice, and hindsight oracle;
- normalized realized regret and harm frequency;
- certificate nonfallback coverage;
- exact and tolerance-unique decision fractions;
- endpoint ambiguity and unsupported within-class specificity;
- per-trajectory ratios and a trajectory-bootstrap interval.

The primary comparison is certificate versus exact fallback. Point comparators
are diagnostics, not alternative certified methods.

## Reproducible workflow

The GitHub Actions workflow is triggered only by a change to:

```text
.github/requests/deform-dlo45-decision-identifiability-v1.json
```

The official public DEFORM repository is pinned to commit
`b73b8b8ecc033caefa693fab7898741d4e6dbeff`. The source job receives sparse
checkouts of only `DLO4/train` and `DLO5/train`; the target job receives sparse
checkouts of only `DLO4/eval` and `DLO5/eval`. Both jobs run with Python 3.12,
NumPy 2.2.6, deterministic thread limits, and content-addressed source artifacts.

The complete one-shot run is GitHub Actions run `33473378340`. Its compact
result artifact has SHA-256 digest
`588e006efe45d5f6bbc5459f9f031a4c4f92c28aaaf4004ef4c0d5c85eba5663`.
The committed evidence is under:

```text
results/science/deform_dlo45_decision_identifiability_v1/
  official-dlo45-one-shot-20260901-v1/
```

That directory contains the dataset provenance, source result, source seal,
target result, and human-readable summary. The sealed source-model SHA-256 is
`a43aed43cd563ee47358e48cab84829dc7eebc77d97725721a11b228f3b6b7f0`.

## One-shot held-data result

The source-only selection chose 16 quotient classes, 16 neighbors, temperature
scale 1.0, and regret tolerance 0.05 for both DLOs. All registered source gates
passed before the held evaluation was opened.

| DLO | Fallback RMSE [mm] | Certificate RMSE [mm] | Reduction | Nonfallback | Harm versus fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLO4 | 31.297 | 29.605 | 5.41% | 45 / 266 | 3 / 266 |
| DLO5 | 37.398 | 36.095 | 3.48% | 37 / 266 | 0 / 266 |

Across all 532 decisions, the certificate RMSE ratio is `0.9572996`, equivalent
to a 4.27% aggregate RMSE reduction. The mean paired trajectory improvement is
4.28%, with a frozen trajectory-bootstrap 95% interval of 3.04% to 5.61%.
The certificate selected a nonfallback action in 82 of 532 decisions.

The diagnostic point estimators achieve larger RMSE reductions but solve a
less conservative problem. For example, their decisions use a single lifted or
point belief rather than controlling worst-case regret over every complete belief
compatible with the registered quotient masses. They are therefore reported as
headroom, not as substitutes for the certificate.

## Claim boundary

The exact optimization is conditional on the finite source-window support,
registered quotient partition, and loss matrix. This experiment is within-DLO
held-trajectory validation. It does not establish unseen-object generalization,
identify a unique physical state or cause, prove the quotient physically
correct, calibrate probabilities, certify arbitrary control actions, or authorize
deployment.

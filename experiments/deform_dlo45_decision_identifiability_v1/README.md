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

## Run

The GitHub Actions workflow is triggered only by a change to:

```text
.github/requests/deform-dlo45-decision-identifiability-v1.json
```

It runs the source and target stages on the self-hosted runner tagged
`gpuserver4090`, uploads the complete compact evidence, and commits only JSON and
Markdown summaries under:

```text
results/science/deform_dlo45_decision_identifiability_v1/<run_key>/
```

## Claim boundary

The exact optimization is conditional on the finite source-window support,
registered quotient partition, and loss matrix. This experiment is within-DLO
held-trajectory validation. It does not establish unseen-object generalization,
identify a unique physical state or cause, prove the quotient physically
correct, calibrate probabilities, certify arbitrary control actions, or authorize
deployment.

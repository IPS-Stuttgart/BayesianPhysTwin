# Deform360 Prob4D source gate

## Purpose

This automated gate decides whether the correlation-aware Prob4D covariance
fit may proceed from the ten public Deform360 calibration objects to the twelve
locked public confirmation objects. It uses released real-world measurements;
it requires neither new recording nor human approval. Confirmation payloads,
future frames, and target outcomes remain closed while this gate is evaluated.

The thresholds are content-addressed before source residuals are evaluated in
`protocols/locks/deform360_official_hub_prob4d_source_gate_v1.json`. A failed
gate is terminal for this method version. Objects, cameras, and failures cannot
be replaced.

## Frozen version-1 result and version-2 scope

The sole version-1 execution is a valid source-support negative. Of 324 frozen
camera streams, 313 had released robot geometry in the causal prefix and 11
were retained with the exact reason
`released-robot-geometry-outside-fixed-camera-prefix`. There were no technical
failures and every source object retained supported views, but version 1
required 324 of 324 streams. It therefore stopped before camera images,
prediction residuals, covariance fitting, or confirmation data were opened.
Version 1 is not rerun, weakened, or reinterpreted.

Version 2 is a separately content-addressed source protocol. Its camera
eligibility is determined only from released synchronized robot/taxel geometry
projected through released camera calibration over the immutable causal
prefix. The policy requires all ten source objects, at least two supported
streams per object, and at least 90% supported streams overall. An unsupported
stream is retained in the plan provenance and excluded; it is never replaced.
Any technical failure remains terminal. Camera pixels, prediction residuals,
calibration outcomes, future frames, and confirmation outcomes cannot affect
eligibility.

The version-2 eligibility lock is
`protocols/locks/deform360_official_hub_prob4d_camera_eligibility_v2.json`.
Passing this eligibility check permits source covariance calibration to run;
it is not the source calibration gate itself and does not authorize
confirmation.

## Statistical unit

The physical object is the transfer unit. Dense point rows first collapse by
the materializer's declared camera/window/spatial correlation cluster. Every
fit then gives equal weight to each training object, independent of its number
of pixels, cameras, windows, or frames.

For each held-out object, the other nine objects fit five variance factors:

- point variance parallel and lateral to the observation ray;
- Sim(3) scale variance;
- Sim(3) rotation-block variance; and
- Sim(3) translation-block variance.

The held-out point score is a three-degree-of-freedom NEES. The held-out gauge
score is a seven-degree-of-freedom NEES computed from the complete scaled
covariance, retaining cross-block covariance. The 90% thresholds are fixed
chi-square quantiles, not values estimated from the held-out object.

## Frozen decision

The gate requires all of the following:

- exactly ten objects, five sheet and five volumetric;
- at least two sealed metric streams, 32 effective point clusters, and eight
  gauge rows per object;
- at least eight of ten leave-one-object-out folds meeting point coverage,
  gauge coverage, and factor-stability limits;
- object-balanced point coverage in `[0.80, 0.98]`;
- object-balanced gauge coverage of at least `0.75`;
- bounded object-balanced point and gauge NEES per degree of freedom;
- no material degradation of coverage error relative to the uncalibrated
  covariance; and
- minimum point and gauge transfer coverage in both strata.

The exact numeric contract and its artifact ID are in the lock file. These are
effect-size and calibration gates, not significance tests.

## Command

```bash
python scripts/science/evaluate_deform360_prob4d_source_gate.py \
  --samples /durable/source-calibration-samples/samples.json \
  --selection protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-spec protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider_spec.json \
  --metric-prior-policy protocols/locks/deform360_official_hub_prob4d_robot_metric_gauge_v1.json \
  --prediction-root /durable/calibration-visual-production \
  --source-calibration-root /durable/source-calibration-result \
  --source-calibration-result /durable/source-calibration-result/source-calibration-result.json \
  --gate-lock protocols/locks/deform360_official_hub_prob4d_source_gate_v1.json \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output-dir /durable/deform360-prob4d-source-gate
```

Publication is atomic and no-overwrite. The result copies the exact gate lock,
binds the sample bundle and source-calibration result by hash, records every
fold and check, and recursively checksums the portable decision.

## Registered execution

`.github/workflows/deform360-prob4d-source-gate.yml` registers the complete
source-only pipeline. Pull requests run contracts only. Empirical execution is
possible solely through the separately reviewed, main-branch one-shot caller
`launch-deform360-prob4d-source-gate-once.yml`, with
`execute_authorized=true`, on the named `workstation2` runner.

The runner consumes the already sealed ten-object visual production and the
released Deform360 robot/camera measurements. It records complete-stream
support before fitting, uploads compact negative or positive evidence before
enforcement, permits no replacement stream, and opens no confirmation payload.
A source-gate failure is therefore a valid terminal result for this method
version. No manual approval, physical registration review, new recording, or
robot execution is part of this public-data path.

## Claim boundary

A pass authorizes only a separately locked, independent evaluation on the
public confirmation objects. It is not itself confirmation evidence and does
not establish prediction improvement, Causal4D benefit, safety, official
benchmark parity, or state of the art.

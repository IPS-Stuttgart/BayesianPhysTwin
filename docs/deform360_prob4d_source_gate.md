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

## Versioned continuation after retained camera support negatives

The first protected execution produced an immutable metric-batch result with
`313/324` supported streams, `11/324` source-side support negatives, zero
technical failures, and support for all ten objects. The support negatives all
mean that released robot geometry lies outside a fixed camera prefix. They are
retained in the denominator and are not replaced.

That execution stopped before sample materialization because its orchestration
required all 324 cameras even though the preregistered gate lock requires only
at least two supported metric streams per physical object. Every object in the
immutable batch has between 29 and 35 supported streams. The actual
object-balanced covariance fit and leave-one-object-out source gate therefore
did not run, and the stopped workflow is neither a positive nor a negative
scientific gate result.

`admit_deform360_prob4d_metric_support.py` provides a separate versioned
continuation. It leaves the original metric batch byte-for-byte unchanged,
verifies its recursive checksums and exact result identity, reads the support
minimum from the pre-existing source-gate lock, and emits a plan over supported
streams only when every object passes that frozen minimum and no technical
failure exists. The admission result binds the full source batch, all retained
support-negative counts, the exact lock, the emitted plan, and a no-replacement
information boundary.

The one-shot workflow
`.github/workflows/continue-deform360-prob4d-source-gate-v2.yml` binds the exact
predecessor batch and then executes:

```text
immutable complete v1 metric batch
  -> frozen per-object support admission
  -> supported-stream metric plan
  -> source sample materialization using metric-batch/metrics
  -> object-balanced source calibration
  -> unchanged frozen leave-one-object-out gate
```

This continuation changes no estimator, object split, covariance threshold,
proper score, or gate margin. It also corrects the orchestration path from the
nonexistent `metric-prefix/` directory to the contract-defined `metrics/`
directory. A valid negative remains a complete outcome. Confirmation data stay
closed regardless of the result.

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

## Independent terminal-result audit

The one-shot source workflow may stop before covariance fitting when the frozen
all-stream support rule fails. That is a valid scientific terminal state, not a
malformed source-gate result. The independent audit workflow therefore has two
strictly separated paths:

- `audit_deform360_prob4d_support_stop.py` validates a pre-calibration support
  stop containing only the pipeline receipt and the checksummed metric-batch and
  support receipts; and
- `evaluate_deform360_prob4d_source_gate.py` validates a completed calibrated
  source-gate bundle.

For a support stop, the dedicated auditor reconstructs the complete metric-batch
result, requires the exact pinned object and stream roster, verifies all retained
support-negative and technical-failure accounting, requires all later stages to
be skipped, and emits `validated-support-negative` or
`validated-technical-negative` with confirmation access set to false. It does
not invent a source-gate result ID or reinterpret a skipped calibration as a
calibration failure.

The support-stop command is target-closed and accepts only exact run, revision,
provider, cohort, and admission identities:

```bash
python scripts/science/audit_deform360_prob4d_support_stop.py \
  --source-root /path/to/compact-source-artifact \
  --output-dir /path/to/new-independent-audit \
  --source-run-id 31297018948 \
  --source-run-attempt 1 \
  --source-run-conclusion failure \
  --source-head-sha ded8910becbbffe958dfd18c84ad91069e7087a4 \
  --source-artifact-id 9033414269 \
  --source-artifact-name deform360-prob4d-source-gate-31297018948-1 \
  --auditor-revision <exact-auditor-revision> \
  --expected-production-result-id 146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89 \
  --expected-admission-id 715ab8479bad4d97eba766cdba1a161f1f6e83e3fd597bb09a2bf8ab8dc91e15 \
  --expected-prob4d-revision 25d90ef7f78ba4307f4555cb636d666004e1bf66 \
  --expected-motioncrafter-revision 9cb4e9679f5f34e249945544052464ef46324bc2 \
  --expected-object-count 10 \
  --expected-admitted-stream-count 324
```

The output is a no-overwrite, recursively checksummed audit bundle. Error text is
not copied into the portable receipt; only its exception type and SHA-256 digest
are retained.

## Claim boundary

A pass authorizes only a separately locked, independent evaluation on the
public confirmation objects. It is not itself confirmation evidence and does
not establish prediction improvement, Causal4D benefit, safety, official
benchmark parity, or state of the art.

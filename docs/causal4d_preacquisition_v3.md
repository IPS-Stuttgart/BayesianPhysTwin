# Causal4D Pre-Acquisition Amendment v3

## Status

This amendment supersedes v2 before any physical execution. It preserves the
immutable `causal4d-preacquisition-v2` tag, all 12 source-panel executions, all
36 confirmatory executions, and every target ID. It adds no acquisition burden.

Canonical SHA-256:

```text
5dd12d62242d672789e802ed6ed4365922e607932d15ef02a76b3208cdb9a1e2
```

## Discovery boundary

The 12-run panel is model-family discovery, not confirmation. No p-value or
null-rejection claim is permitted. Effects must have the predeclared direction,
exceed a repeatability-scaled magnitude threshold, and agree in at least two of
three exact repetitions.

The repeatability floor `sigma_repeat` is the square root of the equal-profile
mean between-repeat variance of object-frame residual coordinates. Metric-level
gates use the corresponding repeatability standard deviation rather than point
or frame counts.

## Cross-fitted shrinkage

The source panel has three leave-one-replicate-out folds:

```text
8 fresh-reset sessions: fit mechanism parameters
4 fresh-reset sessions: freeze mechanism, refit c_base and c_M from each prefix
```

Every source execution is held out exactly once. In-sample shrinkage is not
evidence. A mechanism becomes eligible for confirmatory evaluation only when:

- geometric-mean held-out readout-correction shrinkage is at least 10%;
- at least 8 of 12 sessions shrink;
- mean track and late-track gains each exceed one repeatability SD;
- mean CD degradation is at most half one repeatability SD.

Passing these gates does not confirm a mechanism. It only permits the mechanism
to enter the already locked confirmatory evaluation.

## Signature thresholds

### Reset-separated reversal

`lift_high` and `lower_high` are paired by replicate after independent resets.
Project residuals onto the common action axis and time-normalize them. Require:

- negative lift/lower cosine (reported as sign-flip cosine) at least `0.50`;
- odd-component RMS at least `1.50 * sigma_repeat`;
- the expected direction in at least 2 of 3 pairs.

### Continuous reversal

Every command already uses an uninterrupted out-hold-return waveform. This is
the separate hysteresis/non-closure test. Post-return minus pre-action residual
RMS must exceed `1.50 * sigma_repeat`, persist for at least three frames, and
have a consistent direction in at least 2 of 3 repetitions.

Reset-separated reversal diagnoses direction dependence from matched initial
conditions. Continuous return diagnoses path dependence. They are not
interchangeable.

### Speed

The measured slow/fast peak-speed ratio must lie in `[0.35, 0.65]`. At fixed
direction, amplitude, hold, and contact, the phase-aligned residual difference
must exceed `1.50 * sigma_repeat` and agree in at least 2 of 3 repetitions.

### Hold relaxation

Compare the first and last three frames of the long hold. Require amplitude at
least `1.50 * sigma_repeat`, agreement in at least 2 of 3 repetitions, a
one-exponential log-space `R^2 >= 0.80`, and an observable time constant between
`66.7 ms` and `0.50 s`. This is an eligibility signature, not a material
constant estimate.

## Calibration arithmetic

Each outer fold has nine independent calibration sessions and one score per
session. The achievable nominal grid is therefore multiples of `1/10`. At 90%,
the split-conformal rank is exactly 9 of 9: the largest calibration score sets
the interval.

The report must include the largest and second-largest scores, maximum/median
ratio, leave-one-calibration-session-out thresholds, and interval width by
outer fold. These are fragility diagnostics only and cannot select another
threshold. The claim is finite marginal execution-block coverage under session
exchangeability, not pooled-coordinate or worst-group coverage.

## Contact-registration handoff

The next physical artifact uses schema 2 and cannot encode contact as one exact
node. Every region requires a weighted patch over at least two graph nodes,
centroid covariance, a local normal/tangent frame, overlays from at least three
calibrated views, and two independent reviews. The artifact also locks
camera/controller/support SE(3) transforms and covariance, frame-closure error,
gravity, support geometry, twin-geometry hash, approval identity, timestamp,
and source checksums. Contact uncertainty must be less than half the distance
to the nearest other experimental region.

Generate an incomplete operator template with:

```bash
causal4d-contact-registration template \
  configs/causal4d/sloth_multi_action_v1.json \
  /path/to/contact_registration.v2.template.json \
  --camera-id camera_0 \
  --camera-id camera_1 \
  --camera-id camera_2 \
  --object-node-count 6895
```

No slip pilot may begin until the completed artifact passes
`causal4d-contact-registration validate` and carries independent approval.

## Commands

```bash
causal4d-preacquisition-protocol-v3 generate \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/sloth_preacquisition_v2.json \
  configs/causal4d/sloth_preacquisition_v3.json

causal4d-preacquisition-protocol-v3 validate \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/sloth_preacquisition_v2.json \
  configs/causal4d/sloth_preacquisition_v3.json
```

The physical sequence remains unchanged: contact registration, slip/reset and
synchronization pilot, 12-run source panel, then the confirmatory acquisition.

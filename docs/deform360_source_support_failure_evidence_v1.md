# Deform360 source-support failure evidence v1

This contract converts the frozen official-Hub Deform360 source-support result
into the equal-physical-object input required by the merged provider-failure
census.

It is a deterministic custody transformation. It does not rerun Prob4D,
MotionCrafter, calibration, or BayesianPhysTwin inference.

## Frozen source

The materializer accepts exactly the retained metric batch from:

- source workflow run: `31297018948`, attempt `1`;
- source revision: `ded8910becbbffe958dfd18c84ad91069e7087a4`;
- Actions artifact ID: `9033414269`;
- artifact SHA-256:
  `7247a2a260509c4c226e7ca437aff09d090abf6d2ca08f471a2143ea7d4bf7de`;
- metric-batch result ID:
  `f246394c84fd643b6ec8961dbcb2101a73c34e46d5eaf43961f28429aeb197eb`;
- metric-batch file SHA-256:
  `679550aff53d3b615f63c66ee78318258893867511dd6c33100d1cf10c0f5be6`;
- visual-production result ID:
  `146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89`;
- admitted streams: `324`;
- supported streams: `313`;
- retained support-negative streams: `11`; and
- retained technical-failure streams: `0`.

The ordinary input file must remain at the registered results-side path below
`/mnt/lexar4tb/datasets/deform360/results`. The materializer verifies its exact
bytes, content identity, object roster, camera roster, support accounting, source
artifacts, and closed information boundary.

## Equal-object aggregation

The statistical unit is `physical-object`, not frame, view, camera, point, or
stream.

All ten objects remain rejected because the frozen complete-stream method
terminated before sample materialization, source covariance fitting, or an
object-level downstream update. The transformation therefore never marks one of
the four complete-support objects as accepted.

For the six objects containing at least one retained support-negative stream:

- `technical_valid = true`;
- `provider_support_complete = false`; and
- `result_reason = released-robot-geometry-outside-fixed-camera-prefix`.

For the four objects whose sealed camera roster was complete:

- `technical_valid = true`;
- `provider_support_complete = true`;
- `accepted = false`; and
- the rejection remains unresolved because no downstream object-level admission
  occurred.

Every other independently owned gate remains `null`. In particular, the source
support result does not establish numerical convergence, physical-query
identifiability, gauge consistency, covariance calibration, material identity,
robust support, or physical-model agreement.

The resulting census is therefore fixed at:

- independent physical objects: `10`;
- accepted: `0`;
- unsupported-provider-geometry: `6`; and
- unresolved global rejections: `4`.

This is more conservative than treating the four complete-support objects as
successful and more informative than counting all 324 camera streams as
independent cases.

## Materialization

The reviewed script is:

```text
scripts/science/materialize_deform360_source_support_failure_evidence_v1.py
```

It publishes a retry-safe content-addressed directory below:

```text
results/bayesian-phystwin/deform360-provider-failure-evidence-v1/
```

The directory contains:

- `provider-failure-evidence.json`;
- `materialization-receipt.json`; and
- `SHA256SUMS`.

An existing directory is reused only when every retained byte is identical. It
is never overwritten.

The frozen evidence identities are:

- aggregation policy ID:
  `556635682e0c1366c1d82aa86b80cc1cfaeef7bdb59833d5e1962df24adba665`;
- evidence raw SHA-256:
  `9d14ae5645e35a41ff3b5c53e75641509d872815c1901aa9369f42583976bc2e`;
- evidence canonical content SHA-256:
  `17fd5f106e9796e57fee2f2eb9a04305cc073d383b8ce1c73a6429f99f663eaf`;
- materialization receipt ID:
  `eca77c31bb2041f14ebaf8fe6610568ecfdbe207e18eb1f66ab46ea96b73d467`;
  and
- provider-failure report ID:
  `59a342e4669c2e4c5f09093043cdd76e1d678499b460f5f4e6574fbe43d53f0e`.

## One-shot execution

The reviewed one-shot workflow is:

```text
.github/workflows/launch-deform360-provider-failure-census-v1-once.yml
```

Pull requests run hosted contracts only. A merge changing that workflow starts
one protected-`main` execution on the sole `self-hosted` runner and verifies
`RUNNER_NAME == workstation2` before opening the exact results-side metric
batch.

The workflow then:

1. materializes and validates the content-addressed ten-object evidence;
2. runs `validate_deform360_provider_failure_census_payload`;
3. executes `bpt diagnostic run diagnose-provider-failures`;
4. verifies the fixed `6/4` attribution and all expected content identities;
5. uploads the evidence, report, summaries, execution receipt, and checksums; and
6. posts a bounded terminal receipt to issue `#148`.

## Information boundary

The workflow registers the official and adaptive-confirmation roots only as
lexical deny boundaries. It does not recursively list, hash, copy, or open them.

It keeps all of the following false:

- confirmation payloads opened;
- adaptive-confirmation payloads opened;
- target outcomes used;
- future frames used; and
- replacement allowed.

The result identifies unsupported provider geometry as the dominant observed
source-only failure. It does not retroactively authorize a new provider,
confirmation access, physical-benefit claims, deployment claims, Causal4D
claims, or state-of-the-art claims.

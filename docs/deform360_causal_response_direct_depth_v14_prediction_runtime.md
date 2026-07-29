# V14 Source Prediction Runtime

## Scope

This runtime is an operational child of the frozen V14 estimator. It does not
change the adaptive carrier, event scan, direct-depth observation model,
admission thresholds, robust belief update, fallback, or source advancement
gate.

The runtime exists to turn an admitted source case into one immutable
prediction or exact fallback before any hidden identity, future geometry, or
source metric is authorized.

## Prefix Contract

Each source prediction consumes:

- the already-sealed frame-zero adaptive carrier;
- the already-sealed 76-frame physical and persistence backbones;
- depth and object masks only for frames 0 through 57;
- released measured end-effector origins only for frames 0 through 57; and
- released normalized tactile arrays only for frames 0 through 57.

The actuator position is the translation row
`robot.actions[..., 0, :]`. Rotation rows and aperture metadata are not
mistaken for Cartesian points.

## Tactile Boundary

Released Deform360 tactile values are thresholded, unitless,
episode-peak-relative responses. They are not calibrated contact
probabilities.

V14 uses the framewise maximum over all released sensors and taxels as a
contact confidence. This has three deliberate consequences:

1. duplicating a sensor cannot increase confidence;
2. correlated taxels are not counted as independent evidence; and
3. tactile supplies causal contact support only, never metric geometry,
   association probability, or observation reliability.

The frozen admission threshold is applied to this confidence without fitting a
calibration transform on source outcomes.

## Artifact Chain

The production operator validates:

```text
method protocol
-> finalized 12-object source lock
-> admitted hash-only preflight
-> sealed frame-zero carrier
-> sealed physical/persistence backbone
-> 58-frame prefix custody artifact
-> V14 prediction or bit-exact fallback
```

The post-source-lock runtime protocol binds the exact method, source lock,
admission prelock, physical prelock, twelve admission artifacts, twelve
physical manifests and archives, and the exact source hashes of the runtime
builder, prediction module, runner, and amended preflight validator. The latter
is bound separately because the original method JSON predates the prospective
camera-completeness repair; the admission child lock and every accepted
preflight already use the repaired validator. The runtime protocol is created
only after the source lock exists and before any prefix scan or source outcome
is opened.

Create that child lock from a clean checkout with:

```bash
python scripts/remote/prepare_deform360_causal_response_direct_depth_v14_prediction_runtime.py \
  --repo "$REPO" \
  --method-protocol "$METHOD_PROTOCOL" \
  --source-lock "$SOURCE_LOCK" \
  --admission-prelock "$ADMISSION_PRELOCK" \
  --physical-prelock "$PHYSICAL_PRELOCK" \
  --admission-root "$ADMISSION_ROOT" \
  --physical-root "$PHYSICAL_ROOT" \
  --output "$PREDICTION_RUNTIME"
```

The builder ignores unrelated rank directories but requires exactly one
admitted report and one validated physical carrier for each of the twelve
source-lock cases. It refuses a rejected or duplicate source, a prelock
mismatch, or disagreement among rank, case, object, admission, and physical
hashes.

## Information Boundary

The operator is forbidden from reading:

- object observations after frame 57;
- future tactile measurements;
- future identity trajectories;
- future point clouds or evaluation metrics;
- any target object or target outcome; and
- any held-v8 artifact, process, or identity.

Every rejection returns the selected physical-or-persistence baseline byte for
byte. Source outcomes remain unavailable until all twelve prediction
dispositions are complete and validated.

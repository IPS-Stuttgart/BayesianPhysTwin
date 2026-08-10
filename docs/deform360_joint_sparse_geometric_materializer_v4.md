# Deform360 joint-sparse geometric materializer v4

## Purpose

The v1–v3 Deform360 source attempts established that complete per-camera support
is too restrictive for the retained public data. Version 4 therefore evaluates
whether complementary partial observations are jointly informative at the
physical-object and registered-query level.

The v4 evaluator already exists. This materializer supplies its previously
missing development inputs using only result-side evidence that was produced
before v4:

- the complete v2 robot-metric grids and camera calibrations;
- integrity-bound MotionCrafter prediction support masks; and
- the immutable ten-object development selection and provider locks.

It does not use MotionCrafter point values, scene flow, prediction residuals,
Prob4D calibration outcomes, future frames, adaptive-confirmation data,
confirmation data, or target outcomes.

## Frozen sources

The materializer policy binds the exact retained evidence:

```text
visual production result
146f885351b2af0134b8b3d3c28a76deaa899749b1b1306e0d7061807ae95f89

visible-camera metric batch
2e7a16ce502ac877f56457809683f5b30d40eee5ed290547043010eeed1fefa6

Prob4D revision
25d90ef7f78ba4307f4555cb636d666004e1bf66

MotionCrafter revision
9cb4e9679f5f34e249945544052464ef46324bc2
```

The frozen materializer policy ID is:

```text
08405c7e85a4730b1affb0110f9d50bcb02db26462ce95bda374c8df83ef845b
```

The ten objects are explicitly development data because their v1–v3 support
patterns informed the new protocol. The resulting report cannot calibrate or
open the independent confirmation cohort.

## Partial factors

For each retained camera and overlapping causal window, the materializer
intersects:

1. the released robot-metric valid mask; and
2. the integrity-bound MotionCrafter `valid_mask`.

It never opens `point_map`, `scene_flow`, `deform_mask`, or prediction residuals.
The robot-metric world points are streamed one frame at a time from the retained
NPZ archive, and only valid projected taxels are retained in memory.

Rows are grouped into fixed 2 cm world-space voxels. Within each camera-window,
at most 64 frame/voxel representatives are selected by a deterministic SHA-256
ranking. The full row identity includes object, episode, camera, window, frame,
pixel, world voxel, support-mask digest, and source artifact identities.

Repeated observations of the same object/frame/world-voxel across cameras or
overlapping windows share one correlation-group ID. Their provider composite
weights are set so that, after the evaluator's fixed effective-sample cap, the
group carries exactly one unit of generalized-Bayes likelihood power.

## Registered physical query

The materializer defines five source-independent non-rigid affine modes. Let

```text
z = (world point - object centroid) / object RMS radius.
```

The state Jacobian consists of five symmetric trace-free matrices applied to
`z`:

- one x-versus-y diagonal mode;
- one xy-versus-z diagonal mode;
- xy shear;
- xz shear; and
- yz shear.

The matrices are Frobenius-orthonormal. The query is the identity over the five
mode amplitudes, measured in meters. Translation, rotation, and isotropic scale
are excluded from the physical query and represented as nuisance variables.

## Nuisance and covariance model

Every retained camera receives a seven-parameter normalized similarity nuisance
block:

```text
translation (3) + rotation-like modes (3) + isotropic scale (1)
```

A global root and one child per camera form a causal tree prior in precision
form. The frozen root standard deviation is 5 cm and the camera innovation
standard deviation is 2 cm in equivalent normalized displacement units.

Observation covariance is anisotropic relative to the released camera center:

```text
lateral standard deviation = 5 mm
axial standard deviation   = 20 mm
```

A separate shared 3-D bias is retained and marginalized by the v4 evaluator.
These quantities are structural development assumptions, not calibrated
uncertainty claims.

## Content and custody

Each object is emitted as:

```text
cases/<index>-<object>/descriptor.json
cases/<index>-<object>/arrays.npz
```

The descriptor is the canonical `Deform360JointSparseFactorBatchV4` identity
record. The NPZ contains the exact arrays named by that record. The top-level
manifest binds all descriptor/array byte counts and SHA-256 values, the exact
implementation revision, the original selection and provider lock, and the
frozen v4 policy.

Publication is atomic and non-replacing. The persistent result contains:

```text
materializer-policy.json
v4-policy.json
manifest.json
materialization-result.json
cases/...
SHA256SUMS
```

The subsequent evaluator writes a separate immutable development report. Exit
code 0 means the frozen development-design gate passed; exit code 3 means a
complete support-negative result. Either outcome leaves confirmation closed.

## Information boundary

The protected execution receives only two retained result roots:

```text
/mnt/lexar4tb/datasets/deform360/results/bayesian-phystwin/
  calibration-visual-production/...

/mnt/lexar4tb/datasets/deform360/results/bayesian-phystwin/
  deform360-prob4d-visible-source-gate-v2/.../metric-batch
```

The official/raw and adaptive-confirmation locations are registered only as
forbidden lexical boundaries and are not passed to the materializer command.
The execution cannot authorize confirmation, replace factors, cameras, or
objects, or turn a development pass into a performance claim.

Pull requests execute only the hosted, read-only contract matrix. The data-bearing
`workstation2` job is push-only and can start only after reviewed source reaches
protected `main`.

## Validation boundary

The hosted contract matrix type-checks all six materializer modules, exercises
the support-only NPZ and source-admission contracts, verifies the workflow's
retained-result command boundary, and checks source-distribution membership.
These checks establish implementation and custody integrity only; they do not
convert the frozen structural covariance model into an empirical calibration or
performance claim.

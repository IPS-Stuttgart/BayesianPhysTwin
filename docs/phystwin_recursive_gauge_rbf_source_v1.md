# PhysTwin recursive gauge-RBF source smoke v1

Status: completed negative one-case source smoke. The prefix gate rejected the
recursive update and selected the frozen dense comparator exactly. This
camera-only arm is stopped without expansion or post-open tuning. See
`docs/phystwin_recursive_gauge_rbf_source_v1_result.md`.

## Question

Can strict three-view CoTracker3 material observations improve the frozen
released-dense PhysTwin comparator when the observation update:

- carries metric covariance in square metres;
- keeps perception reliability independent of the PhysTwin innovation;
- treats unknown cross-view correlation conservatively;
- represents shared camera translation as a nuisance rather than physical
  state;
- propagates a full-covariance spatial belief through action-conditioned local
  rotations; and
- falls back exactly when a causal prefix-CD gate does not improve?

This is distinct from the stopped static sparse-identity endpoint arm. It tests
a recursive state and action-conditioned future transport. It does not reopen
that arm's covariance, reliability, or support settings.

## Frozen method family

The intended one-case smoke uses:

- `single_lift_cloth`, already opened and used only for development;
- the exact selected raw MatPhys trajectory with SHA-256
  `5e41ce3bfea780add79c20841084422ad7cad5e6e2443f3c2d2fca9729b8dd72`;
- the existing CoTracker3 cue archive with SHA-256
  `713fd1ac124c72f9835d848f8c2f6e3622667936c8b5626b42f744a8f2347d56`;
- strict support from all three distinct camera poses;
- 16 frame-zero farthest-point centers selected using availability before the
  validation boundary;
- four quantiles of causally supported frames, including the last frame with at
  least four observed selected centers;
- one correlation group per fused frame with an effective-information cap;
- the existing Student-t gauge-aware likelihood;
- the existing dense temporal comparator with `gamma=0.25`;
- local proper-rotation transport estimated from the physical trajectory; and
- a 1% validation-prefix Chamfer improvement requirement.

The validation gate reads released pseudo-track geometry but cannot receive
manual material identities. Manual tracks enter only after prediction sealing.

## Artifact boundary

The runner has three commands:

1. `prepare-prefix` materializes a prefix-only observation artifact and reports
   target-free support/covariance diagnostics.
2. `seal-prediction` verifies the committed protocol and input hashes, writes a
   future trajectory plus covariance, and cannot load the future target or
   manual tracks.
3. `score` verifies the prediction seal before opening the already-authorized
   source future.

The source-smoke gate requires:

- the prefix gate to admit the recursive update;
- at least 5% future manual-track improvement over the dense comparator;
- no more than 1% future Chamfer regression; and
- both candidate metrics no worse than the raw physical trajectory.

Failure stops this exact camera-only arm without tuning on the opened future.
Passing authorizes only an object-disjoint source-panel design. It does not
authorize a fresh target, held-v8 access, or a state-of-the-art claim.

## Completed decision

The recursive filter accepted four causal updates, but their independent
prefix-CD improvement was only `1.409e-7` as a fraction, versus the locked
`0.01` requirement. The candidate therefore fell back exactly and produced
zero change in future CD and manual-track error. The future smoke gate failed.

Do not run this arm over the opened 19-case cohort. Preserve the recursive
belief implementation for future observation sources that add independent
information, but do not tune this strict-three-view camera feeder against the
opened result.

## Calibration boundary

The runner reports coordinate coverage and point NEES from the recursive
correction covariance. This covariance does not yet include total PhysTwin
model-form error, replay variance, or target observation noise. Raw coverage is
therefore a diagnostic and must not be described as calibrated predictive
coverage.

## Controls

Focused tests establish:

- scalar observation-variance compatibility;
- covariance-aware strict multiview loading;
- shared-bias and state identifiability controls;
- full recursive covariance propagation;
- local physical rotation transport;
- zero-residual dense-comparator parity;
- no manual-track argument in the prediction API; and
- byte-exact fallback after a failed prefix gate.

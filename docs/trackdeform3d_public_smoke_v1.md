# TrackDeform3D public smoke v1

## Why this benchmark

The existing Deform360 studies established a hard camera-only limit: coherent
triangulation bias can preserve multiview and pairwise consistency while still
creating a harmful state update. The public TrackDeform3D release provides a
different information pattern: metric RGB-D, calibrated robot end-effector
poses, object masks, graph topology, and material-keypoint trajectories. Its
robot poses and depth channel can therefore test a guarded Bayesian state update
without relying on another correlated multiview camera vote.

The upstream sample contains one chunk each for a DLO, branched DLO, rectangular
fabric, and T-shirt cloth. It is useful for interface and capacity testing, but
it is not an independent benchmark cohort. The locked upstream revision also
contains no license file. No sample result may be called confirmation, state of
the art, or a public benchmark result until reuse terms and the full release are
confirmed with the authors.

## Development and validation boundary

Admission reads NPZ headers and hashes but no RGB-D, pose, mask, keypoint, or
future values. The official tracker then creates a pseudo-observation trajectory
in a separately sealed stage. This trajectory is generated from future RGB-D;
it is useful for a capacity smoke but is not independent material-point ground
truth.

Clip zero was opened for development. It established that a known-action
inextensible graph prior substantially beats frame-zero persistence and sparse
constant velocity, while transferring one static prefix correction slightly
worsens the future. The action-conditioned model below was fixed after that
diagnosis. Clip one is the untouched validation clip: no prediction or future
score may be inspected until the implementation, tests, and prediction lock are
committed.

The temporal and identity boundaries are fixed before that trajectory is
scored:

- one 300-frame development clip and one disjoint 300-frame validation clip;
- 60 prefix frames and 60 untouched future frames;
- 25% of frame-zero identities selected deterministically by spatial
  farthest-point sampling;
- observed identities may update the state during the prefix;
- only disjoint hidden identities are scored in the future.

Future RGB-D and future TrackDeform3D keypoints are evaluator-only. They may not
construct a prediction.

## Frozen arms

The smoke compares frame-zero persistence, sparse-prefix constant velocity, a
known-action graph prior, and a guarded Bayesian discrepancy update. The graph
prior assigns the two chain endpoints to the two end effectors using frame-zero
geometry, transfers measured endpoint displacements along graph geodesics, and
applies 40 iterations of edge-length projection. It uses released robot poses
and frame-zero topology, never future object keypoints.

The new Bayesian arm models the residual in the first four graph-Laplacian modes
as a linear function of an intercept, both end-effector displacements, and both
end-effector frame velocities. Observation standard deviation is fixed at 5 mm,
coefficient prior standard deviation at 30 mm, and corrections are capped at
100 mm. Frames 0--39 fit a prefix-only gate and frames 40--59 validate it. The
update is admitted only if validation RMSE improves by at least 10% and the last
five validation frames do not regress. Rejection returns the known-action graph
prior bit for bit.

Predictive covariance is inflated by a frame-clustered validation score. Each
validation frame contributes one maximum standardized error across all sparse
identities and coordinates; the nineteenth of twenty scores sets the 90%
simultaneous-frame scale. Future coverage and NEES remain descriptive because
one clip does not establish exchangeable calibration.

The primary diagnostics are hidden-identity RMSE, hidden FDE, hidden symmetric
Chamfer, horizon-resolved RMSE, predictive coverage, and NEES. The route advances
only if the physical model beats both kinematic baselines and the guarded update
improves hidden identities on clip one without using observed identities in the
score.

## Next evidence gate

A successful validation smoke authorizes implementation work, not a paper
claim. Before a claim-bearing run, the full dataset must be available under
explicit reuse terms and an object-level source/calibration/target split must be
sealed before target payloads are processed. The intended unit of inference is
the physical object, not frames or keypoints. The current upstream revision has
no license file, and the future evaluator is an upstream tracker rather than
independent ground truth; neither limitation may be softened in result text.

The machine-readable lock is
`protocols/trackdeform3d_public_smoke_v1.json`. Admission lives in
`bayesian_phystwin.trackdeform3d_adapter`; the leakage-safe predictor and scorer
live in `bayesian_phystwin.trackdeform3d_smoke`. The command-line flow creates
separate `prediction_input.npz` and `evaluator_target.npz` carriers before the
prediction process starts.

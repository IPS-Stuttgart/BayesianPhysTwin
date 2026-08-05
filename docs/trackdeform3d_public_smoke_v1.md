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

## First milestone

The first smoke uses only `dlo/chunk_1`, clip zero. Admission reads NPZ headers
and hashes but no RGB-D, pose, mask, keypoint, or future values. The official
tracker then creates the evaluator trajectory in a separately sealed stage.

The temporal and identity boundaries are fixed before that trajectory is
scored:

- 300-frame clip zero;
- 60 prefix frames and 60 untouched future frames;
- 25% of frame-zero identities selected deterministically by spatial
  farthest-point sampling;
- observed identities may update the state during the prefix;
- only disjoint hidden identities are scored in the future.

Future RGB-D and future TrackDeform3D keypoints are evaluator-only. They may not
construct a prediction.

## Arms

The smoke compares exact persistence, constant velocity, a shared graph
physical forward model, and the same model after a Bayesian prefix state update.
The physical arm must use the released robot poses and frame-zero topology, not
future keypoints. A zero or rejected update must return its selected physical or
persistence baseline exactly.

The primary diagnostics are hidden-identity RMSE, hidden FDE, hidden symmetric
Chamfer, horizon-resolved RMSE, predictive coverage, and NEES. The smoke advances
only if the physical model beats both kinematic baselines and the Bayesian update
improves hidden identities without using the observed identities in the score.

## Next evidence gate

A successful smoke authorizes implementation work, not a paper claim. Before a
claim-bearing run, the full dataset must be available under explicit reuse terms
and an object-level source/calibration/target split must be sealed before target
payloads are processed. The intended unit of inference is the physical object,
not frames or keypoints.

The machine-readable lock is
`protocols/trackdeform3d_public_smoke_v1.json`. The adapter entry points are
`inspect_trackdeform3d_chunk` and `deterministic_observed_identity_ids` in
`bayesian_phystwin.trackdeform3d_adapter`.

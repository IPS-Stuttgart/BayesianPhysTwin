# DeformMaster backend v1

This adapter defines the boundary a DeformMaster rollout must satisfy before it
can enter Bayesian-PhysTwin. It does not import DeformMaster and does not treat
the current public demo loader as a causal predictor.

## Why a separate producer contract is required

The public DeformMaster release loads `object_points` for every available frame
and passes the full sequence to the router as `tracks`. Its scene initializer
also computes the simulation offset from all ground-truth surface tracks. A
configuration such as `mpm.max_frames: 300` limits length but does not create an
observation/future split.

Those behaviors are appropriate for the released interactive demo and training
data path, but they cannot support a predictive comparison in which only an
early observation prefix is permitted. The release also states that full
training code is not yet available, so checkpoint target-object exclusion
cannot currently be reconstructed from public code alone.

## Required causal artifact

The runtime manifest binds:

- exact DeformMaster and producer repository revisions;
- checkpoint, configuration, and checkpoint-training manifest hashes;
- a sorted training-object roster that excludes the target object;
- disjoint observation and forecast frame ranges;
- router, initialization, and frame-offset input ranges contained entirely in
  the observation prefix;
- the full known controller-action range, separately from object outcomes;
- explicit false flags for future object tracks, RGB, depth, and outcomes;
- metres, seconds, a right-handed z-up frame, and persistent material identity;
  and
- raw rollout bytes sealed before future scoring.

The raw archive contains four arrays:

```text
driven_surface_positions_m      (T, N, 3)
zero_action_surface_positions_m (T, N, 3)
action_support                  (N,)
frame_zero_points_m             (N, 3)
```

The adapter converts these into the shared six-array `physical_rollout_v1`
contract. For fixed inputs, publication is byte-identical and validation
rederives every output array.

This bundle is a candidate, not a selector. Any source competence gate must
retain the incumbent physical archive byte for byte whenever the DeformMaster
candidate is rejected or unavailable.

## Commands

Seal a producer attestation before any future score is opened:

```bash
bpt experiment run materialize-deformmaster-backend seal-runtime \
  rollout.npz checkpoint.pt config.yaml training-data.json runtime.json \
  --source-revision SOURCE_SHA \
  --producer-repository OWNER/REPOSITORY \
  --producer-revision PRODUCER_SHA \
  --case-id CASE_ID \
  --target-object-id OBJECT_ID \
  --prefix-end-frame-exclusive 6 \
  --time-step-s 0.03333333333333333
```

Materialize and validate the portable candidate:

```bash
bpt experiment run materialize-deformmaster-backend materialize \
  rollout.npz runtime.json checkpoint.pt config.yaml training-data.json bundle

bpt experiment run materialize-deformmaster-backend validate bundle
```

## Current decision

The adapter and synthetic controls pass. The public DeformMaster release at
`c7b3510a38b3fccbfe12cc6557aaf58d9ea823dc` does not pass the causal producer
gate, so no released checkpoint or future trajectory has been scored through
this path. A valid next producer must slice router inputs at the prefix, derive
initial placement from permitted frames only, bind checkpoint training
provenance, and export a prediction seal before outcome access.

This is an integration-readiness result, not a DeformMaster reproduction or
state-of-the-art result.

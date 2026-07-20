# Deform360 Selective Full-Field Virtual Sensing V1

Status: locked before download or media access for the 12 selected objects.
The temporal rule was added before any selected-object access because the
initial lock specified 76 frames but omitted their deterministic source window.
The inherited target pipeline, automatic-mask provenance, comparator builders,
and all execution hashes were likewise added before selected-object access.

## Target-free operational amendment

After all frame-zero backbones had been sealed, but before any sparse
measurement, virtual-sensor prediction, future target, or metric was opened,
the first measurement launch exposed a runtime defect in the direct camera
planner. The frozen specification evaluates all `C(32, 8) = 10,518,300`
subsets in Python and recomputes ray-angle scores even when the first three
lexicographic score components already rule a subset out. No case artifact was
written by the interrupted launches.

The frozen builder and protocol remain byte-for-byte unchanged. The
operational runner
`scripts/remote/run_deform360_selective_measurement_prediction_accelerated.py`
temporarily replaces only the exhaustive plan function with an exact batched
implementation from
`bayesian_phystwin.deform360_exact_camera_subset`. It enumerates combinations
in the same lexicographic order, evaluates the first three integer score
components in NumPy, and computes the unchanged median ray-angle component
only for tied candidates. Strict-greater replacement preserves the original
first-tie rule.

The accelerator is fail-closed against the frozen raw-camera builder SHA-256.
Tests compare it with the literal exhaustive implementation over randomized
support and geometry, batch-boundary ties, a full observation plan, and the
all-tie case. On the first real 32-camera case it produced the 16-center plan
in 6.17 seconds. This is an execution-equivalent optimization, not a method,
threshold, cohort, or information-boundary amendment.

### Prediction-prefix cadence repair

After the first prediction shard was sealed, decoding the second shard exposed
a systematic media-staging defect: FFmpeg 7 had materialized 49 rather than the
locked 58 selected source frames for every case in that shard. The source
episodes contain the complete ranges. No prediction artifact, future target,
particle track, or metric had been opened for an affected case.

The target-free repair utility
`scripts/remote/repair_deform360_selective_prediction_prefix_cadence.py`
re-encodes the same locked half-open source ranges at explicit 30 Hz CFR,
requires exactly 58 decoded frames per camera, and reseals the staged manifests.
It then rebuilds each persistence backbone from its already-sealed frame-zero
points and requires every scientific array in the old and new backbone archives
to be bit-exact. The original artifacts are retained; each repaired case records
both superseded and repaired hashes in
`prediction_prefix_cadence_repair_seal.json`, together with explicit assertions
that no target or future outcome was read. This fixes only prefix
materialization and does not alter the protocol, cohort, method, thresholds, or
information boundary.

The same FFmpeg 7 cadence behavior also affects the 23-frame tail that becomes
authorized only after the prediction-cohort seal. Before any selected future
was opened,
`scripts/remote/stage_deform360_selective_authorized_future_cadence_safe.py`
was added as a fail-closed wrapper around the frozen target-staging script. It
requires the frozen script hash, uses the same exact source-frame indices at
explicit 30 Hz CFR, verifies 58 decoded prefix frames and 81 decoded full-window
frames, and leaves the target reconstruction and evaluation unchanged.

The executable lock is
`configs/sota/deform360_selective_virtual_sensing_v1.json`. Its canonical
checksum is
`d231b0eb06e724ec131c569e88ca482bca44b3340f5e13d11f973feab0cc53dd`.
The lock is validated by
`bayesian_phystwin.deform360_selective_virtual_sensing_protocol`.

## Paper hypothesis

A small number of causal multiview camera tracks can update a dense deformable
state without training on the target object. The update should help points that
were never observed, persist into future frames after each observation, reject
identity-inconsistent tracks, and return the base trajectory exactly when the
measurement is not trustworthy.

The intended paper claim is:

> A training-free, correspondence-selective recursive discrepancy field turns
> sparse camera prefixes into improved hidden full-field forecasts across new
> filament, sheet, and volumetric objects.

This is a new online-assimilation setting, not an official Deform360 Table 4
comparison. Virtual sensing, RBF interpolation, Bayesian filtering, and
conformal risk control are established ideas individually. The contribution is
their causal, simulator-agnostic composition with raw multiview tracking,
hidden-point evaluation, future-only scoring, and exact fallback for deformable
digital twins.

## Development evidence

The method was developed on 27 already-open Deform360 episodes from five
physical objects. Sixteen frame-zero centers were tracked from eight cameras at
frames 19, 38, and 57. Every center was permanently removed from the score;
only frames after an update were scored.

The selected physical/persistence pairwise-gated arm improved over the sealed
physical prior by 28.83% in hidden identity RMSE and 24.15% in hidden symmetric
Chamfer. It won 24/27 and 23/27 episodes, respectively. Against the selected
raw backbone, it improved by 15.74% and 14.10%; all five object means improved,
and both object-cluster intervals excluded zero.

The simpler persistence-only pairwise arm reached 7.646 mm hidden identity
RMSE and 6.787 mm hidden Chamfer, versus 9.357 mm and 8.243 mm for persistence.
That is an 18.29% and 17.66% reduction and improved all five development
objects. The object-cluster intervals are `[-2.344, -0.808] mm` and
`[-2.295, -0.566] mm`. It is the prospective primary arm because it requires no
target-fitted physical model. The physical/persistence selector remains a
secondary composition experiment wherever a physical prediction was sealed
without target outcomes.

These numbers selected the method and are not confirmatory evidence. The
archived development summary is under
`results/sota/diagnostics/deform360_raw_alltracker_pairwise_explicit_arms_f98fcfa/`.

## Frozen method

For each episode, the frozen action-only rule chooses an 81-frame raw window.
It maximizes mean gripper-centre path length weighted by endpoint closure
confidence over candidate starts 8, 14, 20, and so on, breaking ties at the
earliest start. It reads released robot actions and apertures only, never object
geometry, tracks, tactile data, or outcomes. The official five-frame tracking
tail leaves the 76 evaluation frames.

For each resulting 76-frame episode:

1. Select frame-zero object masks with the pinned generic SAM2 selector, then
   reconstruct material points without reading later dense states.
2. Select 16 deterministic farthest-point centers visible in at least two
   cameras.
3. At updates 19, 38, and 57, run the pinned AllTracker model on exactly RGB
   frames `[0,u]` in eight cameras and robustly triangulate the centers.
4. Compare pairwise center distances before and after observation. Accept the
   largest deterministic consensus clique only if it has at least 9 centers and
   70% of available support under the frozen 30 mm plus 10% strain tolerance.
5. Update the frozen global plus local Student-t RBF discrepancy state on the
   accepted clique and add its decayed field to persistence.
6. On insufficient support or correspondence rejection, emit persistence
   bit-for-bit.
7. Before any target opens, also seal an ungated RBF control and an independently
   refit unordered CPD control from the same sparse measurements.

The belief parameters, tracker revision and checkpoint, camera count, center
count, update frames, correction cap, correspondence thresholds, SAM2 source,
and control hyperparameters are all in the lock. Held outcomes may not change
them.

The target is inherited from the development pipeline: the exact eight sealed
measurement cameras, the same frame-zero splat, a 512-point strict visual-hull
minimum on a 120-cubed grid, 500/250 Splatfacto iterations, pinned CoTracker3,
official expected depth with URDF gripper masks, and deterministic Deform360
point fusion. The 81 raw frames yield 76 identity-preserving target frames.

## Prospective cohort

Only top-level directory names from the pinned Hugging Face snapshot were used
to select the cohort. None of these object paths existed on either compute
server before the lock.

| Stratum | Objects | Episodes |
| --- | --- | --- |
| Filament | thread; jump rope; climbing rope; hemp rope | 5/2; 1/4; 0/7; 8/0 |
| Sheet | yellow cloth; handkerchief; wipe cloth; wrap paper | 9/3; 0/4; 8/5; 2/8 |
| Volumetric | doll; ball; rabbit; frog | 0/1; 5/2; 8/9; 5/2 |

The two episodes per object are the first two SHA-256 ranks among IDs 0 through
9 under the fixed seed `deform360-selective-virtual-sensing-v1`. Successful
episodes are never selected or discarded using outcomes. Failed objects are
not replaced.

The six objects reserved for the earlier frame-zero-only experiment are not
used. Their seal therefore remains intact.

## Information order

1. Commit and publish the code, tests, method lock, and source hashes.
2. Download only the 12 named objects at the pinned dataset revision.
3. Select the action window, then expose only its RGB prefix through frame 57
   to the prediction-facing process.
4. Build frame-zero geometry, calibration assets, and causal RGB measurements.
5. Write and hash the frame-zero backbone, every measurement, and every
   virtual-sensor prediction artifact.
6. Account for all 24 episodes with exactly one validated prediction or
   target-free quality failure. Recompute the 9-object and 3-per-stratum gate.
7. Revalidate every underlying artifact and seal the complete cohort.
8. Only then propagate future masks, reconstruct dense futures, build particle
   tracks, and score the authorized successful cases.
9. Report all successful episodes, all object means, all strata, all failures,
   and every predefined comparator.

Any future dense reconstruction, particle track, or target metric opened before
the corresponding prediction hash invalidates that object without replacement.

## Confirmation gate

Episodes are averaged within object and objects receive equal weight. Both
co-primary metrics must pass every gate:

- at least 9 evaluable objects and at least 3 in every stratum;
- at least 10% object-balanced improvement over persistence;
- an object-clustered 95% interval whose upper endpoint is below zero;
- one-sided exact object-level sign-test `p <= 0.05`;
- no stratum mean regression;
- no object regression larger than 10%.

The conjunction is an intersection-union claim, so success requires both hidden
identity RMSE and hidden symmetric Chamfer. A failed gate is a prospective
negative or mixed result. It cannot trigger a replacement arm, threshold, or
cohort.

## Paper threshold

Passing the locked gate would be worthy of a strong paper when accompanied by:

- the selected physical-prior composition on the existing development panel;
- persistence, raw-backbone, ungated-RBF, CPD, and corruption controls;
- risk-coverage and exact-fallback plots;
- qualitative hidden-point trajectories and failure cases;
- runtime and camera/center-count ablations fixed on development data.

It would establish prospective cross-object virtual sensing, not direct
Deform360 leaderboard superiority. Official split and evaluator parity remain a
separate requirement for any state-of-the-art wording.

# Action-Anchored Prefix State Source Result v1

Date: 2026-07-26

Status: source gate failed; stop this route.

## Question

The frozen source smoke tested whether the failed Deform360 Warp adapter was
mainly missing an action-anchored endpoint velocity. It combined two causal
prefix hulls with registered measured-gripper motion, estimated a shared
observation-velocity bias at the contact nodes, and injected the resulting
graph-regularized velocity into the unchanged official Warp simulator.

This adapter was motivated by the hard end-effector path in TrackDeform3D. It
is new Bayesian-PhysTwin code, not an evaluation of the upstream RGB-D method.

## Result

The target-free state gate passed. Chain orientation was decisive, every
rollout was finite, and the fused field's maximum initial speed was
`0.308 m/s`, below the frozen `1.5 m/s` limit. The inferred shared
observation-velocity bias was
`[-0.0288, -0.0044, 0.1379] m/s`.

It did not improve the untouched source future:

| Arm | Mean Chamfer | Late Chamfer | p99 edge strain |
| --- | ---: | ---: | ---: |
| Exact persistence | 47.487 mm | 36.607 mm | 0.000 |
| Official Warp, zero velocity | 47.516 mm | 38.429 mm | 0.133 |
| Camera-topology velocity | 48.005 mm | 39.640 mm | 0.133 |
| Action-only harmonic velocity | 47.460 mm | 38.457 mm | 0.134 |
| Bias-aware action-anchored velocity | 47.759 mm | 39.175 mm | 0.133 |

Relative to zero-velocity Warp, the fused arm changed mean Chamfer by
`-0.51%` and late Chamfer by `-1.94%`, where negative means worse. Relative to
exact persistence it changed them by `-0.57%` and `-7.02%`.

The action-only harmonic arm was effectively tied on the mean: `0.12%` better
than zero-velocity Warp and `0.06%` better than persistence, while remaining
worse late. These changes are far below the frozen 5% transfer gate.

## Interpretation

Measured attachment motion is informative enough to identify a plausible
endpoint velocity, but that velocity is not the missing long-horizon state in
this bimanual rope case. The simulator rapidly loses the small action-only
initial-velocity advantage, while the camera-derived deformation component
slightly worsens the rollout. This agrees with the earlier state-propagation
audit: double-action constraints can contract or redistribute prefix state
corrections.

The result rejects this fixed combination:

- two adjacent visual-hull prefix states;
- a shared translation-bias nuisance estimated at two action anchors;
- rank-full chain-Laplacian velocity smoothing with weight 4;
- hard measured-controller velocities at the registered contact nodes;
- the archived leave-one-source Warp candidate 21.

It does not reject TrackDeform3D as an RGB-D tracker, nor does it reject
action anchors for contact association, online filtering, or longer observed
prefixes.

## Decision

Do not run a multi-episode or fresh-object evaluation of this endpoint-state
adapter. Do not tune the Laplacian weight, speed gate, or velocity blend on
this opened source outcome.

The useful conclusion is narrower:

> Replacing the zero endpoint velocity with one topology-preserving,
> action-anchored estimate does not recover the Deform360 physical-model
> headroom. The next method must update a persistent state/discrepancy belief
> over time or improve the physical response itself.

The existing baseline-relative recursive Bayesian belief remains the stronger
research lead. TrackDeform3D-style hard anchors may still enter that filter as
independent contact evidence, but they should not be promoted as a standalone
state initializer.

## Boundary And Provenance

- Source case: `002-rope-silk`, episode 5, selected by locked metadata.
- Prefix hull frames: 7 and 8; branch at frame 8.
- Untouched scored source future: archived frames 14 through 411.
- The prediction input directory contained only `robot.npz` and the
  source-fitted contact-model JSON.
- No future geometry, future tactile, score-bearing fit/grid JSON, held-v8
  artifact, or fresh target was available to prediction.
- Bayesian-PhysTwin prediction commit:
  `d73887c21fe5bcbc68ec6d94fd0d6c28557d7162`.
- Causal4D commit:
  `310943c382b412f033352872724fc2f9c330910f`.
- Official PhysTwin commit:
  `2b6630528141b9cba5a7677c8b88b2129b4a8390`.
- TrackDeform3D reference commit:
  `9920060a76f7d750f98e429bd1e0f172150c9ffa`.
- Prediction seal file SHA-256:
  `e3902b1d3743b55227c9db588543df4c44293ac728fbbad820c6a31f3c383e5f`.
- Result file SHA-256:
  `3e454e627ea11560948ef6c48a8202286646ab1ac9384925708d26ae789ecaea`.
- Archived evidence:
  `results/sota/diagnostics/deform360_action_anchored_state_source_v1/`.

This is one already-open source episode. It supports a mechanism stop decision,
not object-level transfer, calibration, confirmation, or a state-of-the-art
claim.

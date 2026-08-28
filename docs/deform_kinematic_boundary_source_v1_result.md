# Hard-position DEFORM boundaries: failed source screen

## Decision

Keep the existing paired DEFORM state update. The hard-position alternative
passes all ten implementation controls but fails five of seven registered
value checks. It is not promoted, retuned, or transferred to another object.
The incumbent, prior paired predictions, and upstream DEFORM files are unchanged.

The pre-outcome implementation is
`fee554262ac9d085270a8892f95924e911d3cd5a`; the protocol remains unchanged at
`configs/sota/deform_kinematic_boundary_source_v1.json`. This note and compact
result copies are post-outcome additions. Exactly one native CPU attempt ran.

## Matched results

Fourteen already-open DLO2 trajectories completed, with the registered design
case excluded from the thirteen equally weighted analysis trajectories. All
arms share two initial full states, frozen learned weights/readout, and known
future end-node trajectories. Each paired arm gets the same eight released 3D
prefix observations: nodes 2,4,6,8 at archive frames 41 and 49. Hidden nodes
3,5,7,9 are scored over forecast frames 50:170. This is not automatic RGB
tracking, independently measured robot actuation, or an official SOTA score.

| Arm | Coordinate L1 (mm) | Point RMSE (mm) | Late RMSE (mm) | FDE (mm) |
|---|---:|---:|---:|---:|
| Unchanged incumbent | 10.673 | 25.614 | 27.524 | 19.379 |
| Existing paired state update | 9.594 | 23.066 | 27.089 | 19.519 |
| Hard-position baseline | 10.707 | 25.671 | 27.638 | 19.452 |
| Hard-position paired update, primary | 9.636 | 23.120 | 27.190 | 19.572 |

Relative to the existing paired update, the primary changes coordinate L1 by
**+0.43%**, point RMSE by **+0.23%**, and late RMSE by **+0.37%**. Only 3/13
trajectories improve jointly in L1 and RMSE. The worst trajectory RMSE ratio is
1.020989. The frozen 10,000-replicate paired trajectory bootstrap gives an
RMSE-difference 95% interval of **[-0.021527, 0.126613] mm**. This does not
establish a systematic effect, and is conditional on one already-open object.

| Point RMSE (mm) | Early | Middle | Late |
|---|---:|---:|---:|
| Unchanged incumbent | 23.185 | 23.904 | 27.524 |
| Existing paired state update | 16.918 | 22.717 | 27.089 |
| Hard-position baseline | 23.222 | 23.931 | 27.638 |
| Hard-position paired update | 16.982 | 22.718 | 27.190 |

The hard-position paired arm improves on its own unupdated baseline by 9.94%
RMSE. Thus the sparse state update remains useful inside this alternate
boundary model, but the boundary change adds no demonstrated value over the
existing paired method. The diagnostic baseline itself worsens RMSE by 0.22%
relative to the unchanged incumbent.

## Gate and accounting

| Registered check | Result |
|---|---|
| At least 2% lower L1 than existing paired update | Fail |
| At least 2% lower RMSE than existing paired update | Fail |
| Non-increasing late RMSE | Fail |
| At least 8/13 joint wins | Fail: 3/13 |
| Worst trajectory RMSE ratio at most 1.05 | Pass: 1.020989 |
| RMSE-difference bootstrap upper bound below zero | Fail |
| Sparse update improves hard baseline in L1 and RMSE | Pass |

Prediction accounting: 14/14 ordinary successes; zero retained technical
failures, unsealable predictions, replacements, or omitted trajectories. All
fourteen forecasts were sealed before source future free-node truth was scored.
No target or held-v8 data was accessed. The design case remained excluded.

All ten controls passed: incumbent and old paired byte identity, unchanged
native rest initialization, exact hard-boundary zero-update restart, exact
float32 commanded clamps in both new branches, disabled readout identity,
zero-innovation identity, and both predeclared native/archive replay limits.
The unchanged CPU replay differs from the archived replay by at most
0.003636 mm per coordinate and 0.000407 mm coordinate RMSE. These small replay
differences are distinct from the native end-segment length projection.

The separate frozen arithmetic checker verified 675 bound source files, 28
prediction arrays, all 624 case/arm/horizon metric values, and the seven-check
decision. Maximum arithmetic difference is 1.07e-14. It performed no new
native execution and is not independent human review.

Pre-run verification: 551 local DEFORM tests and 23 exact remote-runtime tests
passed; changed-file Ruff, focused MyPy, and diff checks passed. The run used
Python 3.10, Torch 2.0.1+cu118, NumPy 1.24.3, CPU and one thread, with CUDA hidden.

## Interpretation and provenance

Native DEFORM preserves each prescribed end segment's direction while
normalizing its length. The preceding control-only diagnostic measured a
maximum node-1 point displacement of 11.297 mm (10.869 mm per-coordinate
maximum). Hard clamping removes this discrepancy exactly, but does not improve
the hidden-node forecast. The control mismatch is therefore not demonstrated
to be the performance bottleneck on this fixed source screen. This neither
proves the native assumption physically correct nor rejects other contact or
actuator models. No globally inextensible-rod claim is made for the alternative.

Compact evidence is in `results/sota/deform_kinematic_boundary_source_v1/`.
The full run remains at
`/home/florianpfaff/source-only/deform-kinematic-boundary-source-v1/run-v1`
on gpuserver4090. A local copy was independently rehashed. Existing frozen
reference-centering failures are retained without retroactive amendment.

| Artifact | Identity |
|---|---|
| Lock canonical ID | `e6355398e4aee9912dcd2ea3b0ca6b7a25cd45ce58314f456a38b94286c0f9c4` |
| Lock file SHA-256 | `aaf28f9906ae5372a06fe84b032fabaebcbbc498226424f08b5b09aed33087f4` |
| Prediction seal file SHA-256 | `764781bf8ec1251b46b93afcc6c44e1634b66c44e2fb84af37637a17135f7221` |
| Prediction NPZ file SHA-256 | `f609212d260713b0e0ceb6e395ef35d6368fc93c6e302caf81d85e9c5063a999` |
| Result canonical ID | `675c89e7aeaa299055ce51f1ad496ebf9efdcbce34c05a74df05e17135bbc34f` |
| Second arithmetic canonical ID | `39ea177b9d541783c034e5aa0b0a06a5f732de72378908e829aa18b0bbcd5c39` |

This is a qualified negative ablation, not a new positive headline contribution,
calibrated UQ, independent confirmation, or SOTA result. No DLO1/DLO3 transfer,
DLO4/DLO5, official DLO3 evaluation, reserved/fresh Deform360, PokeFlex,
physical Causal4D, new recording, GPU, push, or main merge followed.

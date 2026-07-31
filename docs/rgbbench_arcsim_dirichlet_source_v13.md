# RGBench ARCSim Source Accuracy v13

## Purpose

V12 established deterministic, finite, exactly actuated ARCSim dynamics over
the complete 16.355 s released fling without reading any point-cloud outcome.
V13 is the first accuracy screen. It uses the same already-declared
`green_tshirt/fling/01` source case and introduces no material, solver, control,
or timing selection.

This is a one-case advancement screen, not a benchmark or state-of-the-art
claim.

## Prediction Boundary

The `simulate` stage may read the released mesh, material metadata, measured
actuator trajectories, qualification evidence, and segmented point-cloud
filenames for timestamp alignment. It must not read a point-cloud coordinate.

The 27 evaluation timestamps are mapped to the nearest frozen 0.01 s ARCSim
step after adding the released camera delay and the unchanged 5 s preparation
plus 5 s wait. The maximum permitted alignment error is 5.0001 ms. A fresh
full-horizon replay must exactly reproduce the qualified final vertex array.
Only then are the 27 simulated states sealed in an NPZ artifact.

The `score` stage is the first operation allowed to open source point-cloud
coordinates. It computes the benchmark's real-to-sim L1 Chamfer distance from
the sealed prediction.

## Frozen Comparators

The screen binds the already-open v2 source artifact for this exact case:

| Comparator | Error |
|---|---:|
| Remeshed PyBullet physical baseline | 59.394 mm |
| Best frozen cross-fitted dynamic baseline | 49.643 mm |
| Published GarmentDynamics cell | 41.900 mm |

ARCSim advances only if it:

1. beats the remeshed physical baseline;
2. beats the best frozen dynamic baseline; and
3. improves over the published GarmentDynamics cell by at least 5%.

A pass authorizes a separately frozen, zero-tuning 27-case source protocol. A
failure closes this ARCSim route without calibration or target access. No
post-score parameter adjustment or second source-case screen is authorized.

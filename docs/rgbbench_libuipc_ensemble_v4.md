# RGBench LibuIPC Ensemble v4

## Motivation

The frozen v3 gate rejected a deterministic interpretation of the LibuIPC
CUDA backend. Its two 0.1 s replays differed by 0.0211515 mm RMSE and
0.287111 mm at the worst coordinate. V4 does not weaken or rerun that gate.
It treats independent replay spread as simulator uncertainty in a new
protocol.

## Target-Free Qualification

The already-open `green_tshirt/fling/01` source carrier is simulated through
the complete 16.355 s preparation, wait, and recorded-action horizon. Three
independent processes run on one explicitly selected physical GPU.

Shell thickness is not tuned. It is fixed to 1.1066527 mm by the released
0.255 kg mass, released 220 kg/m3 density, and the 1.0473845 m2 area of the
hash-bound source mesh. The runner recomputes this identity before simulation.

The operator monitors compute-process occupancy. Any unrelated process
observed on that GPU invalidates the qualification attempt rather than being
silently ignored.

Qualification requires:

- all three processes complete with finite, contract-preserving output;
- maximum pairwise endpoint RMSE at most 0.1 mm;
- maximum pairwise coordinate difference at most 1 mm;
- each driven shoulder vertex finishes within 20 mm of its target;
- mean cloth-vertex displacement from initialization is at least 20 mm.

The 0.1 mm replay-RMSE ceiling is 0.345% of the published 28.99 mm
GarmentDynamics error and 0.219% of the 45.65 mm public PyBullet error. Thus
the replay floor must be negligible at the benchmark scale, but need not be
exactly zero.

## Information Boundary

The run may read only the bound source mesh, source material configuration,
measured actuator trajectories, fixed contact indices, and simulator
outputs. It must not list or read segmented point clouds. Source accuracy,
calibration, and target outcomes remain forbidden.

Passing authorizes a separately committed one-case source accuracy protocol.
It does not authorize direct scoring, calibration access, or target access.
Failure closes this full-horizon arm.

# RGBench LibuIPC Ensemble v4 Result

## Decision

The target-free full-horizon qualification failed. No segmented point-cloud
filename, point-cloud coordinate, source accuracy outcome, calibration
outcome, or target outcome was read. The v4 LibuIPC arm is closed before
accuracy scoring.

## Result

Three independent 16.355 s replays completed on an exclusively monitored GPU.
All 9,865 vertices remained finite and the vertex contract was preserved.

| gate quantity | observed | frozen limit | result |
| --- | ---: | ---: | --- |
| maximum pairwise endpoint RMSE | 17.956 mm | 0.100 mm | fail |
| maximum coordinate difference | 198.432 mm | 1.000 mm | fail |
| maximum driven-pin target error | 41.177 mm | 20.000 mm | fail |
| minimum mean cloth displacement | 526.340 mm | 20.000 mm | pass |

Replay runtimes were 125.84, 122.30, and 122.74 seconds. The mean endpoint
variance was `1.0164e-4 m2`; the maximum coordinate variance was
`1.2831e-2 m2`.

## Interpretation

The 0.1 s v3 replay difference looked negligible, but that conclusion does
not transfer through a complete nonlinear contact rollout. Full-horizon
independent-process divergence is now comparable with the approximately
16 mm gap between the public RGBench PyBullet result and published
GarmentDynamics.

The soft position constraint also does not reproduce RGBench's fixed-point
actuation closely enough. Strengthening that one coefficient might reduce pin
lag, but it would not erase the observed evidence that this CUDA trajectory is
not a stable deterministic backbone. V4 therefore receives no accuracy run
and is not tuned against source outcomes.

The next public-backbone candidate must pass, before scoring:

- deterministic or negligible-spread full-horizon replay;
- hard or independently verified kinematic attachment;
- finite, contract-preserving dynamics;
- the same source-only information boundary.

The exact compact evidence is in
`results/sota/rgbbench_libuipc_ensemble_v4/summary.json`.

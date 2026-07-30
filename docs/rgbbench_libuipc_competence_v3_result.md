# RGBench LibuIPC Competence v3 Result

## Decision

The source-only competence gate failed. No RGBench point-cloud coordinate,
source accuracy outcome, calibration outcome, or target outcome was read.
The v3 accuracy arm is closed.

Both independent LibuIPC processes completed, preserved all 9,865 vertices,
and returned finite arrays. They were not byte-identical:

| diagnostic | value |
| --- | ---: |
| replay RMSE | 0.0211515 mm |
| mean absolute difference | 0.0138514 mm |
| maximum absolute difference | 0.287111 mm |
| differing scalar coordinates | 29,595 / 29,595 |

The frozen rule required exact equality, so the small magnitude does not
rescue v3.

## Runtime Amendments

The pre-outcome implementation was locked at `1173c306`. Three narrow runtime
amendments were then needed:

1. `01a32a7` corrected a source-digest key mismatch. Simulation had not
   started.
2. `b2811ae` copied immutable source arrays into writable pyuipc buffers.
   Simulation had not started.
3. `4be6aea` matched pyuipc's `(N,3,1)` vector layout, unpacked its
   current/rest geometry-slot pair, and made animation callback failures fatal.

The final result is bound to `4be6aea` and the compact evidence in
`results/sota/rgbbench_libuipc_competence_v3/summary.json`.

## Interpretation

LibuIPC's CUDA implementation contains parallel atomic accumulation paths.
The observed result is consistent with order-dependent floating-point replay
noise, although this diagnostic does not identify every source of the
difference. An unrelated GPU workload began after the pre-run idle check and
overlapped the final replays. Its effect on the numerical difference is not
identified. No selective rerun was used to override the frozen failure.

The replay floor is tiny compared with the approximately 29 mm published
GarmentDynamics source error and the approximately 16 mm gap left by the
public PyBullet backbone. It therefore does not show that the shell model is
inaccurate. It shows that a deterministic-Dirac treatment of this backend is
false.

Any successor must be a new protocol. It must:

- propagate an independent-replay ensemble as simulator uncertainty;
- require an exclusive-resource preflight and record process occupancy;
- use a target-free full-horizon stability and replay-spread gate;
- preserve exact fallback and technical-failure accounting;
- freeze the ensemble size and spread thresholds before reading accuracy
  coordinates;
- keep calibration and target garments sealed until source and calibration
  gates pass.

This successor cannot retroactively turn v3 into a pass.

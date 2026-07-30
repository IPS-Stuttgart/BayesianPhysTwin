# RGBench Codim-IPC CHOLMOD v6 result

## Decision

The target-free CHOLMOD competence gate passed. The backend is deterministic,
enforces the registered pin trajectories exactly, and reduced the mean
short-replay wall time by 27.64% relative to the frozen Eigen control. This
result authorizes a separately frozen full-horizon target-free qualification.
It does not authorize reading RGBench point-cloud coordinates or accuracy
outcomes.

## Provenance

- protocol: `rgbbench-codim-ipc-cholmod-v6`
- protocol SHA-256:
  `5b6501c3c9819a61b872789e7344e5c724374fd77150410c0e1a7bb0ecc8a660`
- implementation commit:
  `e546e6691b99c49d7ddf4918b374bd920f9dc2ed`
- Codim-IPC commit:
  `9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a`
- compiled runtime SHA-256:
  `ff79e65d9d6e898aa3987d12a08be79cf3873a9b004506db1e10ea74d930d8ac`
- gate artifact SHA-256:
  `05527a88cbf63a94a4eef1dcf3c8af9fdb03aaa343631238570f22c8550d4eb2`

The runtime identifies its compiled linear solver as `CHOLMOD`. The frozen
protocol also binds the exact control and build patches, patched source files,
CMake cache, runtime module, and system BLAS/LAPACK libraries.

## Target-free results

| Check | Result |
| --- | ---: |
| Complete isolated replays | 2/2 |
| Final arrays byte-identical | yes |
| Final-array SHA-256 | `f45e45...0b8fe` |
| Replay wall times | 83.541 s / 83.736 s |
| Frozen Eigen wall times | 115.820 s / 115.340 s |
| Mean wall-time reduction | 27.64% |
| Mean speedup | 1.382x |
| Maximum pin-target error | 0.0 m |
| Mean vertex displacement | 48.966 mm |
| Maximum vertex displacement | 71.608 mm |
| Vertices / faces | 9,865 / 19,555 |
| Steps per replay | 10 |
| Newton iterations per replay | 126 |
| Point-cloud coordinates read | no |
| Source accuracy outcomes read | no |

The observed 83.736 s maximum is below the preregistered 90 s ceiling. Both
replays produced the same final-array SHA-256:
`f45e45f5132708193c3f5244272bf5969c2ce2b7d2cba95d2c14658e1e70b8fe`.

## Scope

This is implementation, determinism, exact-control, and runtime evidence only.
The run used the released source mesh, source material metadata, and known
actuator trajectories. It did not read segmented point-cloud filenames,
point-cloud coordinates, or source/calibration/target accuracy outcomes.

The next admissible action is to freeze and run a longer target-free
qualification. Source accuracy remains sealed until that qualification passes.

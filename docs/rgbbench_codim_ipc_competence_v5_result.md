# RGBench Codim-IPC competence v5 result

## Decision

The target-free numerical competence gate passed. This result qualifies
Codim-IPC for a separately frozen longer-horizon test. It does not authorize
opening RGBench point-cloud coordinates or making an accuracy claim.

## Provenance

- protocol: `rgbbench-codim-ipc-competence-v5`
- protocol SHA-256:
  `2c769ffb4b92299d2e01c3400f43686a35e246cd1e3d176d522064be115c9660`
- implementation commit:
  `f2cea05c12bff7dc4696c8d60d1ee8bcda1801c8`
- Codim-IPC commit:
  `9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a`
- gate artifact SHA-256:
  `3e02c039129ea48a154405eac3ed1b9b55134e045a1d57174eb8c079de809f17`

The first deployment attempt at implementation commit `b4db1da` stopped before
simulation because the upstream Python driver constructed a log path from the
absolute command-line arguments. Its gate artifact records
`status=technical_failure` and one nonzero replay return code. No replay array
or accuracy outcome was produced. Commit `f2cea05` isolates the upstream
driver's working directory and argument vector; the corrected run used a new
output root and did not overwrite the failed carrier.

## Target-free results

| Check | Result |
| --- | ---: |
| Complete isolated replays | 2/2 |
| Final arrays byte-identical | yes |
| Final-array SHA-256 | `0ae68b...be623` |
| Maximum pin-target error | 0.0 m |
| Mean vertex displacement | 48.966 mm |
| Maximum vertex displacement | 71.608 mm |
| Vertices / faces | 9,865 / 19,555 |
| Steps per replay | 10 |
| Newton iterations per replay | 126 |
| Point-cloud coordinates read | no |
| Source accuracy outcomes read | no |

Both replay files have the same SHA-256:
`0ae68b631ea1c394075fe553ad84c165e53e4c4569d637ebbe4cfe41c0dbe623`.
The exact zero pin error confirms that the new node-index trajectory interface
uses projected Dirichlet controls rather than another soft attachment.

## Scope

This is implementation and numerical evidence only. The run used the source
mesh, source material metadata, known actuator trajectories, and target-free
simulation diagnostics. It did not read segmented point-cloud filenames,
point-cloud coordinates, or source/calibration/target accuracy outcomes.

The next admissible action is a separately frozen longer-horizon qualification
that tests whether exact reproducibility, finite dynamics, and exact control
tracking survive substantially more steps. Source accuracy remains sealed until
that gate passes.

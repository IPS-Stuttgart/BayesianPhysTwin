# RGBench Codim-IPC full-horizon v7 result

## Decision

The target-free full-horizon qualification ended in a technical failure before
the first replay completed. No RGBench point-cloud filename, point-cloud
coordinate, source accuracy outcome, calibration outcome, or target outcome
was read. The Codim-IPC CHOLMOD arm is closed without accuracy scoring.

## Provenance

- protocol: `rgbbench-codim-ipc-full-horizon-v7`
- protocol SHA-256:
  `0d8fc7af8f303b50fd171da3d286f7616fa5ec350d07dec0b24d8f39299beca2`
- implementation commit:
  `c7706aae253174857acd95c1cbb196e1110dbb32`
- Codim-IPC commit:
  `9c6cbe3a5bef09a967ca8d420056adfafdf1fc9a`
- gate artifact SHA-256:
  `c8b44cba46030070a3462fb6c82df56d836c43dce25745b63af62d6479e41577`
- replay-log SHA-256:
  `75d4781230595539ab1ada6ad27e4cf8021a66f582075567f4909b79f4a9d8f9`

## Target-free failure

The solver completed 47 of the requested 1,636 steps, then stalled on step 48.
At operator termination:

| Diagnostic | Observed |
| --- | ---: |
| Elapsed wall time | 1,164.17 s |
| Last Newton iteration on step 48 | 1,510 |
| Last Newton residual | `2.451676e-6` |
| Frozen Newton tolerance | `1.0e-6` |
| Tiny gradient-descent step reports | 1,491 |
| Last line-search alpha | `2.273737e-13` |
| Replay array written | no |
| Point-cloud coordinates read | no |
| Accuracy outcomes read | no |

The final twenty recorded Newton iterations had the same residual, remained
above tolerance, and continued to report a tiny gradient-descent step. The
operator therefore sent `SIGTERM` only to the replay process rather than spend
the rest of the registered five-hour wall-time budget on a non-progressing
solve. The frozen parent preserved a `technical_failure` gate carrier with
return code `-15`; it did not start replay 2.

This is not a formal failure of the five-hour runtime threshold because that
ceiling was not reached. It is a stronger implementation-level blocker:
the selected upstream nonlinear solve failed to make progress under the frozen
full-action source mechanics. The short deterministic v6 result does not
transfer even through the first 0.48 s of the complete trajectory.

## Interpretation

Codim-IPC supplied the two properties missing from the prior LibuIPC attempt:
hard projected actuation and byte-identical short replays. However, exact
control and short-horizon determinism are insufficient when the nonlinear
solver stalls under growing deformation.

Changing the tolerance, timestep, line search, material parameters, or
trajectory after observing this failure would define a new method and require
a new target-free protocol. This branch receives no such tuning and no
accuracy run. The actionable lesson for the main Bayesian-PhysTwin program is
to avoid replacing the working physical prior with a public cloth solver solely
because that solver is nominally higher fidelity; full-action numerical
competence must be established first.

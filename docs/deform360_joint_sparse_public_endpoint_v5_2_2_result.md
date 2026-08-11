# Deform360 joint-sparse public endpoint v5.2.2 result

Date: 2026-08-11

Status: terminal technical failure; source gate not evaluated; confirmation not
authorized.

## Scope

This was the frozen public-data endpoint execution for the already sealed v5.2
source forecasts. Deform360 supplies the real-world RGB, calibration, robot
state, and robot action measurements. No new recording or human approval was
required, and no operator choice was allowed.

PR #492 bound the exact FFmpeg 4.4.2 binary and replaced the unsupported
`-fps_mode cfr` invocation with the preregistered legacy `-vsync cfr` path.
All authoritative checks passed before the merged revision was deployed. Ten
outcome-free preflights then passed without creating the endpoint output root.

## Result

Two workers processed the fixed first object on each GPU shard. Both passed
video materialization and reached automatic SAM2 propagation. Each then failed
because one of its two reserved endpoint cameras returned an incomplete or
empty mask.

| Attempt | Successful cameras | Failed cameras | Reserved support | Result |
| --- | ---: | ---: | ---: | --- |
| `026-sock-cloth` | 12/14 | 2/14 | 1/2 | terminal technical failure |
| `031-cotton-cloth` | 9/13 | 4/13 | 1/2 | terminal technical failure |

No endpoint geometry, depth reconstruction, source score, or source-gate result
was produced. The remaining eight registered objects were not run after the
terminal stop.

## Decision

The frozen processing lock declares technical failures terminal, forbids
implicit replacement, and forbids retry after endpoint inspection. The source
gate also requires complete aggregate evidence. Therefore this execution cannot
authorize confirmation, and the failed reserved cameras must not be replaced by
the many non-reserved cameras that happened to succeed.

This is an operational negative result, not evidence that the guarded Bayesian
update improves or regresses prediction. It identifies the strict two-reserved-
camera SAM2 endpoint contract as the current bottleneck. Any future route must
be a new, prospectively versioned protocol with a target-free probabilistic
support model or exact fallback; it cannot reinterpret this run.

## Provenance

- merged implementation: `ddbb582e53ec273b3e7057eabe41a9dc61b629cb`
- execution lock ID:
  `76b74483790ace51d642889be2e3dbb22149e30f7919b5855a18066434e25189`
- processing lock ID:
  `1ec041288fd9f5564442600d475de28e9c0638a8f702ece8bb477f067b3d1d4f`
- source prediction plan ID:
  `b3835574d9ab47f20f1d529bbb2dca4987af02959b02307074ad2cd8082372f2`
- prediction batch ID:
  `c62d475cc298dafe1d9c0ec30a242faffe36f5ab3ee5fc3d4b578918dbea1aa8`
- prediction receipt ID:
  `f7c6b10222f8c8d9089c5c9ee1f1cde58ae54dd57b29f7dbbd7ce2bfddd959d7`
- first terminal carrier SHA-256:
  `5881b60c97a99d7bb4bbd5c52900f2be1d31b062496a80d22f6fa5ca293dc19c`
- second terminal carrier SHA-256:
  `8ce0f9935fb55f1007266222b0d3203cb881d7d25f1bb5b0998c635475ac7ca0`
- archived terminal carriers:
  `results/sota/deform360_joint_sparse_public_endpoint_v5_2_2/terminal-failures/`
- compact result: `results/sota/deform360_joint_sparse_public_endpoint_v5_2_2/summary.json`
- compact result SHA-256:
  `f5c38d8e70e1ee6cfec0e52ceab052df67ed60ebf3870756ec39e81a7f46e609`
- confirmation payloads opened: false
- target outcomes used: false
- held-v8 artifacts accessed: false

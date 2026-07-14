# PokeFlex Public Preflight V1

This milestone records the first outcome-free PokeFlex compatibility audit on
2026-07-14. It contains no raw PokeFlex data, prediction outcomes, or target
metrics.

## Scope

- Interaction: `poking`
- Object: `3dPrintedBunny`
- Completed and integrity-checked archives: 7
- Selectively staged files: 7 robot logs, 42 camera calibrations, 1,117 OBJ
  meshes
- Staged size outside Git: 2.2 GB
- Readiness config SHA-256:
  `256f6c0585a1eb592583b0a0c017e116baed9126f12119e80f866cd174b58070`

The public archives use zero-padded string frame IDs. The schema adapter was
corrected to accept this documented representation before the real preflight
was evaluated.

## Result

The preflight passed. A metadata-only hash split assigned five development
takes, one calibration take, and one sealed target take.

Enabled gates:

- factual geometry continuation;
- cross-take interventional evaluation;
- pose/wrench contact-candidate generation.

Disabled gates:

- material-identity-dependent metrics;
- timestamp-based delay inference;
- command-versus-measured separation;
- explicit-contact abduction;
- nominal 90% session-level conformal calibration.

Preflight result SHA-256:
`56ff6606c3c90234f5945c23fa3999c45cc4490a0d6530b2086c79f80018b89a`.

## Verification

On `gpuserver6000`, the targeted adapter suite passed 8 tests. The complete
isolated suite passed 433 tests with 4 skips using PyRecEst 2.4.1. The raw ZIPs,
active downloader, and unrelated server worktree were not modified.

## Boundary

This milestone opens a source-only geometry and intervention-QA phase. It does
not establish a PhysTwin fit, a Bayesian improvement, material correspondence,
contact ground truth, calibration, or individual counterfactual validity. The
metadata-selected target remains sealed until a source backend and source gates
are frozen.

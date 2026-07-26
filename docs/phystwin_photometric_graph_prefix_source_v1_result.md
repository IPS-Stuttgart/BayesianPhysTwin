# PhysTwin Photometric Graph Prefix Source v1 Result

## Decision

**Gate failed; exact fallback.**

The one authorized run used the locked implementation at
`32ec07b61c2be89f64ba6a62caef606ce53c5c4a`. It read only the three released
RGB views and cloth masks for frames 114 through 120 of the already-open
`single_lift_cloth` source interaction. It did not read manual identities,
future images, future point clouds, future metrics, or any held-v8 artifact.

## Prefix Result

| Quantity | Result |
| --- | ---: |
| Baseline held-out prefix RGB RMSE | 0.0451818 |
| Candidate held-out prefix RGB RMSE | 0.0451855 |
| Relative improvement | -0.00815% |
| Absolute improvement | -0.00000368 |
| Required relative improvement | +2.0% |
| Required absolute improvement | +0.0005 |
| Fit correlation groups | 274 |
| Validation correlation groups | 206 |
| Valid rendered pixel fraction | 3.129% |
| Selected graph weights | 12 exact zeros |
| Maximum selected node correction | 0.0 mm |

The early-prefix fit converged to a small coefficient vector with L2 norm
0.02338 before the held-out gate. That field did not transfer to the final
three prefix frames. The gate therefore emitted the original graph trajectory
unchanged.

The run retained 93,388 of 123,304 Gaussian primitives and rendered all three
cameras at 212 by 120 pixels. The nonempty correlation-group counts rule out a
trivial no-support rejection. They do not establish that the released
Gaussian appearance is a calibrated observation likelihood.

## Interpretation

This result rejects the tested combination of:

- frame-zero Gaussian appearance;
- fixed 16-neighbor material attachment;
- translation-only Gaussian skinning;
- a persistent rank-4 graph-position field;
- a first-order rendered image response;
- and per-frame-camera-channel affine color nuisance.

It does **not** reject all render-aware state estimation. In particular, it
does not evaluate nonlinear Warp state injection, dynamic appearance,
occlusion-aware correspondence, higher-rank local deformation, or a learned
render residual. Those alternatives are not licensed for tuning on this
case, however. The preregistered future Warp rerun and object-disjoint source
panel are not run because their upstream competence condition failed.

The scientific lesson agrees with the automatic tracker evidence: the
remaining manual-prefix headroom cannot be recovered merely by replacing
generic tracking with this simple material-attached photometric linearization.
The strongest nonduplicative program remains a guarded online belief update
whose admission combines object-relevant observation evidence with
physical/action support and exact fallback.

## Evidence

- Frozen protocol:
  `configs/sota/phystwin_photometric_graph_prefix_source_v1.json`
- Full source report:
  `results/sota/phystwin_photometric_graph_prefix_source_v1/report.json`
- Report SHA-256:
  `984af7be18f525dfc1d68740f3547ff6e973144bf459fc44be041c678ec82e5a`
- Prefix completion seal SHA-256:
  `d52bab8ec2a371c1ddcc2542517493229d51e2a7000ff6dfbd700739f0200b94`
- Uncommitted remote array carrier SHA-256:
  `0cee8708a80811b898c5bd34530b696888adab72d145336a5b696085822c44e7`

The focused solver suite passed 6 tests locally and on native POSIX, and Ruff
passed. The local full suite could not collect because the host system SciPy
binary requires NumPy below 1.25 while the active Python has NumPy 2.2.6; this
is an environment ABI failure unrelated to the new module.

# Follow-up v2: passive prefix revision helps; nonlinear advantage remains small

## Outcome

The registered uncertainty/prefix audit completed in
[Actions run 34259506949](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34259506949)
on `gpuserver4090`, execution commit `aba2b7c920cfaea8f0ae7e9aadc5bbd34ce31590`.
Numerical tests, cache checks and the empirical audit passed. The downloaded
caches were also evaluated in a second local execution: maximum prediction
difference across all eleven methods was **9.795e-14 m**. Separate archive and
metric checks verified the hashes, trajectory rows, NLL, coverage, variance
finiteness and exact clamped-node preservation.

**The GP with calibrated cross-time covariance and passive prefix conditioning
reduces suffix coordinate L1 from 7.326557 to 6.902841 mm (5.7833%), winning all
8/8 validation trajectories.** It also beats a fitting-data-tuned prefix-bias
corrector by **4.3827%, with 8/8 wins**. No new robot experiment, active probing,
simulator retraining, source-test access or official-evaluation access occurred.

## What changed relative to v1

The simulator checkpoint, 40 fitting trajectories, eight validation trajectories,
GP kernel/features and mean shrinkage remain unchanged. The export exactly
reproduced v1's full-forecast values: hybrid 7.912029, ridge 7.240300 and GP
7.122158 mm.

For adaptation, each method receives the same ten recorded free-node position
frames at forecast steps **5, 10, ..., 50**. These are existing dataset
coordinates, with no additional artificial observation noise; this does not
validate camera-derived or noisy prefix observations. Known future clamped-node
motion remains available, as in v1. Forecast scoring uses **only steps 51--498**,
448 steps strictly after the observations.

**Do not compare 6.902841 mm directly to v1's 7.122158 mm:** their scoring windows
differ. The matched unchanged-GP comparator on the new suffix is 7.326557 mm.

Covariance and deterministic bias parameters were estimated from five-fold
trajectory-level cross-fitting of the 40 fitting trajectories. Feature scaling,
anchors and residual weights were refit within each fold. Validation suffix
truth did not select parameters or influence predictions. These folds hold out
the residual learner only: the frozen simulator historically trained on the fit
panel, and its checkpoint historically used validation for selection.

## Matched suffix accuracy

Metric: equal-trajectory mean absolute world-coordinate error over all 12 nodes
and the 448-step suffix, in mm. Four clamped nodes are identical across methods.

| Predictor | Receives ten prefix observations? | Suffix L1 (mm) |
|---|:---:|---:|
| Frozen DEFORM physics-plus-GCN hybrid | No | 8.126430 |
| Existing ridge residual | No | 7.443143 |
| GP residual, unchanged | No | 7.326557 |
| GP + unweighted mean prefix-error correction | Yes | 7.759254 |
| GP + fit-tuned shrinkage of prefix-error correction | Yes | 7.219240 |
| Linear/ridge covariance conditioning | Yes | 6.993554 |
| **GP covariance conditioning** | **Yes** | **6.902841** |
| GP conditioning with cyclically mismatched prefixes (invalid control) | Wrong correspondence | 7.678543 |

Every predefined method, including the unsuccessful bias correction and both
wrong-prefix controls, is retained in the artifact and
[`results/trajectory_metrics_wide.csv`](results/trajectory_metrics_wide.csv).
The hybrid-to-adapted comparison mixes extra prefix information with adaptation;
the matched prefix-bias and conditioned-ridge comparisons are more informative.

| GP conditioning minus comparator | Mean difference (mm) | GP wins | Descriptive paired 95% interval (mm) |
|---|---:|:---:|---:|
| Unchanged GP | -0.423716 | 8/8 | [-0.774488, -0.166162] |
| GP + fit-tuned prefix bias | -0.316399 | 8/8 | [-0.640626, -0.093848] |
| GP + unweighted prefix bias | -0.856412 | 8/8 | [-1.307047, -0.413791] |
| Conditioned linear/ridge model | -0.090713 | 7/8 | [-0.171309, +0.002740] |

Intervals use 5,000 paired trajectory bootstrap resamples. Eight trajectories
belong to one physical object, not eight independent objects. The incremental
nonlinear GP gain over conditioned ridge is **1.2971%**, and its interval crosses
zero. Thus the strongest present result is the value of covariance-aware
revision, not proof that nonlinear GP covariance is indispensable.

## Predictive uncertainty

These are marginal Gaussian working-model scores on **free-node local-frame
coordinates of the suffix**. NLL uses metre units and may be negative. Lower NLL
is better. Coverage is coordinate-marginal, not joint trajectory coverage.

| GP mean / uncertainty variant | NLL (nats/coordinate) | Nominal 95% coverage | Mean interval width (mm) |
|---|---:|---:|---:|
| Unchanged mean, uncalibrated variance diagnostic | -2.523255 | 85.8875% | 52.3770 |
| Unchanged mean, constant fit-only error variance | -2.809529 | 96.2995% | 75.6668 |
| Unchanged mean, calibrated GP covariance | -2.833020 | 95.4404% | 72.1459 |
| Prefix-conditioned mean and covariance | -2.898830 | 95.5613% | 68.3453 |

Calibration repairs the substantial aggregate undercoverage of the raw variance
diagnostic. However, calibrated GP versus same-mean constant variance improves
NLL by only **0.023492 nats/coordinate**, wins **4/8** trajectories, and has a
paired interval **[-0.067850, +0.018416]** for the NLL difference. This is not
strong evidence that query-dependent GP marginal variance alone outperforms a
well-estimated constant variance.

The pooled 95.56% coverage after conditioning also hides heterogeneity:
individual trajectory coverage ranges from **86.13% to 99.81%**. Do not describe
the model as uniformly calibrated across trajectories. CRPS is retained too:
constant-variance GP 8.411569 mm, calibrated GP 8.313942 mm, conditioned GP
7.860246 mm. The raw-variance CRPS is 8.248615 mm, despite its poor coverage and
NLL; there is no claim that every calibration metric improves simultaneously.

## Mechanism and interpretation

The update uses the covariance between observed-prefix and unobserved-future
residuals. Gaussian conditioning transfers measured prefix innovations into the
future mean. A zero cross-time covariance gives exactly zero mean update, while
cyclically assigning another trajectory's prefix worsens mean suffix error to
7.678543 mm. The latter is a deliberately invalid negative control, not a
competitor. Both GP and linear predictions were verified invariant to arbitrary
changes in unobserved suffix targets.

The covariance is empirically calibrated as q_c(a C / mean_diag(C_oof) + b I),
with two positive components per node and coordinate-specific fit-only error
scales. It is **not an untouched or fully Bayesian posterior**. Cross-coordinate
and cross-node covariance, joint calibration, simulator-state restarting,
camera-noise robustness and cross-object transfer were not tested. Gaussian
conditioning itself is standard; this experiment does not establish a unique
Bayesian or standalone methodological novelty claim.

A supported paper-facing claim is: **a trajectory-calibrated residual GP can use
a sparse, passively observed motion prefix to revise a physical-twin forecast,
improving on its unchanged mean and a training-tuned prefix-bias correction on
the historical DLO2 development panel.** No fresh confirmation or official
benchmark result is claimed.

## Evidence identities

- Query export: run `34258802512`, artifact `10069073833`.
- Query archive SHA256: `0f9cd2b454fd7c0798cc4979dfa064c85ee6eba4b799fd0c78a7da0ba5bf40f2`.
- Audit: run `34259506949`, artifact `10069333063`, name `deform-gp-v2-audit-34259506949`.
- Audit archive SHA256: `5445e71933e1878f31f350bb89232edeb6c50babcc0256e181aade43cc0a94eb`.
- Original report SHA256: `0bbe46fbf6260717175f0a52017244517f32867a7a2ba5bdcce31ea9a8c5f687`.
- Prediction SHA256: `c37f5f8c2854a82a84ffdd7f78cee8ec8de4b03062c4d9d7ebfa4006047ed844`.
- Calibration SHA256: `48db11490d2a69845a78285e296fa1229ef767180398a29cead5fd1500c8dac9`.

The full audit artifact retains the original report, predictions and variances,
fold errors/covariance matrices, fit-only parameters, seals, per-trajectory
accuracy/probabilistic CSVs, source and execution identity. The independent
local repetition used the same implementation on the exported native caches;
it was not an independent raw-data or simulator rerun. All additions remain on
the research branch; no merge or official-evidence replacement was performed.

# Positive Opened-Data Evidence: Sparse Residual State Propagation

**Status: promising exploratory result, not fresh confirmation or official SOTA.**

The original successful DEFORM result is unchanged. The new experiment asks a
different online question: can eight permitted sparse prefix observations improve
future predictions for material identities not observed during that prefix?

## What Changed

Keep the original physical trajectory and its learned readout correction. Infer
the remaining pose residual from observations minus that full incumbent readout;
infer a velocity increment from the residual's slope over 80 ms. Interpolate these
increments along material-node index, with exact zero at the actuator nodes.

Clone DEFORM's full prefix endpoint state, preserving twist, material frame, and
previous positions. Apply the pose/velocity increment and roll forward through the
unchanged native dynamics. The new readout is

```text
incumbent + (updated physical rollout - unchanged physical rollout).
```

This preserves the original learned correction instead of double-counting it or
discarding it. Zero update returns the incumbent exactly. This is an empirical
state/readout coupling, not yet a calibrated Bayesian filter, new contact model,
or universally non-worsening guard.

## Main Result

Use the 13 non-design trajectories from the already-open DLO2 archive; all come
from one physical object. Observe four nodes at each of two prefix times, then
score four disjoint nodes over a 1.2 s future. `103.pkl` is reported separately in
the artifact, not included in the aggregate. Values are equal-trajectory means.

| Arm | Coordinate L1 (mm) | Point RMSE (mm) | Joint wins / 13 |
| --- | ---: | ---: | ---: |
| Unchanged physical DEFORM | 11.882 | 28.350 | 0 |
| Unchanged successful incumbent | 10.673 | 25.614 | reference |
| Matched sparse readout persistence | 13.943 | 33.000 | 0 |
| Sparse physical pose reset | 11.017 | 26.212 | 4 |
| Sparse physical pose/velocity reset | 10.790 | 25.818 | 4 |
| Full-prefix physical pose reference | 10.994 | 26.168 | 4 |
| Full-prefix physical pose/velocity reference | 10.824 | 25.888 | 4 |
| Incumbent + propagated pose residual | 9.854 | 23.537 | 7 |
| Incumbent + propagated pose/velocity residual | **9.594** | **23.066** | **9** |
| Same, fixed quarter gain | 10.204 | 24.439 | 10 |

The full-gain coupled update improves L1 by **10.11%** and point RMSE by
**9.95%**. Paired whole-trajectory bootstrap 95% intervals for the differences are
[-2.013, -0.270] mm L1 and [-4.898, -0.532] mm RMSE. The quarter-gain arm improves
4.39% / 4.59%, with 10 joint wins, 12 L1 wins, and 10 RMSE wins. These intervals
are exploratory and are not multiple-comparison-adjusted confirmation tests.

The larger update regresses on four trajectories, with worst RMSE regression
8.08%. Quarter gain reduces the worst RMSE regression to 1.67%, but does not
eliminate it. Exact no-update fallback is an implementation property, not proof
that every admitted nonzero update is safe.

## Horizon Limitation

| Point RMSE | Incumbent | Coupled pose/velocity |
| --- | ---: | ---: |
| First 0.4 s | 23.185 mm | 16.918 mm |
| Middle 0.4 s | 23.904 mm | 22.717 mm |
| Last 0.4 s | 27.524 mm | 27.089 mm |

The effect largely decays. FDE changes from 19.379 to 19.519 mm, a 0.72%
regression with an interval spanning zero. The result supports short-horizon
online state correction, not a persistent long-horizon cure. Periodic sensing
would be a new, separately budgeted experiment, not an extrapolated result.

## Separately Frozen Noise Follow-Up

After opening the parent result, freeze both existing pose/velocity gains and
simulate 16 measurement realizations per trajectory. Average within trajectory
before bootstrapping; repetitions are not new independent cases.

| Simulated observation error | Full-gain L1 / RMSE improvement | Quarter-gain improvement |
| --- | ---: | ---: |
| 1 mm independent coordinate noise | 10.05% / 9.92% | 4.40% / 4.58% |
| Same + 5 mm shared coordinate bias | 9.00% / 9.27% | 4.33% / 4.53% |

Full gain has 9/13 joint wins in the first condition and 8/13 in the second;
quarter gain has 10/13 in both. All paired L1/RMSE difference intervals
remain negative. Matched readout persistence regresses 30.72% / 28.92% and
36.11% / 32.83%, respectively. These are simulated sensor stresses, not measured
camera performance or a calibrated observation likelihood.
They do not establish uniform robustness to arbitrary coherent camera bias.

## Verification and Provenance

- Parent source: `b85b81623ce9d982d4dee32647a00342cea706cb`.
- Noise source: `4627180dd27bc0bc4e9929fc2a659cfbfb7c43c3`.
- Native CPU adapter matches the legacy CPU rollout exactly; archived GPU maximum
  difference is 0.003636 mm, coordinate RMSE 0.000407 mm.
- Zero-update continuation is byte-identical; fixed synthetic state perturbation
  recovery is 100% with zero remaining trajectory difference.
- Independent saved-array/metric recomputation verifies 140 parent and 1,792 noise
  case predictions, all intervals, unchanged incumbent means, and matched readout
  formulas. It is not a second independent native-physics execution.
- Final focused suite: 335 passed, 6 dependency skips; the 36 focused restart/noise
  tests also pass with Torch available, including all tensor-state checks. Ruff
  and focused MyPy pass. No full-repository suite is claimed for this isolated
  experiment-only change.
- All predictions were sealed before their metrics; no failed or replaced cases.
- Parent result file SHA-256:
  `0e83bca3c39f3a6807698fe6c9c20606196943d9ba20545ad6226de917b1f85d`.
- Noise result file SHA-256:
  `43478722e1fbfbe63d1ec4a2c596cf8a0e519485a3ee67675ad9d692833132c4`.

Compact records are in `results/sota/deform_sparse_state_restart_dev_v1/` and
`results/sota/deform_state_restart_noise_dev_v1/`. Full prediction arrays and
source archives remain in the hash-bound local/server `source-only` directories.

## Contribution Decision

This is a materially stronger lead than adding another backend or another static
readout patch: matched sparse information helps when its residual is propagated
through the existing physical dynamics while retaining the learned readout.
State estimation itself is not new; the potential contribution is a demonstrated,
baseline-preserving coupling that transfers across objects/backends and budgets.
Only the DLO2 part of that proposition is supported here.

Next justify a separately frozen multi-object, matched-observation-budget study
and a source-only uncertainty-aware admission rule. Do not relabel the opened
trajectories as fresh or access reserved DLO4/DLO5, held-v8, or other protected
cohorts. Additional prefix observations make this an online comparison, not an
improvement to the original frame-zero-only published benchmark contract. No new
official SOTA, autonomous-perception, calibration, or ICRA-acceptance claim follows.

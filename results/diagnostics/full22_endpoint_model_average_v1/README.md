# Full-22 endpoint model-average diagnostic v1

> **Evidence status:** retrospective, non-claim-bearing diagnostic. The official
> released cohort and its development/confirmation split already informed method
> development. These numbers are not fresh independent validation, calibrated
> deployment uncertainty, or a new state-of-the-art claim.

## Locked execution

- Protocol SHA-256: `8c4021f082b03ef761bc97300eeac11b6f3f92a2bdc52c1941020f6c1f340217`
- Hosted run: `30884863739`
- Hosted artifact: `8882742040`
- Hosted artifact SHA-256: `3ecdf416c6bf534286aecdc4760f69a7c34ce5ecaaf5a4d17b62b932eb07138d`
- Evaluated PR merge revision: `054fedf3ac9c71299fe464ff3b62e68fc14d08e9`
- Summary SHA-256: `6b2baa1d1cddb941c7682aebc360e3fc9622bbd553849dbb95696c7afba7f672`
- Pre-target gates: Ruff, formatting, 48 focused tests, protocol digest, and package consistency all passed.

## Main point result: confirmation 19

| Method | Chamfer (mm) | Track (mm) | ΔCD vs released | Δtrack vs released |
| --- | ---: | ---: | ---: | ---: |
| `released_phystwin` | 11.122 | 22.189 | 0.00% | 0.00% |
| `selected_bayesian_anchor` | 9.828 | 19.523 | -11.63% | -12.02% |
| `model_average` | 9.713 | 19.335 | -12.67% | -12.87% |
| `model_average_anchor_guard` | 9.878 | 19.531 | -11.19% | -11.98% |
| `last_residual` | 9.581 | 19.188 | -13.85% | -13.53% |

The model average improves the selected anchor by **0.115 mm CD** and
**0.189 mm track error** on the equal-case mean. The locked paired bootstrap
assigns about **85%** probability to a mean improvement, but both 95% intervals
cross zero: CD `[-0.359, +0.097] mm`, track `[-0.590, +0.129] mm`.

The simple last-supported-residual baseline remains best. Relative to it, the
model average is worse by **0.132 mm CD** and **0.146 mm track error**. The CD
difference is nonzero under the locked bootstrap: `[+0.029, +0.263] mm`.

## Horizon localization

| Horizon | Anchor CD | Model-average CD | Anchor track | Model-average track |
| --- | ---: | ---: | ---: | ---: |
| early | 7.355 | 7.111 | 15.417 | 14.968 |
| middle | 9.872 | 9.774 | 20.394 | 20.105 |
| late | 12.396 | 12.400 | 22.911 | 23.110 |

The model-average mean advantage is concentrated early and in the middle. At
the late horizon, CD is essentially tied and track error is **0.198 mm worse**.

## Raw predictive calibration

| Posterior | Mean error (mm) | Pred. std (mm) | 90% coverage | NEES / 3 | Mean NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `selected_bayesian_anchor` | 17.665 | 7.583 | 0.519 | 691.1 | 1022.1 |
| `model_average` | 16.641 | 3.667 | 0.380 | 2529.8 | 3779.1 |

Both raw posteriors are unusably overconfident. The model average reduces mean
error by **5.8%** relative to the selected anchor, but reduces predictive
standard deviation by **51.6%**. Its 90% coverage falls by **13.94 percentage
points**, while NEES/3 and NLL become about **3.66×** and **3.70×** larger.

Coverage also deteriorates with horizon for the model average: `0.455` early,
`0.343` middle, and `0.315` late at the nominal 90% level.

The uncertainty score still has ranking value: retaining the lowest-uncertainty
25%, 50%, 75%, and 100% gives mean errors of `7.373`, `10.405`, `13.530`,
and `16.641 mm`. This supports selective use after calibration, not use of the
raw covariance magnitude.

## Why model averaging did not fix uncertainty

- Mean component entropy: `0.1175 nats`
- Median effective component count: `1.000002` of 15
- Median between-model covariance fraction: `2.926e-07`

Cumulative per-track predictive log evidence almost always collapses to one grid
component. Consequently, the nominal mixture contributes essentially no
between-model covariance. This is the central negative result of the diagnostic.

## Guard diagnosis

The frozen anchor guard accepts 15/19 cases. Reusing it for the model average is
counterproductive:

- unguarded model average: `9.713 / 19.335 mm` CD/track;
- anchor-guarded model average: `9.878 / 19.531 mm`;
- selected anchor: `9.828 / 19.523 mm`.

On the four cases rejected by the anchor guard, the model average improves the
fallback from `7.214 / 12.678 mm` to `6.433 / 11.746 mm`. A future model-average
candidate therefore needs its own source-calibrated regret guard.

## Scientific decision

Do **not** promote the current raw model-average covariance or the reused anchor
guard. Preserve the slight mean predictor as an experimental arm. The next
locked experiment should estimate evidence temperature or hierarchical component
weights on development/source groups only, fit a model-average-specific guard,
and conformalize on independent object/session groups before any claim-bearing
evaluation.

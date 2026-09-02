# DEFORM DLO4/DLO5 outer support-adequacy pilot

The existing query-quotient certificate is exact over the finite physical support supplied to it. This experiment tests the separate question of whether that represented support is adequate for the present trajectory.

## Two outer mechanisms

The primary method fits a source-only ridge predictor of realized regret excess over the inner support bound. A disjoint source-calibration set converts its errors into a one-sided complete-trajectory maximum conformal margin. The deployed upper bound is never allowed below the original inner bound. A decision is retained only when the inflated bound remains below the registered regret tolerance.

A secondary logistic gate estimates the probability that realized normalized regret exceeds the registered tolerance. It is an operational signal diagnostic, not a distribution-free certificate.

## Information boundary

Every gate feature is available before the future internal-node outcome:

- the inner certificate's source-support regret bound and complete regret vector;
- quotient concentration and maximum quotient mass;
- kernel concentration;
- expected action gap and expected fallback advantage;
- hypothesis-action agreement;
- residual disagreement and unsupported specificity;
- action identity, DLO identity, and current frame.

The following are explicitly forbidden as gate features:

- realized target regret or harm;
- nearest-source residual error measured against the target suffix;
- oracle action;
- any future internal-node coordinate.

## Source design

All 112 official DLO4/DLO5 training trajectories are evaluated by five-fold trajectory-disjoint cross-fitting of the already selected inner-certificate settings. A deterministic 64/24/24 trajectory split is then used for outer-model fitting, ridge selection and threshold selection, and final source calibration.

## Target design

The frozen outer model is applied without target adjustment to the 28 official evaluation trajectories. The study reports nonfallback coverage, realized tolerance violations, harmful departures, retained RMSE gain, DLO-specific results, and exact fallback behavior.

## Boundary

This is retrospective public-data method development on a target roster that has already been opened by earlier experiments. Source-to-target exchangeability is not assumed to be established. A favorable outcome motivates a separately frozen support-misspecification theorem and confirmation protocol; it is not deployment authorization or a universal safety guarantee.

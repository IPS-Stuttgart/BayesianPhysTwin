# GP follow-up v2: calibrated uncertainty and passive prefix revision

## Question

Does the v1 GP provide useful predictive uncertainty, and can its cross-time
posterior covariance improve future predictions after observing a short prefix
of an already recorded trajectory? This is offline passive assimilation, not
active probing, a robot experiment, simulator retraining, or an intervention.

The original 40 fit / 8 validation split, simulator checkpoint, linear and GP
mean hyperparameters, and residual shrinkage 0.25 remain unchanged. Historical
source-test and official evaluation remain forbidden. This is development
analysis on a previously used validation panel, not fresh confirmation.

## Fit-only estimation

Use deterministic five-fold trajectory-level cross-fitting on the 40 fit
trajectories (seed 20260909). Feature normalization, anchors and GP/ridge weights
must be recomputed inside each fold. Fold predictions hold out the residual
learner only: the frozen simulator was historically trained on this fit panel,
and its checkpoint was historically selected using validation. Do not call this
an end-to-end blind cross-validation estimate.

For each mean model, estimate per-node/per-local-coordinate error second moments
from out-of-fold residuals of the *shrunken* predictor. The constant-diagonal
uncertainty comparator uses these moments directly. For GP and linear posterior
covariance shapes, fit two positive covariance components per node using the
same out-of-fold errors at 20 uniformly spaced forecast steps:

    Cov(error_c) = q_c [a C / mean_diag(C_oof) + b I].

Here C = shrinkage^2 D precision_inverse D^T is the fitted weight-uncertainty
shape, q_c is the out-of-fold coordinate error second moment, and a,b are fitted
by Gaussian joint likelihood (positive bounds 1e-6..1e3). The intercept follows
the existing unpenalized-intercept convention. This is empirical covariance
calibration, not an unmodified or fully Bayesian posterior. No cross-coordinate
or cross-node covariance is claimed. Log scores are Gaussian working-model
scores; correlated time steps are not independent statistical replicates.

## Registered comparisons

1. Same-mean uncertainty: constant diagonal versus calibrated linear/GP
   covariance. Report marginal Gaussian NLL in metre units, CRPS, nominal 95%
   coordinate coverage and mean interval width. Exclude the four exactly
   clamped nodes from probabilistic scores. Also retain the uncalibrated GP
   predictive-variance diagnostic around the actual shrunken mean.
2. Prefix forecast revision: observe forecast indices 4,9,...,49 (10 frames,
   through step 50) and score only indices 50..497. Primary candidate uses the
   calibrated GP cross-time covariance for Gaussian conditioning of the fixed
   GP mean. Compare to the unchanged GP, the unchanged ridge, GP plus average
   prefix-error correction, and GP plus a per-node prefix-bias coefficient
   fitted on out-of-fold fitting trajectories and clipped to [0,1]. Also run
   calibrated linear posterior conditioning.
3. Controls: a zero cross-time covariance must leave the mean unchanged;
   permuting validation trajectory prefixes cyclically (a deliberately wrong
   correspondence, never a valid predictor) is a negative control. Predictions
   must not depend on unobserved validation suffix targets. Clamped coordinates
   must remain bitwise equal to the baseline.

The observed prefix is never included in forecast scoring. All adapted methods
receive the same ten recorded frames. No covariance or adaptation parameters
are selected using validation targets. Report all predefined methods including
worse ones. Aggregate equally across trajectories, with paired trajectory
bootstrap intervals (5000 replicates; seed 20260909) as descriptive summaries.
The primary prefix metric is all-node coordinate L1 on the 448-step suffix;
free-node L1 and per-trajectory outcomes are retained as additional diagnostics.

## Evidence

Export native fit/validation query tensors with unchanged v1 checks. Retain
fold identities, covariance/bias parameters, prediction seals, all candidate
scores, per-trajectory CSVs, source hashes and any technical failures. Changes
that repair implementation errors must be distinguished from outcome-driven
scientific changes. Neither an accuracy gain nor useful Gaussian conditioning
alone establishes a unique Bayesian methodological contribution.

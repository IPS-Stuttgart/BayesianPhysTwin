# Passive visible-to-hidden uncertainty-scale update

This is a focused continuation of PR #939, not a retuning of the completed
partial-observation point-accuracy experiment. The earlier result and its
negative stronger-model decision remain unchanged.

## Question

Does observing a large prediction error on visible parts tell us how uncertain
we should be about the unobserved future? All seven competitors have exactly
the same conditional point prediction. The experiment tests distributional
predictions, not another Bayesian-versus-ridge mean comparison.

A parallel conditional-query experiment in PR #940 already examines integration
of parameter uncertainty in a small-data surrogate. This study instead keeps the
DEFORM hybrid anchor and tests posterior conditioning of a shared residual scale
using the current visible residual. It is not a replication of that small-data
experiment or a claim that Student-t inference is new.

## Model

Let the current observed residual O and future hidden residual H be jointly
Gaussian conditional on a positive latent scale lambda. Their covariance is
lambda times a source-estimated joint covariance C. The mean and C are fitted
from source trajectories; a source-selected nugget regularizes the observation
block. The nugget is part of the predictive model and scales with lambda. No
artificial measurement corruption is introduced.

The prior is inverse-gamma with shape nu/2 and scale s*(nu-2)/2, so its mean is s.
The usual empirical Gaussian conditioning gain determines the point prediction.
If q is the visible innovation's squared Mahalanobis norm and m is the number
of observed coordinates, then conditioning gives

```
posterior shape = (nu + m)/2
posterior scale = (s*(nu-2) + q)/2
hidden Student-t degrees of freedom = nu + m
hidden Student-t scale matrix = (s*(nu-2)+q)/(nu+m) * Schur(C)
```

This exact conjugate calculation integrates lambda, not physical material
parameters. C, the mean, nu, the nugget and calibration parameters are fitted
empirically. There is no fully Bayesian hyperparameter claim. Related standard
literature includes Shah, Wilson and Ghahramani, *Student-t Processes as
Alternatives to Gaussian Processes*, arXiv:1402.4306.

## Strong controls

All methods share the exact same mean and conditional covariance shape. They
vary only the scalar uncertainty and predictive family:

- Independently calibrated static Gaussian.
- Bayesian observation-conditioned scale mixture (Student-t).
- Gaussian with exactly the Bayesian mean and conditional covariance.
- The same Gaussian with its own source-fitted scale recalibration.
- Independently calibrated static Student-t.
- Direct heteroscedastic Gaussian variance regression.
- Direct heteroscedastic Student-t variance regression.

The direct regressions have a continuous nonnegative slope on the visible
Mahalanobis energy, fitted separately for the two observation counts. Their
degrees-of-freedom grid includes every Bayesian conditional df. They can
represent the Bayesian conditional family; an analytic test verifies that
inclusion. Thus this is a comparison of source-fitted procedures, not proof
that equivalent frequentist distributions cannot reproduce a Bayesian result.

## Data and information boundary

Use the unchanged canonical DLO4/DLO5 source/evaluation loader and hash-bound
hybrid checkpoints from the parent experiment. Each of the two objects has
56 source trajectories and all 14 previously opened evaluation trajectories.
Each source set is deterministically split into 32 covariance/mean fit,
12 hyperparameter-selection, and 12 scale-calibration trajectories. The
backbone was previously trained on all 56, so these are not wholly independent
calibration records. The complete source-only selections and models are sealed
and uploaded before this experiment prepares target outcomes.

The six fixed visibility masks and 18 fixed observation anchors are inherited
from the earlier experiment. The forecast horizon is fixed at 30 frames. Only
hidden internal nodes are scored. The current visible residual values are the
only target measurements that affect uncertainty. No target hyperparameter
selection, active sensing, new robot experiments, dataset mutation or reserved
cohort access is performed.

The published DEFORM anchor remains conditional on the recorded future clamp
trajectories. Coordinate masking is not actual camera occlusion. No population
claim across arbitrary physical objects or new calibration guarantee is made.

## Decision rules and stop condition

The primary score is equal-object, equal-trajectory, equal-mask joint negative
log density per hidden coordinate. CRPS, marginal coverage/width and Brier
scores for an absolute coordinate prediction error exceeding 20 mm are retained.
Complete trajectories are bootstrapped within the two fixed objects; windows,
coordinates and masks are not treated as independent experimental units.

A comparator is passed only if all the following hold: mean NLL improves by at
least 0.01 nats per coordinate; the paired 95% interval is strictly favorable;
both fixed objects improve; and the CRPS interval upper endpoint is less than
+0.1 mm. Three separate flags are retained: scale updating versus static
Student-t; integration versus exact-moment Gaussian; and the stronger criterion
against every independently fitted control. The weaker flags cannot replace a
failed stronger flag.

If the stronger criterion fails, this scale-mixture implementation should not
become the paper's distinctive Bayesian headline. No target-side tuning or
favorable replacement result is authorized. A valid negative scientific outcome
is a successful completed workflow.

## Execution and verification

The workflow is `.github/workflows/partial-observation-scale-v1.yml` and starts
only when `.github/requests/partial-observation-scale-v1.json` changes on the
existing research branch. It uses `[self-hosted, Linux, X64, gpuserver4090]`.
The original source rollout is replayed in its frozen runtime; the new numerical
experiment uses an isolated NumPy/SciPy environment. Existing datasets, models,
and result directories are not altered.

Thirteen analytic and information-exclusion tests cover inverse-gamma conjugacy,
Gaussian limits, exact moment matching, direct-control inclusion, SciPy density
parity, quadrature checks of CRPS, masks, hidden/future poisoning, and both a
shared-scale positive control and independent-hidden-scale negative control.
The ten parent tests are also rerun. The pipeline was exercised locally on
synthetic fixtures before real execution; those fixtures are software checks,
not empirical evidence.

`verify.py` imports no experiment implementation. It reconstructs the held-out
residuals from prepared arrays, checks the predictive densities using SciPy,
reaggregates every scalar result and independently reconstructs all paired
bootstrap intervals. This is artifact verification, not another simulator run.

All source models, decisions, 1176 per-case/mask/arm records, scalar scoring
inputs, hashes, and execution logs are retained as Actions artifacts. The PR
remains draft and changes no production API or paper claim automatically.

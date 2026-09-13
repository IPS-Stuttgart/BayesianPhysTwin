# Partial-observation conditioning v1

## Question and scope

Does a joint Gaussian residual belief around a fixed DEFORM hybrid let sparse
observations improve the still-unobserved deformation and its recorded future?
This is a new retrospective operator on previously opened DLO4/DLO5 records,
not an independent confirmatory cohort. No robot, active probing, new data,
real-camera provider, or physical action selection is involved.

The baseline is the pinned source-trained DEFORM physics-plus-GCN hybrid from
workflow 33361441865, not bare rod physics. Source trajectories are replayed
through the existing checkpoint without optimization. Target predictions are
reused byte-for-byte after verifying their SHA-256. The original all-coordinate
point errors must reproduce before the new test is scored.

## Fixed observation operator

Only eight internal nodes (original indices 2 through 9) are eligible. The four
boundary nodes are excluded from all new scores. The cached physical predictor
uses the first two complete recorded frames and the complete recorded boundary
trajectory; forecasts are therefore conditional on those known boundaries.

Observations arrive independently at 18 fixed anchors in the 498-frame forecast
array: 25, 50, ..., 450. The experiment is one-observation conditioning, not a
recursive filter. Forecast offsets are 0, 10 and 30 frames. Offset zero tests
reconstruction, while offsets 10/30 test future prediction.

Six masks reveal 2/8 or 4/8 internal nodes: two spread patterns and four
contiguous left/right patterns. Only the complementary internal identities are
scored, even at future offsets. No additional measurement corruption is added.
This is artificial withholding of real recorded coordinates, not camera
occlusion. The measurement variance is a source-selected regularizer rather
than a independently measured sensor noise variance.

## Shared predictive mean and residual beliefs

All conditioning methods start from the same frozen physical prediction plus a
source-fitted node bias. For each horizon, the residual state stacks current and
future internal coordinates (only one copy at horizon zero). The prior mean and
covariance are fitted to source windows, never to evaluation outcomes.

The structured model shrinks the source empirical covariance toward an inverse
second-difference spatial-precision prior, with exponential temporal correlation.
It is an empirical-Bayes Gaussian *output-discrepancy* belief, not an identified
posterior over physical parameters or corrected material state. Its grid is
fixed in run.py. A separate empirical covariance model shrinks toward the
diagonal and has its own equal-source validation. All these priors have exactly
the same coordinate marginal variances before conditioning.

Each object has 56 official source trajectories. A name-hash split puts 40 into
residual-model fitting and 16 into validation. Every nontrivial model and
heuristic is chosen using only that split, then refitted on all 56 source
trajectories. The backbone itself was already trained on all 56: these are
residual-model validation records, not independently held-out backbone records.
No independent-calibration claim follows.

## Comparators

- Unmodified physical forecast and the common bias-corrected prior.
- Structured Gaussian conditioning; diagonal and fixed sign-scrambled,
  marginal-matched destruction controls; independently tuned empirical covariance.
- Direct regularized linear regression on observed residuals, plus global
  translation, spatial linear interpolation, and frozen-current interpolation.
  Their regularizers/gains are selected on the same source validation records.
- An independently implemented information-form Gaussian MAP solution. It must
  reproduce the structured posterior mean within 1e-8 metres. It is an
  equivalence control, never counted as a Bayesian point-prediction win.

## Primary analysis and uncertainty

The primary endpoint is hidden-node **3D point RMSE at +30 frames**, with equal
mask, complete-trajectory, and DLO weighting. Reconstruction and +10-frame RMSE,
coordinate L1, and translation-free hidden-shape RMSE are secondary. Translation
removal subtracts the hidden-node centroid of each frame's error; it does not
align scale or orientation or alter primary scoring.

For Gaussian arms, report joint Gaussian NLL per hidden coordinate, normalized
joint NEES, marginal 90% coverage, and full interval width. Do not describe
coverage without width or call these raw covariances calibrated.

Paired intervals bootstrap complete trajectories within each of the two fixed
DLOs, not frames, nodes or individual observations. These intervals condition on
two particular physical objects and do not estimate arbitrary-object transfer.
The stronger model-value flag requires the structured method to beat every
non-equivalent empirical/ridge/translation/interpolation/frozen-interpolation
control by at least 1% in each DLO and have negative paired upper confidence
limits. A win only against diagonal uncertainty establishes a narrower benefit
of dependence, not Bayesian superiority. Negative scientific outcomes leave the
workflow successful and remain in its result tables.

## Execution and custody

The push trigger watches only request.json on the dedicated research branch.
Workflow phases: analytic tests -> source checkpoint replay -> source model
selection -> upload source seal -> load complete 14+14 official targets -> score
all methods -> upload complete compact evidence, including failure receipts.
There is no workflow_dispatch trigger. The runner uses gpuserver4090. No
reserved cohort, other campaign, dataset, checkpoint or existing cache is
modified. New source replay arrays stay in the run-specific runner cache;
artifacts contain compact metrics, models, provenance, logs and the protocol.

Tests cover Gaussian/MAP parity, marginal matching, positive-definiteness,
correct-dependence positive control, diagonal negative control, hidden-target
poisoning, mask disjointness, interpolation, frame alignment and metrics.

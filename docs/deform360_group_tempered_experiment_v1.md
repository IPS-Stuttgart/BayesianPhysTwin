# Deform360 group-held-out tempered endpoint experiment v1

## Scientific motivation

The locked full-22 PhysTwin follow-up selected an evidence temperature of 128 at
the upper edge of its registered grid. Tempering restored meaningful mixture
uncertainty, but the tempered point mean was slightly worse than the original
model average and the development-selected disagreement guard did not transfer.
The result identified cumulative evidence scale as a real failure mode while
showing that the historical 22-case cohort is too reused and too small for a
claim-bearing calibration conclusion.

This experiment tests the resulting low-dimensional hypothesis on released
Deform360 trajectory artifacts. It does **not** reuse the historical three-case
PhysTwin development split. Instead, it creates disjoint path-defined
object/session groups and freezes source, calibration, and target roles before
any archive array is parsed.

The implementation is
`scripts/science/run_deform360_group_tempered_experiment.py`; its locked protocol
is `protocols/deform360_group_tempered_experiment_v1.json` with canonical
SHA-256
`b1732d062b8766e27d0d96feaa88400906955522e73d3bb63b8ec78b8994aa58`.

## Information boundary

The experiment proceeds in five ordered phases.

1. Discover candidate NPZ paths and derive a group identity from the object
   directory plus the first non-generic nested acquisition/session directory.
   A deterministic archive cap is applied round-robin across groups.
2. Hash every selected archive byte-for-byte and derive a deterministic,
   salted source/calibration/target split from group names. Hashing binds the
   target inputs but does not parse target arrays or expose outcomes.
3. Open source-group arrays only. Select an evidence temperature and an exact
   fallback guard, then fit a smooth nondecreasing horizon-scale shape.
4. Open calibration-group arrays only. Fit one finite-sample conformal
   multiplier, serialize the complete selection, compute its content ID, read
   it back, and verify that ID.
5. Only after that verification are target-group arrays parsed and scored.

A grid-boundary temperature is an intentional inconclusive result. In that
case, target arrays are never parsed.

## Compared point predictors

All predictions are rolling and causal. For a prefix ending immediately before
one residual frame, the experiment compares:

1. persistence;
2. the latest valid residual;
3. the original cumulative-evidence model average;
4. per-observation evidence normalization at temperature 1;
5. per-observation evidence normalization at the source-selected temperature;
6. the selected normalized mean behind the source-selected guard, with exact
   latest-residual fallback for every rejection.

For fixed-identity trajectories, identity RMSE and correspondence-free symmetric
Chamfer RMSE are reported. Packed visual hulls are evaluated through centroid
translation error and symmetric Chamfer RMSE. Metrics are averaged within an
archive, then within a path-defined group, and target inference uses paired
group bootstrap resampling.

## Evidence normalization and temperature

The original endpoint bank accumulates predictive log evidence across all valid
prefix observations. V1 divides each component's cumulative log evidence by the
track's update count before applying a scalar temperature. This makes the
weighting invariant to duplicating an otherwise identical evidence prefix and
tests whether the earlier temperature of 128 was largely compensating for
sample-count scaling.

The fixed grid is `2^k` for every integer `k` from -16 through 16. Source groups
select the temperature with minimum equal-group, equal-archive one-step Gaussian
negative log likelihood. Ties prefer the value closest to one and then the
smaller value. Target access requires an interior optimum so the selected scale
is bracketed on both sides.

## Guard

The point fallback is always the latest valid residual. The score is the
finite-sample 90th percentile, across updated tracks, of

`||selected_mean - fallback|| / predictive_standard_deviation`.

Source groups choose a monotone threshold that minimizes equal-group,
archive-balanced regret while admitting no positive regret in any source group.
A rejection reproduces the fallback exactly; no interpolation or post-hoc blend
is permitted.

## Uncertainty calibration

For horizons one through eight, source groups produce finite-sample 90% NEES
scales. Their group medians are approximated by a weighted log-linear function
of `log(1 + horizon)`, constrained to be nondecreasing and at least one.
Calibration groups then contribute one finite-sample multiplier based on their
90th-percentile adjusted NEES ratios. Target covariance is the frozen source
shape times this frozen multiplier.

This is a group-separated calibration experiment, not a deployment-calibration
claim. Coverage is reported with equal archive and equal target-group weighting.

## Registered decision

The guarded normalized candidate must beat latest residual in paired target-group
bootstrap inference: its mean combined point error must be lower and the upper
95% bootstrap bound must be below zero. Harmful accepted updates may not exceed
5%. Separately, calibrated 90% coverage must lie within five percentage points
of 90%. Both conditions are required for the registered pass.

A valid negative or boundary-inconclusive result exits successfully and is
retained. Only protocol, integrity, environment, or dataset-contract failures
make the workflow fail.

## Claim boundary

This is an external, group-held-out diagnostic on released Deform360 trajectory
artifacts. It is narrower than the official action-conditioned Deform360 world
model and does not reproduce the official Table-4 protocol. Prior repository
work has touched Deform360, so the path-hash split protects this execution from
target-informed fitting but does not retroactively make the release globally
unseen. A positive result justifies a separately locked official-adapter study;
it is not itself an official benchmark, state-of-the-art, tactile-benefit, or
deployment-calibration claim.

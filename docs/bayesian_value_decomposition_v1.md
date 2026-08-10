# Bayesian-value decomposition v1

The BayesianPhysTwin point estimate is close to a simple last-residual
comparator on the released PhysTwin cohort. A prospective comparison should
therefore identify **which part of the Bayesian deployment stack contributes
value**, rather than test only one final candidate against the physical
fallback.

## Required arms

The registered ladder contains the physical fallback plus four matched arms:

| Method ID | Predictive mean | Deployment treatment |
| --- | --- | --- |
| `physical_fallback` | unchanged physical predictor | always deployed |
| `last_residual` | held last residual | no selective fallback |
| `last_residual_guarded` | byte/numerically identical raw last-residual prediction | registered uncertainty and guard |
| `bayesian_mean_guarded` | Bayesian posterior mean | the same frozen deployment policy family |
| `bayesian_full_guarded` | complete recursive Bayesian belief | structured uncertainty, reliability, recursion, guard, and exact fallback |

`last_residual` and `last_residual_guarded` must have identical raw predictive
samples, Gaussian moments, median, and intervals on every unit. Only their
deployment decision may differ. This makes their deployed-score difference an
attribution of selective deployment rather than a hidden change of predictor.

## Required contrasts

The machine-readable template in
`protocols/templates/bayesian_value_decomposition_v1.json` registers four
candidate-minus-reference contrasts:

1. `last_residual` versus `physical_fallback`;
2. `last_residual_guarded` versus `last_residual`;
3. `bayesian_mean_guarded` versus `last_residual_guarded`;
4. `bayesian_full_guarded` versus `bayesian_mean_guarded`.

Report both raw and deployed differences. A guard can improve tail risk while
worsening the mean through fallback, or improve the mean while rejecting some
otherwise accurate cases; reporting only the final deployed mean would hide
that mechanism.

## Frozen experimental boundary

Before target outcomes are opened, bind:

- complete physical-object or independent-session development, calibration, and
  target sets;
- the exact physical fallback, last-residual implementation, Bayesian candidate,
  guard, query, horizon, and interval construction;
- the observation provider, causal cutoff, model/checkpoint set, covariance
  semantics, and independent-anchor declaration;
- technical-failure and exclusion rules;
- proper-score configuration and component-pair list;
- group-clustered bootstrap settings and all decision margins; and
- `RunManifestV2` identities for every arm and output.

Every rejected deployable arm must reproduce the exact physical fallback for the
registered physical loss and the corresponding fallback predictive distribution
for every proper score.

## Required reporting

For each metric and horizon report:

- raw and deployed physical loss;
- energy, variogram, Gaussian log, and weighted interval scores where the
  corresponding predictive representation is valid;
- candidate-minus-reference paired group means for all four contrasts;
- acceptance, rejection, exact-fallback, and harmful accepted-update counts;
- 90th- and 95th-percentile regression;
- interval coverage and width;
- risk--coverage curves; and
- results conditioned on source reliability and identifiable rank.

A positive result supports only the registered readout/model-discrepancy and
physical-query contract. It does not by itself establish a dynamically
admissible simulator-state correction, general calibration, deployment safety,
or state of the art.

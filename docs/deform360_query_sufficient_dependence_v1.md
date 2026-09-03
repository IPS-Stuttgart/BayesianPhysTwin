# Deform360 query-sufficient dependence compression v1

## Question

The completed Deform360 same-mean study showed that the full low-rank dependence
model improves registered-query decision loss and Brier score relative to both a
diagonal covariance and a marginal-preserving scrambled covariance. Predictive
means and all coordinate marginals were identical, but the full query
distribution was not calibrated.

This extension asks a narrower representation question:

> Can the complete dependence advantage for the five frozen tactile queries be
> retained while discarding every shared-factor direction that is invisible to
> those queries?

No new measurement or target is opened. The exact 92-object result, parent
confirmation, source-only calibration, point predictor, query bank, event
thresholds, decision rule, and bound carrier receipts are reused.

## Exact factor construction

For one object, let the frozen predictive covariance be

```text
Sigma = D + U U^T,
```

where `D` is the unchanged diagonal and `U` is the shared factor. Stack the five
registered scalar query weights into `W`. An orthonormal latent projection `V`
preserves their complete joint covariance when

```text
range(U^T W^T) subseteq range(V).
```

The minimum rank within latent orthogonal projections is therefore

```text
rank(W U) <= 5.
```

The materialized query-sufficient factor is `U V`; the diagonal, predictive
mean, query calibration, and decision rule remain unchanged. The experiment
computes this subspace directly and independently through the pinned Prob4D
`query_preserving_compression` implementation. Their ranks and projectors must
agree, and the complete `5 x 5` query covariance is audited numerically.

## Arms

1. `full_low_rank`: exact frozen reference covariance.
2. `query_sufficient_portfolio`: minimum factor subspace preserving the joint
   five-query covariance.
3. `leading_energy_matched_rank`: observation-energy/PCA projection at the same
   retained rank.

The previously completed diagonal and marginal-preserving scrambled controls
remain the evidence that dependence has decision value. They are reproduced
exactly but are not redefined by this extension.

## Primary gates

The run succeeds scientifically only when:

- all 92 bound objects reproduce the complete original full-arm study exactly;
- every bound numeric fingerprint and action is unchanged;
- no unbound numeric payload is opened;
- every retained rank is at most five;
- the direct minimum-rank construction and pinned Prob4D implementation agree;
- the complete five-query covariance is preserved within the frozen tolerance;
- every frozen query metric, event probability, and execute/fallback decision is
  preserved within the frozen tolerance; and
- the original full-versus-diagonal/scrambled dependence-value result remains
  positive.

The matched-rank energy arm is a falsification control, not a promotion gate.

## Representation accounting

Factor payload counts only float64 entries of the materialized shared factor.
The unchanged diagonal, mean, and metadata are excluded from both arms. The run
also reports the mandatory simpler fixed-query baselines:

- a full `5 x 5` query covariance cache;
- its symmetric 15-entry representation; and
- the five scalar variances actually consumed by the frozen decision rule.

For an immutable query portfolio these caches are smaller than a factor. The
factor result is useful only when an existing downstream interface must retain a
structured covariance factor or when the construction is amortized across
compatible consumers.

## Reproduction

The numerical workflow is triggered by adding exactly one frozen request file
and runs on:

```text
[self-hosted, Linux, X64, gpuserver4090]
```

It uses the verified official Deform360 snapshot at:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360
```

The clean official snapshot is not available on `gpuserver6000`, so that runner
is intentionally not eligible for this study.

## Claim boundary

This is retrospective representation evidence on an already opened public-data
cohort. It preserves the covariance and decisions of five registered predictive
queries. It does not preserve the complete tactile-field covariance, establish
query calibration, test the posterior-conditioning theorem on Deform360,
demonstrate a real Prob4D perception provider, establish unseen-object transfer,
or authorize deployment safety or a paper claim automatically.

## Exact-bound mirror execution

The verified official snapshot remains the normative source. Because
`gpuserver4090` is occupied by the separately authorized PoseIt archive hash, a
second workflow may use `/mnt/lexar4tb/datasets/deform360` on `gpuserver6000`
only as an operational carrier mirror. The mirror path is not accepted on name
or directory layout alone. Every selected robot, tactile, and median numeric
carrier and every bound action must reproduce the immutable 92-object parent
receipts; any absent or changed bound carrier stops the run. Extra files, caches,
and episodes are inventoried but never selected or numerically opened. The
mirror workflow cancels the still-queued official-root execution before the
numerical job is admitted, preventing duplicate evaluation.

# Controlled Prob4D-to-BayesianPhysTwin Decision Result v1

## Status

The frozen calibration/target-separated controlled study passed every registered
criterion. The result advances the explicit-gauge persistent observation model
to a genuinely fresh physical-object or acquisition-session gate. It is not a
real-world confirmation or state-of-the-art result.

The target panel contained 384 independently seeded synthetic object/session
groups across nominal correlated observations, common-mode bias, grouped
outliers, weak identifiability, large gauge uncertainty, and mixed stress. The
48 calibration groups and their target-free risk scores fixed each method's
deployment threshold before the target groups were generated.

## Result

| Method | Deployed RMSE | Improvement vs physical | Accepted | Harmful accepted |
| --- | ---: | ---: | ---: | ---: |
| Physical fallback | 6.166 mm | 0.00% | 0/384 | 0 |
| Naive last-frame state | 6.166 mm | 0.00% | 0/384 | 0 |
| Marginal gauge, persistent identities | 1.231 mm | 80.04% | 372/384 | 8 |
| Explicit gauge, framewise identities | 0.719 mm | 88.34% | 375/384 | 0 |
| **Explicit gauge, persistent identities** | **0.534 mm** | **91.33%** | **373/384** | **0** |
| Explicit gauge plus metric anchor | 0.551 mm | 91.06% | 371/384 | 0 |

For the primary explicit-gauge persistent arm, the paired 95% bootstrap
interval for deployed-minus-physical RMSE is `[-5.880, -5.365] mm`. Every
registered scenario improves, with reductions ranging from 86.51% under
grouped outliers to 95.95% under nominal correlated observations. Rejected
updates reproduce the physical fallback exactly.

Accepted primary predictions have mean nominal-90% coverage of 98.68% and a
mean predictive width of 0.946 mm. The explicit joint-gauge formulation also
avoids the eight harmful accepted updates observed under covariance
marginalization. This is the decisive controlled evidence that shared gauge
dependence and persistent causal identity should be represented jointly rather
than folded into rowwise covariance.

## Registered Decision

All six criteria pass:

- at least 10% deployed RMSE improvement;
- paired bootstrap upper endpoint below zero;
- harmful accepted rate at most 5%;
- no scenario regression above 2%;
- exact fallback for every rejection; and
- noninferiority to the marginal persistent arm.

The registered decision is
`advance-to-fresh-physical-object/session-gate`.

## Reproducibility

- BayesianPhysTwin revision:
  `04cc243aea82bfec1b8a2481ef99b38b357e4123`
- Prob4D revision:
  `aa8ffc6541011d044561e09870569a14ab3f586f`
- Canonical protocol SHA-256:
  `921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111`
- Report ID:
  `c592807d62e9f5121acf85747432574601264160de67b15e9a1c8e48a12cc040`
- Pre-outcome verification: 5 focused tests and 1,677 full-suite tests passed;
  29 full-suite tests were skipped; Ruff and protocol-hash checks passed.

The checksummed report, complete 2,304-row target trial table, protocol copy,
and compact summary are in
`results/diagnostics/prob4d_bpt_controlled_decisive_v1/`.

## Claim Boundary and Next Gate

This controlled generator uses the real Prob4D observation-factor types and
the real BayesianPhysTwin prior-aware gauge-and-bias solver, but it does not
establish that a real camera pipeline supplies competent factors on unseen
objects. The next experiment must therefore freeze a real provider and evaluate
fresh physical objects or acquisition sessions with:

1. persistent material identities observed only in the allowed prefix;
2. explicit joint gauge covariance, without rowwise double counting;
3. a source-calibrated baseline-relative risk guard;
4. bit-exact physical fallback;
5. disjoint hidden identities and future-only scoring; and
6. object- or session-clustered accuracy and calibration statistics.

No previously opened Deform360 or PokeFlex target may be repurposed for that
confirmation. A physical pass, not this synthetic result, is required before
claiming state of the art.

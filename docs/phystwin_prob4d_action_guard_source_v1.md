# Action-conditioned Prob4D source diagnostic

## Question

The static Prob4D state field occasionally helps but does not transfer broadly.
This source-only diagnostic asks whether its amplitude should follow the known
action-conditioned physical prediction instead of remaining constant.

The candidate first uses the frozen bias-aware state update at the fit endpoint.
For future frame `t`, it computes signed physical progress

```text
q_t = <x_t - x_0, x_fit - x_0> / ||x_fit - x_0||^2
```

from the selected simulator trajectory and applies `q_t` times the endpoint
correction. Progress is capped at magnitude 2.0. Negative progress retracts or
reverses the field. No future Prob4D or target observation enters the candidate.

Admission uses the same disjoint prefix boundary as the static experiment and is
strictly stronger: the temporal candidate must improve or tie both validation
metrics relative to both the selected Bayesian baseline and the raw static
candidate, with at least 0.1% balanced improvement over each. Rejection is a
bit-exact baseline fallback.

## Synthetic checks

Focused tests establish that the candidate:

- recovers a synthetic action-gain discrepancy better than endpoint persistence;
- rejects a coherent common camera bias;
- retracts under action reversal and caps extrapolation;
- leaves the prefix admission decision unchanged when only the predicted future
  trajectory is mutated;
- preserves bit-exact fallback.

## Open additional cloth source

The method was fixed before its future scores were inspected on the already-open
11-case additional cloth cohort.

| Arm | Future CD (mm) | Late CD (mm) | Future change |
| --- | ---: | ---: | ---: |
| Selected Bayesian anchor | 5.547 | 6.379 | reference |
| Raw action-conditioned candidate | 8.228 | 8.975 | +48.33% |
| Guarded action-conditioned candidate | 5.520 | 6.351 | -0.49% |

One case, `cloth_shirt_fold`, passed. It improved future CD by 0.296 mm. The other
ten cases were exact fallbacks, and no harmful candidate was admitted. The
six-garment cluster interval for guarded minus selected future CD is
[-0.0740, 0.0000] mm; five garment clusters are exact ties.

## Cross-cohort falsification

The exact frozen candidate was then sealed on the already-exploratory PhysTwin-19
cohort before future point clouds or manual tracks were scored. Prefix validation
rejected all 19 candidates. Outcome scoring confirmed exact fallback everywhere.

The unguarded candidate regressed by 20.58% in future Chamfer and 12.33% in future
manual-track error. It also regressed by 13.10% and 9.74% at late horizon. The
guarded result is exactly the selected Bayesian baseline: 9.815 mm future Chamfer
and 19.531 mm future track error.

## Decision

Signed action progress improves one open cloth episode but does not transfer even
to the separate exploratory cohort. Neither the static endpoint field nor this
one-dimensional temporal extension advances to a fresh-object evaluation.

This result narrows the next method requirement. A useful observation update must
infer time-varying state or discrepancy from multiple causal measurements, model
shared camera bias explicitly, and carry a source-calibrated regret certificate.
Scaling one endpoint field is insufficient.

## Evidence

- Additional source result:
  `results/sota/diagnostics/phystwin_prob4d_action_guard_source_v1/additional11_result.json`
  (canonical SHA-256
  `095f29fa7a835344e353c5917ec1089d372b6feca402caa02b859e524da6d37a`).
- PhysTwin-19 prediction seal:
  `results/sota/diagnostics/phystwin_prob4d_action_guard_source_v1/exploratory19_prediction_cohort_seal.json`
  (canonical SHA-256
  `2891def7d884915cdb64dbc208b56e4f64c5dae7e5de1eea4816556f350a351b`).
- PhysTwin-19 result:
  `results/sota/diagnostics/phystwin_prob4d_action_guard_source_v1/exploratory19_result.json`
  (canonical SHA-256
  `89859249e6d6fe68375bcf7d1c7e0d4b68a02c2aa0c6584a9647ce999278966`).

# Full-22 Tempered Endpoint Diagnostic v1 Result

## Decision

The frozen retrospective gate failed all four criteria. Do not promote this
candidate to fresh grouped calibration or a prospective target evaluation.

This result is useful because it separates a real uncertainty defect from the
remaining point-prediction limit: correlation-aware evidence tempering repairs
component collapse materially, but the frozen endpoint-filter family still
cannot beat the last-supported residual.

## Confirmation-19 Result

Lower is better. Values are equal-case means in millimetres.

| Method | Chamfer distance | Track error | Future access for selection |
| --- | ---: | ---: | --- |
| Last-supported residual | **9.581** | **19.188** | none |
| Historical model average | 9.713 | 19.335 | none |
| Prefix-selected tempered mixture | 9.724 | 19.333 | none |
| Prefix-guarded tempered mixture | 9.586 | 19.273 | none |

The guarded method accepted 3/19 cases. Relative to last residual, it regressed
Chamfer by 0.043% and track error by 0.441%. It won both metrics in only 2/19
cases, and its worst case-metric regression was 7.05%, above the locked 2%
limit. The paired 95% intervals include zero because 16 cases are exact ties;
that does not rescue the failed point and tail gates.

## Uncertainty Mechanism

Tempering did what it was designed to do:

| Confirmation-19 diagnostic | Historical mixture | Prefix-selected tempered |
| --- | ---: | ---: |
| Mean component entropy | 0.117 nats | 1.016 nats |
| Effective component count | 1.01 | 2.80 |
| Between-model covariance fraction | approximately 0 | 17.56% |
| Mean predictive std. | 3.667 mm | 5.175 mm |
| Nominal 90% coverage | 37.96% | 48.72% |
| NEES / 3 | 2529.8 | 2188.9 |

The mixture is less collapsed and less overconfident, but 48.72% coverage at a
nominal 90% remains unusable. Independent grouped calibration is still required
for any probabilistic claim.

## Headroom Audit

A post-open per-case oracle choosing independently between the tempered
candidate and last residual improves only:

- Chamfer distance by 0.22%; and
- track error by 0.40%.

Both ceilings are below the frozen 0.5% advancement threshold before accounting
for selector error. No different guard can make this candidate family worth a
fresh evaluation.

## Consequence

The experiment closes endpoint evidence temperature as the route to the
published 8/15 mm frontier. Keep the V2 power-posterior API as an honest
uncertainty ablation, but do not claim calibrated covariance and do not spend a
fresh cohort on this point predictor. A SOTA attempt now needs new state or
observation information, not another weighting rule over the same endpoint
filters.

## Provenance

- frozen implementation and protocol revision:
  `5669559ce7703f605829e7c385ff02c2f73d2c33`;
- protocol canonical SHA-256:
  `cc73650107eb44fbece39dd68e29c0b53ee880c66e6ca293a61ebcf2f86e27e8`;
- raw summary SHA-256:
  `f0b815d91c54ff2396fc81f9eff47e64ca4f131bb58cdd89334d265aebb42c91`;
- compact readout SHA-256:
  `f443d18a3020d6f2acbcb644fcbef527d259e8ba5a0e4151320410b7fad0ad55`;
- server run root:
  `/mnt/corsair/florianpfaff/full22-tempered-endpoint-v1-5669559`;
- compact evidence:
  `results/diagnostics/full22_tempered_endpoint_v1/`.

The exact pre-outcome revision passed 1,681 tests with 29 skips locally, 59
focused tests on `gpuserver6000`, changed-file Ruff, and `git diff --check`.

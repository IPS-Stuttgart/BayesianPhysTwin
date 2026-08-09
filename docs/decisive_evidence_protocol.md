# Decisive evidence protocol

This protocol defines the evidence that should be produced before Bayesian-PhysTwin claims an advantage over a simple residual correction. It is designed for prospective Prob4D → Bayesian-PhysTwin studies, but the result contract is independent of the observation producer.

## Experimental boundary

Split by physical object or acquisition session into development, calibration, and target sets. Freeze the following before any target outcome is opened:

- the split and statistical unit;
- all candidate implementations and package revisions;
- the unchanged physical fallback;
- each method's risk score and operational acceptance rule;
- confirmatory risk thresholds selected only on source/calibration data, plus the secondary target-coverage grid;
- loss metrics and prediction-horizon labels;
- reliability calibration and identifiable-rank definition;
- interval construction and nominal coverage levels;
- regression quantiles and all claim gates;
- the group-clustered bootstrap replicate count, seed, and confidence level.

The minimum prospective comparison is:

| Arm | Observation treatment |
|---|---|
| `B0_physical_fallback` | Unchanged physical prediction; this is the exact fallback for every guarded method. |
| `B1_last_residual` | Simple visual or last-residual correction. |
| `P1_prob4d_marginal_gauge` | Prob4D fused observations with gauge uncertainty marginalized. |
| `P2_prob4d_explicit_gauge` | Prob4D unfused factors with explicit gauge nuisance variables. |
| `P3_prob4d_metric_anchor` | `P2` plus an independently calibrated metric anchor. |

Every nonfallback arm must be evaluated under the same fallback contract. Rejected units return the exact `B0_physical_fallback` outcome; they are not dropped from the denominator. The primary threshold-native risk–coverage view evaluates `risk_score <= threshold` at every distinct score and never splits a tied score block. Confirmatory thresholds must be selected on source or calibration data and frozen before target outcomes are opened. A separately named matched-count view accepts the same number of units per method and remains a secondary equal-coverage diagnostic.

## Required endpoints

For each registered loss metric, report:

1. the operational acceptance coverage and exact-fallback frequency;
2. harmful accepted-update frequency, both over all units and conditional on acceptance;
3. raw and deployed mean loss relative to the physical fallback;
4. worst-case, 90th-percentile, and 95th-percentile per-unit regression;
5. threshold-native risk–coverage curves for every candidate, including zero- and full-acceptance endpoints;
6. separately labeled matched-count curves and paired deployed performance against the registered reference method at equal coverage;
7. predictive coverage and interval width by prediction horizon;
8. raw and deployed performance conditioned on inferred reliability;
9. raw and deployed performance conditioned on identifiable rank;
10. paired equal-group bootstrap intervals versus the fallback and registered reference method.

The primary endpoint should be paired held-out future physical-prediction error. A mean improvement alone is insufficient when a method has a material high-quantile or worst-case regression.

## Input contract

`bpt evidence summarize` consumes one JSON object with contract `bayesian-phystwin-decisive-evidence-v1`.

```json
{
  "schema_version": 1,
  "contract": "bayesian-phystwin-decisive-evidence-v1",
  "protocol_id": "prob4d-bpt-prospective-v1",
  "statistical_unit": "object-session-horizon",
  "claim_boundary": "prospective target set; no post-opening tuning",
  "reference_method": "B1_last_residual",
  "records": [
    {
      "unit_id": "object-07/session-02/early",
      "group_id": "object-07",
      "metric": "track_error_m",
      "method": "P2_prob4d_explicit_gauge",
      "loss": 0.018,
      "fallback_loss": 0.021,
      "risk_score": 0.14,
      "accepted": true,
      "deployed_loss": 0.018,
      "horizon": "early",
      "reliability": 0.86,
      "identifiable_rank": 5,
      "intervals": [
        {
          "nominal_coverage": 0.9,
          "covered": true,
          "width": 0.011
        }
      ]
    }
  ]
}
```

There must be exactly one record for every `(metric, unit_id, method)` combination. Within one `(metric, unit_id)`, every method must be present and must declare the same `fallback_loss`, `group_id`, and horizon. The analyzer fails closed if:

- an accepted record's `deployed_loss` differs from its raw `loss`;
- a rejected record's `deployed_loss` differs from the exact fallback loss;
- a method is missing for a unit;
- fallback outcomes differ between methods;
- a configured reference method is absent.

`risk_score` is ordered so that lower values mean safer predictions. The primary `bayesian-phystwin-threshold-risk-coverage-v1` output accepts every unit satisfying `risk_score <= threshold` at each distinct threshold, includes exact zero- and full-acceptance endpoints, and admits tied scores only as a complete block. Its points are invariant to row order and `unit_id` naming. The secondary `bayesian-phystwin-matched-count-risk-coverage-v1` output accepts the exact same count for every method at each requested target coverage; boundary ties may be broken deterministically by `unit_id` and are explicitly marked. Paper tables must identify which contract produced every reported risk–coverage point.

`intervals[].width` is the full predictive interval width, not a half-width. `horizon` may be a registered label such as `early`, `middle`, or `late`, or a nonnegative numeric prediction step.

## Paired group-clustered uncertainty

The additive `bayesian-phystwin-group-clustered-paired-bootstrap-v1` analysis resamples independent `group_id` values with replacement. Each group receives equal weight. When a group contains several horizons, sessions, or registered unit rows, those losses are averaged within the group before resampling. Frames, views, tracks, points, and tactile taxels therefore cannot increase the effective sample size.

The same sampled group indices are applied to every method, the physical fallback, and the registered reference method. The output reports percentile intervals for the paired mean-loss difference and relative change of means, together with the bootstrap probability that the candidate has lower loss. With fewer than two independent groups, point estimates are retained but interval status is `insufficient_independent_groups`; a degenerate single-group interval is not presented as inferential evidence.

## Command

```bash
bpt evidence summarize \
  evidence.json \
  evidence-summary.json \
  --reference-method B1_last_residual \
  --coverage 0.0 \
  --coverage 0.25 \
  --coverage 0.5 \
  --coverage 0.75 \
  --coverage 1.0 \
  --regression-quantile 0.9 \
  --regression-quantile 0.95 \
  --reliability-edge 0.0 \
  --reliability-edge 0.25 \
  --reliability-edge 0.5 \
  --reliability-edge 0.75 \
  --reliability-edge 1.0 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260805 \
  --bootstrap-confidence 0.95
```

The output binds the input path and SHA-256 digest and records every analysis setting. Include both files in the run manifest and bind the manifest to the frozen protocol, split, baseline, method-freeze, and claim identifiers.

## Interpretation

A Bayesian claim is supported only when the registered Bayesian arm improves the primary error at relevant matched coverages without unacceptable harmful-update frequency or high-quantile regression, and when its uncertainty is informative: coverage should approach the nominal level without achieving it solely through uninformatively wide intervals. The paired group-clustered interval must support the claim at the registered statistical-unit level; row-level or frame-level resampling is not an acceptable substitute. Reliability and identifiable-rank strata must be treated as diagnostics unless their decision rules were frozen before target opening.

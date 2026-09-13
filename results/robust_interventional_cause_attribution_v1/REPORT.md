# Robust/open-set interventional attribution

**Decision:** `robust-open-set-attribution-supported`

| Method | Resolved | Overall accuracy | Unknown recall | False physical promotion | Held-action RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `factual` | 100.00% | 16.96% | 0.00% | 69.81% | 0.8641 |
| `closed_interventional` | 100.00% | 83.34% | 0.00% | 60.02% | 0.0031 |
| `robust_open_set` | 100.00% | 100.00% | 100.00% | 0.00% | 0.0040 |
| `wrong_action` | 100.00% | 33.33% | 0.00% | 100.00% | 0.8303 |

Deterministic query-bound coverage: **100.00%** over 8334 registered-cause trials.

## Frozen criteria

- `robust_overall_accuracy_at_least_0_99`: **pass**
- `unknown_detection_recall_at_least_0_99`: **pass**
- `closed_set_unknown_recall_zero`: **pass**
- `robust_bound_coverage_one`: **pass**
- `wrong_action_accuracy_below_robust`: **pass**
- `near_confounding_not_promoted`: **pass**

## Near-confounding stress

```json
{
  "closed_set_point_label": "cause_b",
  "robust_cause_a_status": "identifiable_but_unstable",
  "robust_cause_b_status": "identifiable_but_unstable",
  "family_falsified": false
}
```

## Claim boundary

Controlled finite-family mechanism evidence. It does not establish natural real-world cause labels, family completeness, unseen-object transfer, or safety.

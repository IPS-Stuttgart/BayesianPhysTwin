# DEFORM support-aware outer certificate

Source selection: **strict-source-selected**
Strict source gate: **True**
Big-result development gate: **True**

| Method | Nonfallback | RMSE gain | Tolerance violations | Inner-bound exceedances | Harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inner_certificate` | 82 | 4.27% | 44/82 | 45/82 | 3/82 |
| `source_selected_outer` | 36 | 2.00% | 3/36 | 4/36 | 2/36 |
| `full_correction_only` | 36 | 2.00% | 3/36 | 4/36 | 2/36 |
| `best_source_ridge` | 48 | 2.41% | 12/48 | 13/48 | 3/48 |
| `ridge_0.1_outer_envelope` | 0 | 0.00% | 0/0 | 0/0 | 0/0 |
| `ridge_1_outer_envelope` | 0 | 0.00% | 0/0 | 0/0 | 0/0 |
| `ridge_10_outer_envelope` | 0 | 0.00% | 0/0 | 0/0 | 0/0 |
| `ridge_100_outer_envelope` | 0 | 0.00% | 0/0 | 0/0 | 0/0 |

Retrospective source-frozen development on the previously opened official DEFORM DLO4/DLO5 held trajectories. The outer policy uses only the original source-test trajectories and outcome-free target prefixes/actions. It can test whether a support-adequacy layer reduces realized regret violations while retaining utility, but it is not fresh confirmation, an unseen-object result, a distribution-shift theorem, deployment authorization, or safety.

# Deform360 held-out action-family analysis v6

The target action family is absent from every same-object source episode for **21 objects**. Selection uses action metadata only, not prediction outcomes.

| Comparator | Comparator RMSE | Ensemble RMSE | Relative change | 95% object bootstrap | W/T/L | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.72820417 | 0.69182314 | **−5.00%** | **[−0.04765612, −0.02640160]** | **20/0/1** | **1.049e−05** |
| `state_kernel` | 0.71134238 | 0.69182314 | **−2.74%** | **[−0.02649641, −0.01313996]** | **20/0/1** | **1.049e−05** |
| `shuffled_action_control` | 0.71166163 | 0.69182314 | **−2.79%** | **[−0.02818648, −0.01279920]** | **20/0/1** | **1.049e−05** |

The source is the immutable 92-object confirmation run `33335779766`, artifact `9738998271`, artifact SHA-256 `e98f9e2687f568d0d0fcabec9ce0393a7e1b34ca3019acb1e14fdf894885a948`.

This supports action-family transfer within a known physical object. It is not zero-shot object transfer, dense 4-D geometry validation, globally fresh confirmation, or strict counterfactual evidence.

# DEFORM DLO4/DLO5 decision-identifiability result

This is a source-frozen, within-DLO held-trajectory evaluation on the 
official DEFORM DLO4 and DLO5 evaluation trajectories.

| DLO | Baseline RMSE [mm] | Certificate RMSE [mm] | Ratio | Nonfallback | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLO4 | 31.297 | 29.605 | 0.946 | 0.169 | 0.4635 |
| DLO5 | 37.398 | 36.095 | 0.965 | 0.139 | 0.5072 |

## Combined

- Certificate RMSE ratio: `0.9573`
- Mean paired trajectory improvement: `4.28%`
- 95% trajectory-bootstrap interval: `[3.04%, 5.61%]`
- Nonfallback decisions: `82` / `532`

## Claim boundary

This public-data result evaluates a frozen finite-action policy within DLO4 and DLO5. The exact certificate is conditional on the registered finite source-window support, quotient partition, and loss matrix. The pickle carrier co-locates permitted prefixes, future endpoint actions, and held internal-node outcomes; the code enforces semantic slicing but cannot provide byte-level channel separation. The result does not identify a unique physical state, prove the quotient physically correct, establish unseen-object generalization, calibrate uncertainty, or authorize deployment.

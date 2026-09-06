# DLO2 GP residual development pilot

Retrospective development only. No source-test or official evaluation read.

| Method | L1 (mm) | Improvement vs fixed ridge | Wins / 8 |
| --- | ---: | ---: | ---: |
| baseline | 7.912029 | -9.28% | 1 |
| ridge_fixed | 7.240300 | 0.00% | 0 |
| ridge_inner_selected | 7.597553 | -4.93% | 3 |
| independent_gp | 7.581486 | -4.71% | 2 |
| material_gp | 7.474714 | -3.24% | 4 |

GP hyperparameters were selected using 8 trajectories inside the original 40-trajectory fitting set.
The physical checkpoint is unchanged. The original validation set was already used in historical development.
Finite-rank GP means have a deterministic kernel-ridge equivalent. Raw covariance is not calibrated.

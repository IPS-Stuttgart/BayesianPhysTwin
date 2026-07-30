# Cloth Sim2Real v1 calibration result

## Evidence boundary

Calibration used the frozen runner and method at `886490c`. Every repeat-1
prediction was bound to the committed source-gate artifact and sealed before
its future point clouds were opened. No repeat-2 prefix or future point cloud
was opened for this result.

The calibration-gate artifact has SHA-256
`332a278d3d47d4d5a05fa8723f1e4bb3b41c690ab88a381905f46cad79d0c2b8`.

## Accuracy gate

The independent dynamic calibration gate passed:

| Calibration dynamic case | Physical symmetric L1 CD | Guarded CD | Relative change |
| --- | ---: | ---: | ---: |
| Chequered rag | 74.97 mm | 69.29 mm | -7.58% |
| Cotton rag | 97.01 mm | 85.59 mm | -11.77% |
| Linen rag | 82.49 mm | 82.49 mm | 0.00% exact fallback |
| Object-balanced | | | **-6.45%** |

The result meets all locked requirements: at least 5% object-balanced dynamic
improvement, at least two cloth wins, and no cloth regression above 5%. The
linen dynamic prefix guard rejected its correction, demonstrating the exact
fallback path on independent data.

The quasi-static result remains negative. Chequered regressed 18.60%, cotton
regressed 3.65%, and linen was nearly unchanged, for an object-balanced 7.41%
regression. Quasi-static contact-rich motion is therefore outside the
validated positive domain of this v1 method.

## Uncertainty calibration

Raw nominal 90% coordinate coverage remained only 40.3--51.5% on dynamic
trials. The predeclared calibration rule takes, for each of all six calibration
trials, the 90th percentile of absolute coordinate residual divided by raw
predictive standard deviation, then freezes the largest required multiplier.
The resulting standard-deviation multiplier is **5.05205**.

This is a conservative empirical temperature, not a formal 90% split-conformal
guarantee. With six independent trial scores, the largest finite order
statistic has rank 6/7, below 90% resolution. Target evaluation must report
both raw and scaled coverage.

## Decision

The calibration gate authorizes repeat-2 target evaluation for the primary
dynamic online-continuation claim. The method, candidate bank, admission rule,
and 5.05205 uncertainty multiplier are frozen before any target prefix is
read. Quasi-static results remain secondary negative evidence and may not be
folded into a positive headline.

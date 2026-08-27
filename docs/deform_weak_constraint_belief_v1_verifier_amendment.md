# Checker-only conformal boundary amendment

The prediction, inference, calibration, metric, and gate implementation remains
exactly at `b87eea1e9477ad2b5c4445691f8b1af1b353a701`. There was one empirical run.
The original checker invocation and its failed log are retained.

The independently factored Cholesky NEES agreed numerically with the registered
direct-solve NEES, but the discontinuous `NEES <= chi-square_3(0.9)` coverage test
disagreed for two of 480 events in one DLO2 source-conformal trajectory. A conformal
threshold is fitted from an actual order statistic, so an event can lie precisely
on the floating-point boundary. The two calculations rounded differently. This
was not a prediction, identity, calibration-scale, or statistical-gate discrepancy.

The revised checker continues to compute NLL, NEES, volume, width, calibration
scales, and continuous-score assertions independently through Cholesky factors.
Only within 64 float64 epsilons of the fixed chi-square threshold does it reproduce
the declared direct-solve/einsum operation order for binary coverage membership.
It also checks that the two NEES calculations agree numerically before applying
that convention. A synthetic 128-covariance exact-boundary regression fixture
tests this condition. No scored metric or scientific tolerance is changed.

The amended checker runs outside the untouched source archive using an explicit
`--prediction-source-root`. Its own SHA-256 is recorded separately. All 952 frozen
prediction-source file hashes must still match the original receipt. No prediction,
response bank, fit, calibration artifact, result, gate, or original failure is
rewritten or rerun by this amendment.

# Forecast-aware Sensing: Verifier-only Runtime Amendment

The sole prediction run at source commit
`f62e4ae496df1f9bcd73ef0d61e379f0e08968b3` completed all 30 trajectories, passed
its complete prediction barrier, and produced the registered scores. Its primary
advancement decision is FAIL. No prediction or scientific parameter is amended.

The first separate verifier stopped in its native replay initialization: NumPy
advanced indexing produced non-C-order action buffers, which the upstream
DEFORM network cannot flatten using its `.view` operation. The production runner
already makes owned C-order copies at this boundary. The failed verifier log is
retained; it did not complete a verification report.

This analysis-only amendment makes those same two causal input copies in the
independent verifier and adds C-order, Fortran-order, and strided-array regression
tests proving identical values, unchanged input, and owned contiguous buffers.
An explicit `--prediction-source-root` permits running the amended checker from
outside the untouched frozen prediction checkout. The checker still verifies
every original source-receipt entry and records its own file SHA-256 separately.
The first standalone invocation also exposed a missing import path for the
pre-existing metric verifier. It stopped before verification began. The amended
checker explicitly adds both the frozen `scripts` and `scripts/remote` paths;
these unsuccessful checker logs are retained with the final audit.

After clean native replay succeeded, the noise check exposed a third checker-only
defect: in-place addition rounded the independently regenerated noisy queries back
to the raw archive's float32 dtype. The producer uses out-of-place addition and
retains float64 noise. The verifier now does the same arithmetic in its separate
noise generator, with float32/float64 and shared/independent synthetic regression
tests. Exact equality is retained; no numerical tolerance is relaxed. This failed
checker log is also preserved.

No likelihood, inference, query schedule, native prediction, metric, threshold,
denominator, protected-data boundary, or decision changes. The empirical run is
not retried. Only the read-only numerical verification is repeated, including
the previously registered clean native replay. This post-score checker-runtime
repair must be disclosed with the final result and must not be described as a
pre-outcome amendment or an independent empirical replication.

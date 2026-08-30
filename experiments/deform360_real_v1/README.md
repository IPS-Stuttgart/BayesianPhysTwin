# Deform360 real-data evaluation v1

This experiment runs a bounded, read-only diagnostic on the public Deform360
release mounted at:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360
```

It is deliberately carrier-adaptive because the mounted release may contain raw
captures, released annotations, or locally materialized official annotations.
The evaluator prioritizes, in order:

1. official `pcd_clean/*.npz` point-cloud sequences;
2. fixed-identity `(T,N,3)` trajectory archives;
3. official raw or aligned `(T,16,32)` tactile fields.

The third path is a real-measurement dynamics diagnostic, but it is not a 4-D
geometry or action-conditioned intervention result. The representation used by a
run is written explicitly to `result.json` and `report.md`.

## Methods

Every one-step prediction is formed from the causal prefix only. The registered
comparison arms are:

- persistence;
- last residual / constant velocity;
- a generalized-Bayes mixture over velocity lags 1, 2, 4, and 8.

The candidate covariance is diagonal residual variance plus low-rank
between-model spread. The report includes primary error, per-dimension Gaussian
NLL, marginal 90% coverage with width, and normalized joint NEES computed with
the Woodbury identity. Cases are aggregated equally, with an object-balanced
sensitivity summary and paired bootstrap interval.

## Information and claim boundary

The protocol excludes the registered reserved-object roster before opening any
payload. Carrier selection uses file names and the registered development-object
order, never achieved scores. Raw data, videos, tactile arrays, point clouds, and
trajectory arrays are not uploaded. Only compact JSON, CSV, Markdown, and
provenance records are retained.

This is a retrospective non-confirmatory evaluation. It does not establish the
official Deform360 benchmark, fresh-object confirmation, causal intervention
benefit, deployment calibration, safety, state of the art, or a paper claim.

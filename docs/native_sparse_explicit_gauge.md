# Native sparse explicit-gauge inference

BayesianPhysTwin can consume claim-bearing Prob4D observation factors without
materializing the compatibility tensor of shape `M x 3 x 7K`.

The native path keeps one local `3 x 7` gauge Jacobian and one gauge index for
each active observation row. It retains the complete joint `7K x 7K` gauge
prior, including cross-window covariance, and accumulates the state/gauge/bias
normal-equation blocks directly.

## Strict artifact path

```python
from bayesian_phystwin.sparse_explicit_gauge_prob4d import (
    update_claim_bearing_native_sparse_explicit_gauge_from_artifacts,
)

update = update_claim_bearing_native_sparse_explicit_gauge_from_artifacts(
    validated_bundle,
    sparse_stack,
    physical_linearization,
    physical_prediction_xyz_m=physical_prediction_xyz_m,
)
```

Admission is identical to the dense reference bridge:

- the claim-bearing factor envelope and provider attestation are validated;
- the active sparse rows are independently reconstructed from the validated
  neutral bundle;
- every numerical array, row identity, group identity, gauge identity, causal
  cutoff, and the complete joint gauge prior must match exactly;
- conditional point covariance and explicit gauge covariance are consumed once;
- association probability remains a generalized-Bayes row power rather than a
  source-reliability score; and
- accepted and exact-fallback results bind the same immutable provider,
  calibration, runtime, physical-linearization, and sparse-stack lineage.

The observation batch deliberately contains zero dense gauge columns. Gauge
ownership is held by `SparseGaugeDesignV1`, which contains the block-local
Jacobians, indices, IDs, and complete joint prior.

## Numerical semantics

`update_sparse_prior_aware_gauge_belief` preserves the existing prior-aware
solver's:

- grouped nominal/outlier Student-t mixture;
- source-reliability precision scaling and provider-final likelihood powers;
- prior-aware Schur identifiability test;
- shared, view, and independent-anchor bias variables;
- IRLS fixed point and Cholesky posterior solve;
- working Gauss-Newton covariance and exact-mixture-Hessian diagnostic;
- physical-response-relative update limit; and
- fail-closed exact fallback reasons.

Only the gauge design representation and accumulation strategy change. Dense
and sparse paths are retained in parallel so deterministic parity tests can
serve as a reference oracle.

## Memory accounting

For `M` active rows and `K` gauges, the compatibility bridge requires

```text
M * 3 * 7K * sizeof(float64)
```

bytes for its first dense gauge tensor. The native path does not allocate that
tensor. It records the avoided byte count in the adapter lineage and solver
diagnostics. Its persistent gauge storage is the `M x 3 x 7` local design plus
the complete `7K x 7K` prior and the final state-plus-nuisance normal system.

This removes the former 256 MiB compatibility limit. It does not make the final
joint gauge system independent of `K`; very large gauge graphs can still require
a sparse linear-algebra backend in a later version.

## Evidence boundary

The implementation is validated by deterministic dense/sparse parity,
observation-row permutation, gauge-block permutation, fallback, provenance, and
memory-accounting tests. These tests establish numerical and contract
conformance only. They do not provide new observation-quality, physical-object,
harmful-update, calibration, or Causal4D intervention evidence.

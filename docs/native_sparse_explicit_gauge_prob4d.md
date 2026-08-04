# Native sparse explicit-gauge inference

Prob4D exports one local `3 x 7` gauge Jacobian per observation row, a gauge
index per row, and one complete joint covariance over all gauge windows. The
original BayesianPhysTwin compatibility bridge expanded those local blocks into
an `M x 3 x 7K` dense tensor. Most entries were exactly zero, so long causal
prefixes could hit the bridge's 256 MiB allocation limit before inference.

The native path keeps the producer representation sparse:

```python
from bayesian_phystwin.sparse_explicit_gauge_prob4d import (
    update_claim_bearing_sparse_explicit_gauge_from_artifacts,
)

update = update_claim_bearing_sparse_explicit_gauge_from_artifacts(
    validated_bundle,
    sparse_stack,
    physical_linearization,
    physical_prediction_xyz_m=physical_prediction,
)
```

The solver whitens each local block with its metric point covariance, then
accumulates the exact state/gauge/bias normal-equation blocks. The normal matrix
remains dense because the supplied joint gauge prior may couple any two
windows; only the redundant per-row zero blocks are removed. Robust grouped
Student-t responsibilities, prior reliability, generalized-Bayes row power,
state identifiability, trust-region fallback, and returned posterior covariance
retain the dense solver's semantics.

The adapter independently validates the same claim-bearing envelope and binds:

- the observation and physical-linearization artifact identities;
- the complete joint gauge prior and covariance-calibration identities;
- association probability, source reliability, nominal probability, and
  provider composite weights as separate quantities;
- row identities, gauge indices, canonical views, runtime revision, and
  simulator/action-prefix lineage; and
- the dense-equivalent byte count together with an explicit record that no
  dense gauge design was allocated.

The dense bridge remains the numerical oracle for small problems. Tests require
matching inference decisions, state and nuisance coefficients, robust weights,
identifiable subspaces, and posterior covariance under correlated cross-window
priors. Malformed or lineage-inconsistent factors fail before an innovation is
formed.

## Scientific boundary

This is an enabling implementation, not evidence of state-of-the-art accuracy.
It makes longer and more heavily overlapping causal Prob4D prefixes tractable
without weakening covariance accounting. A scientific run must still use a
locked source-only admission rule, exact physical fallback, grouped calibration,
and a genuinely independent target cohort. Causal4D consumes only the accepted
or fallback BayesianPhysTwin belief and must not assimilate the visual factors a
second time.

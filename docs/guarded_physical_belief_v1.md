# Guarded physical-belief and Causal4D provider API

Bayesian-PhysTwin now exposes a narrow, versioned boundary for downstream
causal inference. Causal4D should consume `PhysicalBeliefV1` artifacts or a
`PhysTwinReplayProviderV1`; it should not import experiment-specific or
underscore-prefixed Bayesian-PhysTwin helpers.

## Decision stages

The implementation distinguishes three decisions that must not be conflated:

1. **candidate validity**: the numerical update is causal, identifiable,
   finite, locally plausible, and passes the nonlinear-closure check;
2. **guard acceptance**: a frozen, source-fitted prospective certificate
   accepts the candidate for every preregistered primary loss;
3. **belief selection**: the complete candidate belief is selected only when
   both preceding decisions are true.

`GuardDecisionV1` records the certificate, development partition, observation
artifact, physical linearization, optional nonlinear-closure artifact, and
primary losses. `select_physical_belief` returns the exact immutable baseline
object on rejection. It does not synthesize fallback by zeroing correction
coefficients or covariance.

## Complete belief contract

`PhysicalBeliefV1` is a non-pickled, content-addressed particle belief over:

- endpoint position and velocity;
- physical parameter particles and normalized weights;
- readout/process discrepancy means and variances;
- provider identity and provenance metadata.

`PhysicalBeliefSelectionV1` binds the baseline, candidate, guard decision, and
selected belief IDs. On fallback,

```text
selected_belief_id == baseline_belief_id
```

and the exact baseline object is reused.

## Row-bound physical linearization

`PhysicalLinearizationV1` binds observation rows to physical Jacobians using the
full `(frame, entity, view, window)` identity tuple. It also binds the immutable
baseline belief, action prefix, simulator revision, future-query Jacobian, and
the action-conditioned physical response. The trust-region scale is derived
from this response artifact rather than supplied as an unrelated scalar.

`build_gauge_aware_batch_from_artifacts` fails closed on a row permutation or
artifact mismatch before constructing an update batch.

## Prior-aware inference

`update_prior_aware_gauge_belief` complements the existing strict-minimax
projection with a prior-aware Schur-complement mode. Tight independently
calibrated gauge or bias priors can therefore resolve a direction continuously;
diffuse priors retain the conservative fallback behavior.

The robust update uses a group-level nominal/outlier multivariate Student-t
mixture:

- association probability is not reused as prior reliability;
- residual-independent nominal probability is used inside the mixture;
- composite likelihood weight remains a separate information power;
- degrees of freedom must exceed two when inputs are covariance matrices;
- the reported matrix is explicitly a working Laplace/IRLS covariance and must
  be evaluated prospectively for coverage.

## Nonlinear closure

`evaluate_nonlinear_closure` compares the locally predicted future query with a
nonlinear PhysTwin replay. A candidate that exceeds both the absolute and
relative closure tolerances is invalid before the prospective regret guard is
applied.

## Provider usage

```python
from bayesian_phystwin.causal4d_provider_v1 import (
    ProviderManifestV1,
    build_physical_belief_from_provider,
    save_physical_belief,
)

manifest = ProviderManifestV1(provider_revision=exact_git_sha)
belief = build_physical_belief_from_provider(provider)
save_physical_belief("twin_belief.npz", belief)
```

Frozen experiments should continue to record exact repository revisions.
Normal development can validate the provider schema and capabilities instead
of importing one implementation snapshot's private functions.

# Basis-invariant physical-query subspace selection

## Status

`bayesian_phystwin.invariant_query_subspace` is a prospective numerical and
identifiability utility. It does not change the frozen dense or native-sparse
prior-aware solvers, the strict v2 admission layer, any registered experiment,
or any historical artifact identity.

The currently registered Deform360 10+12-object method does not use this module.
A later protocol must select it explicitly before opening its target cohort.

## Problem

The existing prior-aware solvers diagonalize nuisance-marginalized conditional
information and apply information, identifiability, and physical-query
thresholds to individual eigenvectors. This is well defined when information
eigenvalues are separated. Inside an exactly or numerically repeated eigenspace,
however, the individual eigenvectors are arbitrary. Rotating state coordinates
can therefore change which eigenvectors pass a query-sensitivity threshold even
though the physical information and query are unchanged.

The physically meaningful object is the selected subspace projector, not one
particular eigenbasis.

## Selection rule

Let

- `K` be state information before nuisance marginalization;
- `C` be nuisance-marginalized conditional state information;
- `P` be the state prior covariance;
- `Q` be the declared physical-query Jacobian; and
- `S` be the positive-semidefinite square root of `P`.

The utility applies three spectral projectors:

1. **Information projector.** In prior-standardized coordinates, retain the
   spectral subspace of `S.T @ C @ S` above the frozen relative-information
   threshold. Repeated and near-repeated eigenvalues are selected together
   within the declared spectral tolerance.
2. **Identifiability projector.** Inside that information subspace, solve the
   generalized problem for `C v = lambda K v` and retain the spectral subspace
   whose worst admitted generalized eigenvalues meet the identifiability
   threshold.
3. **Query projector.** Inside the identifiable subspace, diagonalize the query
   Gram matrix and retain the spectral subspace above the frozen relative query
   sensitivity.

Each retained projector is converted to a deterministic basis by projecting
canonical coordinate axes and applying two-pass modified Gram--Schmidt. The
returned physical span is invariant to arbitrary rotations inside repeated
information eigenspaces. The basis is deterministic for a fixed coordinate
system, while scientific comparisons should use the projector or decoded
physical query rather than column signs.

## Usage

```python
from bayesian_phystwin.invariant_query_subspace import (
    InvariantQuerySubspaceConfigV1,
    select_invariant_query_subspace,
)

selection = select_invariant_query_subspace(
    known_information,
    conditional_information,
    state_prior_covariance,
    query_state_jacobian,
    config=InvariantQuerySubspaceConfigV1(
        minimum_information_fraction=1e-4,
        minimum_identifiable_fraction=0.10,
        minimum_query_sensitivity_fraction=1e-3,
    ),
)

if selection.admissible:
    reduced_state_jacobian = selection.project_state_jacobian(
        state_jacobian
    )
```

A prospective solver can use `state_mapping` as the retained state basis, run
an already validated inference method with an identity prior in reduced
coordinates, and then use `lift_state_mean` and `lift_state_covariance` to
restore the original coordinates while preserving untouched prior variance
outside the selected subspace.

## Numerical contract

The implementation:

- validates finite symmetric information and prior matrices;
- fails closed on non-positive-semidefinite spectral operators outside the
  declared tolerance;
- requires positive-definite known information on the selected information
  subspace;
- never uses a pseudoinverse, hidden diagonal jitter, or outcome-dependent
  threshold repair;
- returns immutable arrays; and
- reports separate information, generalized-identifiability, and query spectra.

The tests include exact repeated eigenvalues, near-repeated eigenvalues,
orthogonal and nonorthogonal state reparameterizations, information and
identifiability filtering, zero-query fallback, lifting, immutability, and
adversarial contract inputs.

## Claim boundary

These tests establish algebraic and numerical invariance only. They do not show
improved provider competence, physical prediction, uncertainty coverage,
harmful-update risk, Causal4D counterfactual benefit, deployment safety, or
state of the art. Promotion requires a separately frozen experiment comparing
this selector with the historical basis-dependent path on independent physical
objects or sessions, under the same guard and exact fallback.

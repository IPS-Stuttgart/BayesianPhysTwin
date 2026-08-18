# Formal guarded-inference guarantees v1

## Scope

This document records three invariants used by the supported
`bayesian_phystwin.inference.v1` boundary. They are deliberately narrower than a
physical-accuracy or deployment-safety theorem. The guarantees concern complete
belief routing and algebraically equivalent local uncertainty representations
under the stated preconditions.

The executable regressions live in
`tests/test_portable_contracts_formal_guarantees_v1.py` and are part of the
stable portable-contract coverage pattern.

## Proposition 1: complete-belief fallback noninterference

Let `B0` be the exact caller-owned baseline belief and `Bc` a distinct complete
candidate belief. Let the guard decision bind both artifact identities. The
stable finalizer selects

```text
Bc  when inference_admissible and regret_guard_accepted
B0  otherwise.
```

For either rejection route, the selected Python object is `B0` itself, not a
new object reconstructed from numerically equal arrays. Consequently every
field owned by the complete belief—mean, covariance, particle or hypothesis
weights, material identity, discrepancy state, nuisance moments, and
provenance—remains the caller's baseline field by object containment.

This is stronger than array equality and prevents a rejected update from
silently retaining candidate covariance, metadata, or posterior weights. The
result record binds the baseline, candidate, guard decision, selection, selected
artifact, and `exact_fallback` flag.

### Preconditions

- both complete beliefs expose valid lowercase SHA-256 artifact identities;
- the guard decision binds those exact identities;
- inference admissibility in the decision equals the candidate-inference value;
- application code consumes `selected_belief`, rather than separately mixing
  fields from the baseline and candidate.

### Boundary

Exact fallback does not prove that the baseline is accurate or safe. It proves
only noninterference by the rejected candidate relative to the declared
baseline object.

## Proposition 2: retained-basis reparameterization invariance

Consider one admitted local Gaussian update with prior
`x ~ N(m, P)`, linear observation model `y = Hx + e`, and
`e ~ N(0, R)`. For any invertible retained-coordinate transform `x = Tz`, use

```text
m_z = T^{-1} m
P_z = T^{-1} P T^{-T}
H_z = H T.
```

Performing the same update in `z` coordinates and mapping the posterior back by
`T` yields the same posterior mean and covariance as the direct update in `x`
coordinates, up to floating-point roundoff.

The invariant justifies treating an invertible change of basis inside an already
retained, prior-supported state subspace as representation rather than new
scientific information. It does not permit changing the retained rank,
identifiability threshold, nuisance model, prior, likelihood, or physical query.

The regression uses linear solves and the Joseph covariance form; it does not
rely on explicit matrix inversion.

## Proposition 3: dense/low-rank query-covariance parity

Let a covariance be represented as

```text
P = D + F F^T,
```

where `D` is the conditional covariance and `F` is a shared low-rank factor. For
any fixed linear query matrix `Q`,

```text
Q P Q^T = Q D Q^T + (Q F)(Q F)^T.
```

Thus dense and factorized representations of the same covariance must produce
identical query covariance up to numerical roundoff. The factor must not also be
added into `D`; doing so would count the shared uncertainty twice.

This proposition concerns representation parity only. It does not establish
that the covariance is calibrated, that the factor semantics are correct, or
that the corresponding update improves a physical query.

## Cross-repository consequence

A downstream Causal4D consumer should receive the selected complete
BayesianPhysTwin belief and its routing receipt. It must not reinterpret raw
Prob4D factors after BayesianPhysTwin has rejected them. Under exact fallback,
no evidence from the rejected update is consumed as an accepted physical-belief
change.

## Verification

Run the focused regression with:

```bash
pytest -q tests/test_portable_contracts_formal_guarantees_v1.py
```

The repository's existing guarded-inference, portable-contract, provider, and
installed-wheel suites remain authoritative for serialization, API isolation,
and ecosystem compatibility.

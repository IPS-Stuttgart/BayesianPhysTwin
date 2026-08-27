# Query-quotient belief lifting v1

## Purpose

A physical observation can identify the value of a registered downstream query
without identifying a unique latent physical cause. Treating one member of that
query-equivalence class as if it had been observed would add unsupported physical
specificity. Ignoring the evidence entirely would discard information that is
valid at the query level.

`bayesian_phystwin.query_quotient_belief_v1` implements the intermediate case. A
pre-registered partition maps each finite physical hypothesis to one query class.
The evidence updates only the masses of those classes. BayesianPhysTwin then
constructs a complete hypothesis belief by preserving the prior conditional
belief inside every class.

This makes the existing query-identifiability principle executable while keeping
candidate construction separate from admission and exact fallback.

## Registered objects

Let the finite physical hypotheses be `i = 0, ..., H-1` with prior weights
`p_i`. A registered map `c(i)` assigns every hypothesis to one of `K` contiguous
query classes. The class definition must be frozen independently of the outcome
used to form the posterior. In the three-repository workflow:

- Causal4D can own the intervention/query semantics and the registered class map;
- Prob4D can own uncertainty-bearing evidence for the quotient-class masses;
- BayesianPhysTwin owns the complete candidate lift, its information audit, the
  downstream ambiguity certificate, and the later candidate-or-fallback decision.

The module does not infer the partition from numerical closeness. Tolerance-based
pairwise grouping need not be transitive, so the caller must supply a canonical
partition explicitly.

## Minimum-information complete-belief lift

Write

```text
p_C(c) = sum_{i: c(i)=c} p_i
```

for the prior class masses and let `nu_c` be the posterior class masses. For every
class with positive `nu_c`, `p_C(c)` must be positive. The canonical full belief
is

```text
q_i* = nu_{c(i)} p_i / p_C(c(i)).
```

Thus `q*(i | c) = p(i | c)`: evidence changes only class masses and leaves every
within-class prior odds ratio unchanged.

For any other full belief `q` with the same quotient posterior, the KL chain rule
is

```text
KL(q || p)
  = KL(q_C || p_C)
    + sum_c q_C(c) KL(q(. | c) || p(. | c)).
```

The second term is reported as **unsupported specificity**. It is information in
the full physical posterior that is not determined by the quotient posterior.
The canonical lift has unsupported specificity zero and is the unique minimizer
of `KL(q || p)` over all full beliefs with class masses `nu`.

This is a standard relative-entropy projection result. The contribution of this
contract is not a new KL identity; it is the binding of that identity to a
registered physical-query quotient, a complete-belief candidate, an explicit
unsupported-specificity diagnostic, an ambiguity envelope, and the existing
exact-fallback boundary.

## Downstream ambiguity envelope

A complete array of weights is syntactically necessary for recursive inference,
but the quotient posterior may not identify every downstream physical quantity.
For a scalar hypothesis functional `f_i`, every full lift of `nu` satisfies

```text
sum_c nu_c min_{i: c(i)=c} f_i
  <= E[f] <=
sum_c nu_c max_{i: c(i)=c} f_i.
```

These bounds are exact. The implementation computes them componentwise for one
or more registered endpoints. A zero-width interval means that endpoint is
constant inside every posterior-supported class and is therefore determined by
the quotient belief. A nonzero interval exposes unresolved physical ambiguity;
it must not be hidden by the canonical lift's single numerical expectation.

For vector endpoints the intervals are componentwise. Extrema for different
components need not be jointly attainable.

## Example

```python
import numpy as np

from bayesian_phystwin.query_quotient_belief_v1 import (
    minimum_information_query_lift,
    query_ambiguity_envelope,
)

prior = np.array([0.10, 0.20, 0.30, 0.40])
classes = np.array([0, 0, 1, 1])
quotient_posterior = np.array([0.75, 0.25])

lift = minimum_information_query_lift(
    prior,
    classes,
    quotient_posterior,
)

# [0.25, 0.50, 3/28, 1/7]; prior odds are preserved within each class.
print(lift.lifted_weights)
print(lift.information.unsupported_specificity_nats)  # numerical zero

# Query value is constant inside each registered class.
query = np.array([10.0, 10.0, 20.0, 20.0])
print(query_ambiguity_envelope(quotient_posterior, classes, query).summary())

# A latent physical coordinate differs inside both classes and stays ambiguous.
physical_coordinate = np.array([0.0, 2.0, -1.0, 3.0])
envelope = query_ambiguity_envelope(
    quotient_posterior,
    classes,
    physical_coordinate,
)
print(envelope.lower, envelope.upper)  # [-0.75], [2.75]
```

All returned arrays are immutable. Inputs are rejected when probabilities are
invalid, class labels are noncanonical, a positive posterior class has zero prior
support, or an arbitrary posterior violates prior support.

## Admission and claim boundary

The quotient lift constructs a candidate only. It does not establish that the
registered quotient is physically correct, identify a unique physical cause,
authorize deployment, validate a provider, calibrate uncertainty, prove
held-out-action transport, or certify safety. A candidate still has to pass the
separately frozen BayesianPhysTwin support, calibration, selective-risk, and
ambiguity rules. Failure returns the exact caller-owned physical fallback.

The implementation tests are deterministic mechanism tests. They establish the
algebraic and software contract; they are not empirical evidence for Causal4D,
Prob4D, a real provider, unseen-object transfer, or physical intervention value.

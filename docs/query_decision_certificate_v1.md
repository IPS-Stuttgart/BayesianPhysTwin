# Query decision certificate v1

## Purpose

A registered observation may update only query-equivalence-class masses without
identifying one latent physical hypothesis. That ambiguity need not force
abstention when every supported complete belief leads to the same downstream
decision, or when one decision has acceptably small worst-case regret.

`bayesian_phystwin.query_decision_certificate_v1` computes that certificate
exactly for finite hypotheses and finite actions. It is designed to consume the
same prior support, class map, and quotient posterior used by
`query_quotient_belief_v1`.

## Registered objects

Let:

- `p_i` be the physical prior over finite hypotheses;
- `c(i)` be the outcome-independently registered quotient class;
- `lambda_c` be the posterior quotient mass;
- `L(i,a)` be the registered loss of action `a` under hypothesis `i`; and
- `epsilon` be a regret tolerance fixed independently of target outcomes.

The feasible complete beliefs are

```text
Q_lambda^p
  = {q: sum_{i:c(i)=c} q_i = lambda_c for every c, and q << p}.
```

Only positive prior support is feasible. Positive quotient mass for a class with
zero prior mass fails closed.

## Exact pairwise gap and regret

For actions `a` and `b`,

```text
Delta_bar(a,b)
  = sup_{q in Q_lambda^p} E_q[L(H,a)-L(H,b)]
  = sum_c lambda_c
      max_{i:c(i)=c, p_i>0} (L(i,a)-L(i,b)).
```

The equality is exact because each quotient-class conditional may concentrate
on its own supported maximizer.

The exact worst-case regret of `a` is

```text
Reg_bar(a)
  = sup_q (E_q[L(H,a)] - min_b E_q[L(H,b)])
  = max_b Delta_bar(a,b).
```

The implementation returns all pairwise gaps, all action regrets, the
deterministic lowest-index minimax action, an `epsilon`-admissibility mask, and a
zero-regret robust-optimality mask.

## Example

```python
import numpy as np

from bayesian_phystwin.query_decision_certificate_v1 import (
    query_decision_certificate,
)

prior = np.array([0.10, 0.20, 0.30, 0.40])
classes = np.array([0, 0, 1, 1])
quotient_posterior = np.array([0.60, 0.40])

# Rows are hypotheses; columns are candidate actions.
loss = np.array(
    [
        [0.0, 2.0, 3.0],
        [1.0, 0.5, 2.0],
        [0.2, 1.0, 0.0],
        [2.0, 0.1, 0.4],
    ]
)

certificate = query_decision_certificate(
    prior,
    quotient_posterior,
    classes,
    loss,
    regret_tolerance=0.10,
)

print(certificate.minimax_action_index)
print(certificate.minimax_worst_case_regret)
print(certificate.tolerance_admissible_action_mask)
```

## Use by the guard

The certificate is scoped to one registered action set, loss, quotient, and
tolerance. A guard may use it as follows:

1. validate source support, lineage, covariance semantics, and the quotient;
2. compute the minimum-information complete candidate;
3. compute the decision certificate over every prior-supported quotient lift;
4. admit the candidate only for the registered decision when the required
   regret criterion passes; and
5. otherwise return the exact caller-owned physical fallback.

Passing this certificate does not authorize arbitrary state interpretation,
other actions, other losses, or other queries.

## Interpretation

The certificate separates three levels:

1. **State identification:** one latent physical explanation is determined.
2. **Query identification:** a declared physical endpoint is determined.
3. **Decision identification:** the endpoint may remain ambiguous, but one
   action is robustly optimal or has bounded worst-case regret.

The third level can hold when the first two do not.

## Claim boundary

The result is exact only for the supplied finite hypotheses, positive prior
support, registered quotient masses, action set, and loss matrix. It does not:

- establish that the quotient is physically correct;
- validate an observation provider;
- identify a unique physical cause;
- calibrate a predictive distribution;
- justify the loss function or regret tolerance;
- establish held-out intervention transport;
- authorize deployment outside the registered decision; or
- certify safety.

Every returned array is immutable. Inputs fail closed when probabilities,
support, class labels, losses, dimensions, or tolerance are malformed.

The composed query-conditional trust router wraps the certificate together
with its registered loss matrix, content-addresses that complete finite
problem, and recomputes every gap, regret, and mask before admission. This
preserves the established certificate API while preventing substitution of an
unregistered solution. See `docs/query_conditional_trust_v1.md`.

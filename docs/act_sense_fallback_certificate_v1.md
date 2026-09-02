# Exact act--sense--fallback certificate v1

## Problem

Let `H` be a finite physical-hypothesis set, `A` a finite set of terminal
physical actions, and `C` an outcome-independently registered quotient. The
observation fixes class masses `lambda_c` but not the conditional distribution
inside each class. The compatible complete beliefs are therefore

```text
Q_lambda = {q: sum_{h in c} q_h = lambda_c for every c, q << p}.
```

A direct decision certificate can already evaluate an action without selecting
one unsupported state inside a class. The active extension adds deterministic
diagnostic probes.

For probe `s`, let `g_s(h)` be its registered finite outcome under hypothesis
`h`, and let `c_s >= 0` be its cost. A contingent plan `pi` maps every possible
probe outcome to a terminal action. Its complete hypothesis-wise loss is

```text
L_(s,pi)(h) = c_s + ell(h, pi(g_s(h))).
```

A direct action is the special case with no probe.

## Exact reduction

Form the finite plan set

```text
P = A union {(s, pi): s is a probe and pi maps outcomes of s to A}.
```

For plans `u` and `v`, the exact largest compatible loss gap is

```text
Delta_bar(u,v)
  = sum_c lambda_c max_{h in c, p_h > 0} [L_u(h) - L_v(h)].
```

The exact worst-case regret is

```text
Reg_bar(u) = max_v Delta_bar(u,v).
```

This is the existing query-decision theorem applied to complete contingent
plans. No within-class point state, Jeffrey lift, or post-probe posterior is
needed. The entire sensing policy is certified before the probe is executed;
after the registered outcome is observed, its frozen action map is applied.

With `A` terminal actions and `K_s` realized outcomes for probe `s`, the number
of plans is

```text
P_count = A + sum_s A ** K_s.
```

The v1 implementation explicitly enumerates this set and then applies the exact
finite certificate, giving `O(H * P_count ** 2)` time. It fails closed if the
declared plan cap is exceeded.

## Operational rule

Direct plans are enumerated before sensing plans, so exact numerical ties prefer
acting without an unnecessary probe. Let `u*` be the lowest-index minimax plan
and `epsilon` the registered regret tolerance.

- If `Reg_bar(u*) <= epsilon` and `u*` is direct, **act**.
- If `Reg_bar(u*) <= epsilon` and `u*` is contingent, **sense**, then execute
  its frozen outcome-conditioned action.
- Otherwise, return the exact caller-owned **fallback** action.

The fallback is not inferred or approximated by the certificate.

## Why this is not information gain

A probe can reveal many latent-state bits while leaving every action comparison
unchanged. Conversely, one binary outcome can determine the action. The
controlled occluded-rope experiment records exactly this separation:

- the three-outcome texture camera has entropy gain `ln(3)`;
- the two-outcome side tug has entropy gain `ln(2)`;
- the camera's best contingent-plan worst-case regret is `1.55`;
- the tug's is `0.20`, and it is selected at tolerance `0.25`.

Thus the objective is decision identification, not full-state reconstruction or
maximum entropy reduction.

## Claim boundary

The result is exact only for the supplied finite hypotheses, prior support,
quotient masses, terminal loss matrix, deterministic probe outcomes, probe
costs, enumerated contingent plans, and tolerance. It does not establish that a
probe can be executed and reset physically, that the loss transfers to a target
system, that a provider is calibrated, or that an accepted plan is safe.

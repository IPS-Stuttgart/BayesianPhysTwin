# Support-robust act--sense--fallback certificate v1

## Purpose

The exact contingent-plan certificate controls regret only over the registered
finite physical support. This extension makes a declared support miss explicit.
It asks whether a direct action or complete probe-then-act plan remains
admissible when up to `epsilon` probability mass may come from unknown physics.

## Ambiguity class

Let `Q(lambda)` be the existing set of complete represented beliefs compatible
with the registered quotient masses and prior support. Let `P` be the finite
roster of direct and probe-contingent plans. For every plan `p`, the unknown
physical domain supplies only a loss interval

```text
L_unknown(p) in [lower[p], upper[p]].
```

The complete ambiguity class contains all mixtures

```text
(1-rho) q + rho r,
q in Q(lambda), 0 <= rho <= epsilon,
```

where `r` may induce any complete plan-loss vector in the registered
axis-aligned box. This rectangular envelope is intentionally explicit; it is
not inferred by the certificate.

## Exact pairwise theorem

For plans `p` and `b`, let

```text
Delta_0(p,b)
```

be their exact represented-support worst-case loss gap, and define

```text
M(p,b) = upper[p] - lower[b].
```

Then the exact worst-case gap over the contaminated ambiguity class is

```text
Delta_epsilon(p,b)
  = Delta_0(p,b) + epsilon * max(0, M(p,b) - Delta_0(p,b)).
```

For fixed unknown mass `rho`, represented and unknown contributions separate:

```text
(1-rho) Delta_0(p,b) + rho M(p,b).
```

This expression is affine in `rho`, so the maximum over `[0, epsilon]` occurs
at zero or `epsilon`, yielding the formula. The support-robust regret of a plan
is the maximum pairwise gap over all benchmark plans.

## Maximum tolerated support miss

For a registered regret tolerance `tau`, the module solves every pairwise
linear inequality and reports the largest `epsilon` in `[0,1]` for which each
plan remains admissible. This quantity is a sensitivity budget, not an estimate
of real domain shift.

## Operational policy

1. Enumerate direct actions and deterministic probe-outcome action maps.
2. Compute exact represented-support plan regret.
3. Expand it by the declared support-miss envelope.
4. Select the lowest-regret plan only when its robust regret is at most the
   registered tolerance.
5. Otherwise return the caller-owned fallback action exactly.

The policy can therefore switch from a cheap probe to a safer probe as the
support-miss bound grows, before eventually falling back.

## Limits

The guarantee is conditional on the finite represented support, quotient,
terminal losses, deterministic probe maps, scalar probe costs, unknown-domain
loss box, miss-probability bound, and tolerance. It does not estimate or
validate those inputs, establish a distribution-shift rate, justify physical
probe execution, or certify deployment safety.

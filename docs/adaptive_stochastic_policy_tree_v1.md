# Adaptive stochastic policy-tree certificates

## Question

A physical twin may not be able to certify a terminal action from its current
partial belief. A single fixed probe can also be insufficient. The relevant
question is then not whether enough data can identify the complete physical
state, but whether a **complete adaptive sensing-and-action policy** can be
certified before any probe is executed.

This module treats a finite-depth policy tree as one ex-ante meta-action:

- a leaf executes one registered terminal action;
- an internal node executes one registered stochastic probe;
- each registered probe outcome selects a child tree;
- a probe may be used at most once on one root-to-leaf path;
- every branch is fixed before sensing begins.

No terminal action or second probe is re-optimized after an outcome is observed.

## Exact certificate

For hypothesis `h`, terminal action `a`, probe `u`, outcome `z`, likelihood
`K_u[h,z]`, and sensing cost `c_u`, define recursively

```text
V_h(a) = L[h,a]
V_h(u,{pi_z}) = c_u + sum_z K_u[h,z] V_h(pi_z).
```

Let quotient class `c` have registered posterior mass `lambda[c]`. For complete
policy trees `pi` and `rho`,

```text
Delta(pi,rho)
  = sum_c lambda[c]
      max_(h in c, prior[h] > 0) (V_h(pi) - V_h(rho)).
```

The exact common-comparator worst-case regret is

```text
Reg(pi) = max_rho Delta(pi,rho).
```

The maximum ranges over the same registered direct and sensing policy-tree
class. This avoids comparing a direct action against one oracle class and a
sensing tree against another.

The implementation emits one policy only when the minimax tree is unique and
its regret is within the registered tolerance. Otherwise it returns the exact
caller-owned fallback action before any sensing cost is incurred.

## Finite enumeration and safe compression

Depth and roster sizes are explicitly capped. Enumeration is exact before two
finite-support reductions:

1. trees with equal represented-hypothesis expected-loss vectors are reduced to
   the first canonical representative;
2. a sensing tree whose expected loss is componentwise dominated on every
   represented hypothesis is removed.

Direct action trees, including fallback, are always retained. These reductions
preserve the finite-support minimax result. They define the retained policy
class for any subsequent out-of-support or conformal analysis; policies removed
as represented-support equivalent are not silently reintroduced later.

## Trajectory-conformal whole-tree envelope

The branch is stacked on the complete-plan conformal certificate. Logged data
can evaluate every frozen adaptive tree by following its precommitted branches.
For calibration trajectory `j`, decision `d`, and retained tree `pi`, let
`R[j,d,pi]` be realized regret and `B[pi]` its registered finite-support bound.
With fixed positive scales `s[pi]`, the trajectory score is

```text
S[j] = max_(d,pi in C) max(0, R[j,d,pi] - B[pi]) / s[pi].
```

The split-conformal order statistic supplies simultaneous inflation over every
registered decision and retained policy tree on one exchangeable future
trajectory. The full calibrated tree is chosen before sensing. If the calibrated
minimum is nonunique, infinite, or over tolerance, fallback is returned before
probing.

This is trajectory-marginal validity under exchangeability. It is not pointwise
conditional validity, unseen-object transport, or a deployment-safety theorem.

## Strict adaptive separation

The controlled study has 16 physical hypotheses with four binary latent
variables:

- a routing variable that determines which task coordinate matters;
- task coordinate `x`;
- task coordinate `y`;
- a decision-irrelevant nuisance variable.

The nuisance probe is both cheaper and more accurate than the three relevant
probes, and therefore has the largest state-information gain. Yet it cannot
reduce decision regret.

At regret tolerance `0.20`:

- direct actions return fallback at regret `0.45`;
- every at-most-one-probe policy returns fallback;
- every fixed nonadaptive two-probe policy class is no better than fallback;
- a depth-two adaptive tree reaches regret `0.129` by measuring the route and
  then measuring only `x` or `y` on the corresponding branch;
- the nuisance probe is never selected;
- every noisy outcome has positive probability under every hypothesis, so the
  complete physical state remains unidentified at every leaf.

Thus adaptive decision identification is strictly weaker and cheaper than full
state identification.

## Relation to neighboring modules

- `act_sense_fallback_certificate_v1` certifies direct actions and one-step
  deterministic contingent plans.
- `support_robust_act_sense_fallback_certificate_v1` permits declared bounded
  support miss for those complete plans.
- `conformal_complete_plan_certificate_v1` calibrates whole one-step plans from
  complete trajectories.
- `adaptive_stochastic_policy_tree_v1` adds stochastic probes, multiple adaptive
  sensing stages, one common comparator over the complete tree class, and a
  conformal wrapper that selects the whole tree before probing.

The module does not replace target-directed linear experiment design or causal
attribution. It addresses a different layer: finite-horizon sensing and action
selection once a physical hypothesis/quotient model and probe likelihoods have
been registered.

## Reproduction

```bash
python -m pytest -q \
  tests/test_adaptive_stochastic_policy_tree_v1.py \
  tests/test_adaptive_stochastic_policy_tree_controlled_v1.py

python -m experiments.adaptive_stochastic_policy_tree_v1.run \
  --output /tmp/adaptive-stochastic-policy-tree-v1.json
```

## Claim boundary

The exact result is conditional on the supplied finite support, quotient masses,
probe likelihoods, sensing costs, terminal losses, finite depth, compression
rules, and regret tolerance. The conformal result additionally requires a tree
roster frozen before calibration and exchangeable complete trajectories. The
method does not validate the hypothesis family, observation model, probe
physics, reset semantics, action losses, target transport, online execution,
deployment, or safety.

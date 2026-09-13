# Active decision-identifying probes v1

`bayesian_phystwin.active_decision_probe_v1` extends the finite
query-decision certificate from passive **act or fall back** behavior to a
finite active experiment:

1. the current observation fixes only posterior mass on registered quotient
   classes;
2. a source-registered probe produces one of finitely many outcomes;
3. the terminal action may depend on the observed outcome; and
4. the certificate asks for the contingent policy with the smallest
   worst-case expected regret over every complete belief compatible with the
   current quotient masses and positive prior support.

No within-class point belief is selected.

## Exact finite formula

Let `pi(o)` be a deterministic terminal policy, `beta(o)` a benchmark policy,
`K[i,o] = P(o | h_i, probe)`, and `L[i,o,a]` the registered terminal loss.  For
fixed policies, the exact worst compatible expected loss gap is

```text
G(pi,beta)
  = sum_c lambda[c]
      max_{i in c, p[i] > 0}
        sum_o K[i,o] * (L[i,o,pi(o)] - L[i,o,beta(o)]).
```

The exact worst-case regret of `pi` is `max_beta G(pi,beta)`.  The active
certificate enumerates every deterministic `outcome -> action` policy and
returns the lowest-index minimax policy.  A zero value means that the contingent
policy is Bayes-optimal for every complete current belief compatible with the
registered quotient masses.  A separately declared tolerance gives a bounded
expected-regret certificate.

With one outcome, the construction reduces exactly to
`query_decision_certificate_v1`.

## Minimum-cost probe selection

`select_minimum_cost_decision_probe` evaluates a finite probe portfolio and
returns the cheapest probe whose optimal contingent policy satisfies the
registered regret tolerance.  Ties are resolved by smaller certified regret and
then declaration order.  When no probe passes, the selector returns no probe
index and requires caller-owned fallback.

This gives an executable version of the principle:

> Acquire only the information needed to identify the decision, not the entire
> latent state.

## Controlled mechanism

Run:

```bash
python -m experiments.active_decision_probe_v1.run_controlled \
  --check experiments/active_decision_probe_v1/controlled_result.json
```

The four-state example contains:

- no probe: state entropy 2 bits and minimax regret 1;
- a one-cost decision probe: residual state entropy 1 bit but minimax regret 0;
- a four-cost state probe: state entropy 0 and minimax regret 0.

The minimum-cost exact selector therefore chooses the decision probe.  It leaves
two latent states unresolved after either outcome while identifying the terminal
action exactly.

## Computational boundary

For `O` outcomes and `A` terminal actions, there are `A**O` deterministic
policies.  The current exact implementation forms policy-pair regret and is
intended for small registered outcome alphabets.  It fails closed when the
declared policy-count cap would be exceeded.

## Claim boundary

The certificate is conditional on the supplied finite hypothesis support,
quotient masses, probe likelihood, outcome-conditioned loss table, action set,
and tolerance.  It does not establish that a probe is physically realizable or
safe, that the support contains future dynamics, that the likelihood is
calibrated, that source losses transport to a target, or that continuous robot
control is certified.

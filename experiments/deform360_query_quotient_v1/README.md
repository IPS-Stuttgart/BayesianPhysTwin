# Deform360 source-only query-quotient pilot v1

## Scientific purpose

This experiment is the first public real-data instantiation of the
query-quotient contract. It asks whether a causal response prefix improves the
probability assigned to a registered future-motion class while keeping latent
physical specificity and decision ambiguity explicit.

The public object is `002-rope-silk`. Only the six already-open source episodes
`0, 2, 5, 6, 7, 9` are used. Episodes `1, 3, 4, 8` remain forbidden and are not
opened by the workflow.

## Hypothesis and query

The finite hypothesis family is a 71-point grid over the velocity persistence
coefficient

```text
rho in [-0.5, 1.25].
```

For the registered six-frame horizon, each hypothesis implies the displacement
factor

```text
F_6(rho) = sum_{k=0}^{5} rho^k.
```

The outcome-independent quotient has three classes:

- `strongly_damped`: `F_6(rho) < 2`;
- `moderate_continuation`: `2 <= F_6(rho) < 4`;
- `persistent_continuation`: `F_6(rho) >= 4`.

The real held-out query label is formed from the least-squares scalar mapping the
current persistent-point velocity to the observed future six-frame displacement.
The held future is used only for Brier/logarithmic scoring, never for the prefix
posterior or class definition.

## Belief comparison

For each leave-one-source-episode-out prefix reset, the experiment records:

- the prior and posterior quotient probabilities;
- multiclass Brier and logarithmic scores against the real future query;
- the Jeffrey / forward-KL `I`-projection lift;
- the full prefix posterior;
- uniform, prior-MAP, and reverse-prior within-class lifts;
- total, quotient, and unsupported-specificity information;
- the exact ambiguity envelope for expected `rho`; and
- whether complete lifts sharing one quotient imply conflicting latent decisions.

The Jeffrey lift must have zero unsupported specificity by construction. A
positive public-data result would require improved held-out query scores; it
would not validate a unique physical cause or a complete latent posterior.

## Execution boundary

The workflow runs on:

```text
[self-hosted, Linux, X64, gpuserver4090]
```

and reads official `pcd_clean.tar` archives below:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360/
  processed-repository/processed/002-rope-silk/
```

Archives are streamed read-only. The dataset is not extracted into, rewritten,
or uploaded by this experiment. Only compact JSON, CSV, Markdown, and provenance
records leave the runner.

This is a same-object, source-only development pilot. It authorizes no fresh
confirmation, unseen-object, physical-contact, calibration, safety, or paper
claim.

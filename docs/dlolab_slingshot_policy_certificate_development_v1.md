# DLO-Lab Slingshot policy-gain certificate development v1

## Purpose

The frozen Slingshot certified-guard study exposed a useful failure mode. The
posterior predictive policy had substantial mean value on 288 fresh public
simulator worlds, but the simultaneous six-action regret guard retained only
1.38% of that value. Its calibration target was stronger than the deployed
decision required: it covered every alternative action before selecting one.

This development study asks whether a **fixed selected policy** can receive a
tighter certificate. It does not alter or reclassify the failed 288-world study.
Those worlds and their per-world outcomes are not inputs here.

## Method

Let `pi(z)` be the posterior-predictive action selected from prefix observation
`z`, and let `g(z)` be its realized reward gain over the byte-identical incumbent.
A source-only local predictor estimates `g` from the mean gain of the five
nearest opened development observations for the same selected action.

The neighborhood feature concatenates:

- three registered rod points relative to the sphere center at each prefix time;
- temporal increments of all four observed points.

Both components cancel the registered shared xyz sensor bias. Feature scaling is
fitted only on the reference rows. Reference rows are canonically ordered before
stable distance ranking.

For an independently fitted predictor `g_hat`, split-conformal calibration uses

```text
s_i = g_hat(z_i) - g_i
q   = rank ceil((n_cal + 1) * (1 - alpha)) of {s_i}
L(z) = g_hat(z) - q
```

The candidate action is admitted only when `L(z) >= -0.002`; otherwise the
incumbent action is selected exactly. For a fixed policy and predictor, every
harmful admitted action is then a one-sided coverage failure, so its marginal
probability is at most `alpha` under the usual split-conformal exchangeability
assumption. This is not conditional coverage and is not a physical-safety claim.

## Opened-data capacity diagnostic

The diagnostic uses the parent's 32 evaluation and 19 calibration worlds, all
of which were already opened and are now development data. Each query excludes
its own outcome from the five-neighbor predictor. The same 51 rows are then used
to describe the attainable one-sided offset, so this is explicitly a
retrospective leave-one-out capacity diagnostic, not prospective coverage
evidence.

At `alpha=0.10`, the policy-level diagnostic admits 6/51 updates, has zero harms
beyond the frozen 0.002 numerical margin, and obtains mean reward gain 0.005342
over all 51 worlds. The unguarded posterior policy gains 0.011496 but harms 18
worlds. The older prospective simultaneous-action guard gained 0.000220 on its
separate 288-world panel.

The result is enough to justify a new prospective test, not enough to claim that
the new certificate transfers.

## Prospective boundary

A claim-bearing successor must freeze the feature, five-neighbor predictor,
posterior policy, `alpha=0.10`, 0.002 harm margin, and exact incumbent fallback
before generating a disjoint calibration roster. The local reference set,
calibration set, and final certification worlds must be mutually disjoint. The
calibration outcomes may set only the registered conformal order statistic.
Final decisions must be sealed before final all-action futures are opened.

No physical data, protected target, held-v8, DLO4, or DLO5 material is used.
This is a public-simulator method-development result, not an official benchmark,
robot-safety, real-world, or SOTA claim.

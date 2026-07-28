# Action-response admission

`ActionResponseAdmissionV1` is a provider-neutral, prefix-only certificate for
one question:

> Did the observed deformable object respond to the measured action in a
> spatial direction supported by the causal physical rollout?

This check sits between perception and Bayesian state updating. It is needed
because multiview consistency alone cannot distinguish a true object
displacement from a coherent camera bias, while an almost-static window can
make exact persistence substantially better than any nonzero update.

## Inputs and separation of roles

The certificate consumes:

- a PhysTwin rollout ending at the causal prefix boundary;
- associated material-point observations from one or more sensors;
- metric observation covariance;
- residual-independent prior reliability;
- association probability as a separate support quantity;
- a measured actuator trajectory;
- physical action support over graph nodes; and
- explicit sensor and material-point correlation groups.

It does not consume a candidate state update, future observations, hidden
identities, or future loss. The physical innovation therefore cannot lower
prior perception reliability, and it is not counted twice.

## Shared-bias treatment

With declared low-motion reference nodes, the method estimates a robust
per-frame shared translation from their observation-minus-physics residual.
Without such references, it projects both physical and observed response into
the translation-invariant spatial subspace.

The second mode intentionally cannot certify pure global translation. Under

```text
y = d + b + e,
```

the worlds `(d = u, b = 0)` and `(d = 0, b = u)` are camera-observationally
indistinguishable. Exact fallback is the correct result without an independent
anchor.

## Correlation handling

Rows with the same material cluster are collapsed with covariance
intersection, so repeated pixels or time samples do not accumulate independent
precision. Sensors with the same group ID are treated as unknown-correlated
duplicates: the weakest evidence is retained and the independent-group count
does not increase.

The default certificate requires three independent groups, action-aligned gain
and direction, enough distinct material clusters, and agreement across groups.
All numerical thresholds are source-frozen protocol choices, not target-tuned
constants.

## Deployment boundary

Admission is necessary but not sufficient. A complete deployment remains:

```text
source-causal observation belief
        +
causal physical response and measured action
        |
        v
ActionResponseAdmissionV1
        |
        v
bias-aware candidate inference
        |
        v
nonlinear PhysTwin closure
        |
        v
source-calibrated regret upper bound
        |
        v
complete candidate belief or exact baseline belief
```

Rejected admission must route the unchanged baseline belief object. A passing
certificate only says that a nonzero update is causally supported; it does not
claim that the update will beat the baseline. That remains the job of the
source-calibrated baseline-relative regret guard.

## Evaluation status

The implementation adds the missing reusable certificate and synthetic
controls. It does not alter or authorize any existing prospective or held
evaluation. Before a fresh-object run, the thresholds and downstream regret
model must transfer across already-open source objects with disjoint future
identities and no major worst-case regression.

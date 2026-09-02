# Adaptive stochastic policy-tree controlled result

## Result

The registered 16-hypothesis mechanism separates task-relevant adaptive sensing
from full-state information gathering.

| Registered policy class | Exact worst-case regret | Operational output |
| --- | ---: | --- |
| Direct terminal actions | 0.450 | exact fallback |
| At most one stochastic probe | 0.450 | exact fallback |
| Fixed nonadaptive two-probe class | 0.450 | exact fallback |
| Adaptive depth-two tree | **0.129** | **sense** |

The adaptive tree first measures `route`. It then measures `x` after route
outcome zero and `y` after route outcome one. It reduces worst-case regret by
**71.33%** relative to fallback.

The `nuisance` sensor is the cheapest and most accurate sensor and has the
largest mutual information about the complete state. It is never selected by
the decision-regret policy because it cannot change the registered action.

Every registered probe has full-support noise: every outcome has positive
likelihood under every physical hypothesis. Consequently, all 16 physical
hypotheses remain possible after every realized policy path. The finite action
is identified while the complete state remains unidentified.

## Reproduction

```bash
python -m experiments.adaptive_stochastic_policy_tree_v1.run \
  --output /tmp/adaptive-stochastic-policy-tree-v1.json
```

The generator writes a content-addressed JSON result and refuses to overwrite an
existing file. The focused controlled-evidence test regenerates the result and
checks the strict separation.

## Interpretation

The result establishes a method-level strictness example:

```text
direct decision unidentified
< one-step sensing insufficient
< fixed two-step sensing insufficient
< adaptive two-step decision identified
< complete state still unidentified.
```

It does not establish real probe validity, target-domain calibration, online
robot performance, or deployment safety. The next empirical stage must freeze a
complete virtual- or physical-probe tree roster before calibration outcomes and
must evaluate all retained trees on independent complete trajectories.

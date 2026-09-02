# Controlled active decision-identifying probe

This experiment is the executable mechanism check for
`bayesian_phystwin.active_decision_probe_v1`.

Four equally likely latent states occupy one unresolved quotient class.  States
0--1 require terminal action 0 and states 2--3 require terminal action 1.

The registered probe portfolio is:

| Probe | Cost | Outcomes | State information | Decision result |
| --- | ---: | ---: | --- | --- |
| `no_probe` | 0 | 1 | none | worst-case regret 1 |
| `decision_probe` | 1 | 2 | leaves two states per outcome | regret 0 |
| `state_probe` | 4 | 4 | identifies the state | regret 0 |

The exact minimum-cost selector chooses `decision_probe`.  This is the strict
separation needed by active decision-identifiable physical twins: the cheapest
probe identifies the action while deliberately leaving the complete state
unidentified.

The committed `controlled_result.json` is deterministic and checked in CI.
This is mechanism evidence only.  It does not validate a real physical probe or
a source-to-target likelihood model.

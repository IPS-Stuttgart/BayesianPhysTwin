# Causal4D Repository Migration

Causal4D is maintained at
[FlorianPfaff/Causal4D](https://github.com/FlorianPfaff/Causal4D).

The implementation, public-data adapters, tests, protocol configurations,
remote runners, and frozen Causal4D evidence were extracted with their Git
history. Bayesian-PhysTwin now owns the physical belief, graph, observation,
and PhysTwin/Warp provider code. Causal4D consumes those versioned artifacts
and owns intervention abduction and held-out interventional prediction.

Existing `causal4d-*` commands are installed by the Causal4D package. The
following integration diagnostics also moved:

| Previous command | New command |
| --- | --- |
| `bpt-structural-protocol` | `causal4d-structural-protocol` |
| `bpt-diagnose-phystwin-discrepancy-location` | `causal4d-diagnose-phystwin-discrepancy-location` |
| `bpt-aggregate-phystwin-discrepancy-location` | `causal4d-aggregate-phystwin-discrepancy-location` |
| `bpt-diagnose-phystwin-propagated-state` | `causal4d-diagnose-phystwin-propagated-state` |
| `bpt-aggregate-phystwin-propagated-state` | `causal4d-aggregate-phystwin-propagated-state` |

Historical Bayesian-PhysTwin tags remain immutable. In particular,
`v0.3.0-causal4d-aip` still identifies the original monorepo milestone, while
new Causal4D development and releases use the standalone repository.


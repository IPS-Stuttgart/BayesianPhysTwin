# Transport4D public development evidence

Decision: **public-development-tier-separation-established**

| Shift | Tested tier | Supported | Relative change | Wins |
|---|---|---:|---:|---:|
| `same-object-cross-backend` | `exact_coefficients` | true | 2.985% | 8/8 |
| `same-object-cross-backend` | `low_dimensional_correction` | true | 2.250% | 6/8 |
| `cross-object-operator` | `exact_coefficients` | false | -16.590% | 0/28 |
| `cross-object-operator` | `procedure_only` | true | 6.803% | 28/28 |

The same DLO3 correction transfers unchanged across physical backends
for the same object, but fails on every DLO4/DLO5 target trajectory.
Refitting the registered procedure on each target object succeeds on all
28 target trajectories. The transferable object is therefore hierarchical,
not a single coefficient vector.

These outcomes were already open before Transport4D was designed. They are
method-development evidence, not a fresh confirmation of the tier selector.

The matrix establishes a public real-data motivation and strict positive/negative tier separation only. Every numerical outcome in this protocol was available before Transport4D was designed. It does not validate the tier selector on an untouched target, establish a harmful-transfer probability, prove cross-object transport, or authorize a paper, deployment, or safety claim.

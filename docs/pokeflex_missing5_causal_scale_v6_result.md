# PokeFlex missing-five causal scale V6

## Scope

V6 is an exploratory, target-disjoint refinement of the frozen V5 PokeFlex
candidate. It uses 30 already-open public real-world actions to decide, at each
future frame, whether the Cylinder or Heart correction scale may increase. The
five unavailable official target archives and their mesh outcomes were not
used. No held-v8 artifact was accessed.

The selector uses only information available before the predicted frame:
normalized episode phase, update magnitude, prior-motion magnitude, their
ratio, and their cosine. A robust source transform and a group-conservative
nearest-neighbor model estimate the lower gain across source takes. Admission
requires both source support and a lower-envelope gain above `0.001` mm.

| Object | V5 scale | Admitted V6 scale | V6 fallback |
| --- | ---: | ---: | --- |
| `3dPrintedCylinder` | `0.25` | `0.375` | exact V5 |
| `3dPrintedHeart` | `0.1875` | `0.25` | exact V5 |
| `3dPrintedPizza` | `0.125` | none | exact V5 |
| `Pillow` | `0.125` | none | exact V5 |
| `Sponge` | `0.125` | none | exact V5 |

Unsupported observation updates still return the released checkpoint exactly.
Malformed features, out-of-distribution features, insufficient source gain,
and all unpromoted objects return V5 exactly. The state innovation is not used
as a perception-reliability feature, and no target outcome can enter the
decision.

## Source result

| Audit | Result |
| --- | ---: |
| Leave-one-take-out | 12/12 wins, 0 regressions |
| Mean leave-one-out gain | 0.4245% |
| Minimum leave-one-out gain | 0.0528% |
| Leave-two-take-out | 60/60 wins, 0 regressions |
| Sensitivity bank | 36/36 configurations without regression |
| Permutation controls | 0/1000 matched both observed mean and minimum |
| Synthetic positive controls | 12/12 passed |
| Harmful synthetic-region admissions | 0 |

The compact leave-one-take-out development comparison is:

| Object | V5 mean CD | V6 selected mean CD | Mean change |
| --- | ---: | ---: | ---: |
| `3dPrintedCylinder` | 3.6826 mm | 3.6570 mm | -0.0256 mm (-0.6931%) |
| `3dPrintedHeart` | 4.1835 mm | 4.1769 mm | -0.0066 mm (-0.1559%) |
| Equal-take aggregate | 3.9331 mm | 3.9170 mm | -0.0161 mm (-0.4088%) |

Cylinder contributes the clearer effect: 0.6931% mean and 0.3444% minimum
leave-one-out gain. Heart contributes 0.1559% mean and 0.0528% minimum gain;
its smallest absolute gain is only 0.00165 mm. The Heart branch should
therefore be regarded as plausible but fragile until independent target data
exist.

The source gate passes, so V6 is justified as a frozen candidate for the exact
five archives. It is not yet justified as a larger preregistered study or a
state-of-the-art claim. The next empirical action remains a single sealed
execution on those exact archives when they become available.

## Evidence

- Model canonical SHA-256: `4b6835f6ab57787be007855141081a3a10cea30eba736d47863b68fa7acf6ffa`
- Model file SHA-256: `130fd65b84723b05a8a7b6926b2c82bea2f356d3ff35c613fb7d6bf4f51373fa`
- Source result canonical SHA-256: `e7f17d5bd9045a3634e4f07b32ae9217ea8432ddd24ed387ab1c3512dd18483f`
- Source result file SHA-256: `733dd4376ea7fcb62d67528479cf70c6d040f71b3ddf29e2dfb6e00950d645d1`
- Source artifacts: 30/30 matched the V5 inventory byte for byte
- Independent regeneration: byte-identical model and source result
- Official target outcomes used: false
- Held-v8 accessed: false

The public source observations are physical measurements from real PokeFlex
executions. Human approval is not a scientific prerequisite for this public
dataset analysis; the remaining dependency is access to the exact five
author-provided archives and enforcement of the registered prediction barrier.

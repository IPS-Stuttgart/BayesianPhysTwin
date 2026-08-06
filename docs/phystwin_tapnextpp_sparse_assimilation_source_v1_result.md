# TAPNext++ Sparse Assimilation Source Result

Date: 2026-08-06

Status: source gate failed; stop before independent evaluation.

## Question

The frozen TAPNext++ RGB-D provider had already passed its separate eight-case
prefix-transfer gate. This study asks whether those automatic sparse metric
observations improve Bayesian-PhysTwin's untouched future beyond the existing
dense graph-persistence correction.

Four sealed arms were evaluated on eight already-open source cases:

1. the physical rollout;
2. dense graph persistence;
3. dense persistence plus a sparse update at geometry-associated graph nodes;
4. dense persistence plus the graph-smoothed sparse update, the primary arm.

The study is an opened-source transfer test. It is not independent
confirmation or a fair open-loop state-of-the-art comparison.

## Aggregate Result

Values are equal-case means. Errors are in millimetres; lower is better.

| Arm | CD | All track | Observed track | Hidden track | Late track | 90% coverage | NEES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Physical | 11.431 | 20.664 | 21.835 | 20.008 | 22.837 | 31.10% | 24.116 |
| Dense persistence | **8.004** | **16.094** | **16.175** | **15.915** | **19.591** | 98.12% | 1.167 |
| TAPNext++ direct | **8.003** | 16.159 | 16.320 | **15.915** | 19.668 | 98.50% | 1.086 |
| TAPNext++ graph | 9.278 | 17.571 | 17.220 | 17.783 | 20.774 | 98.65% | 0.659 |

Relative to dense persistence, the primary graph arm regresses:

- Chamfer distance by 15.91%;
- all-identity track error by 9.17%;
- queried-identity track error by 6.46%;
- disjoint hidden-identity track error by 11.74%; and
- late track error by 6.04%.

The direct-node arm is essentially neutral. It changes CD from 8.004 to
8.003 mm while increasing all-track error from 16.094 to 16.159 mm.

## Frozen Gate

| Criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| CD gain over dense persistence | at least 5% | -15.91% | no |
| All-track gain | at least 5% | -9.17% | no |
| Observed-track gain | at least 10% | -6.46% | no |
| Hidden-track regression | at most 2% | 11.74% | no |
| Joint CD and all-track nonregressions | at least 6/8 | 2/8 | no |
| Hidden future support | at least 95% | 98.08% | yes |
| Conditional 90% coverage | at least 80% | 98.65% | yes |
| Coverage no farther from 90% than dense | required | farther | no |
| Failed-provider exact fallback | required | exact | yes |

The registered decision is `stop-before-independent-evaluation`.

## Case Results

Each pair reports dense persistence followed by the primary graph arm.

| Case | CD | All track | Observed track | Hidden track |
| --- | ---: | ---: | ---: | ---: |
| `double_lift_cloth_1` | 8.919 -> 8.832 | 17.223 -> 17.185 | 18.415 -> 19.408 | 16.270 -> 15.413 |
| `double_lift_zebra` | 12.292 -> 14.722 | 23.476 -> 27.690 | 23.466 -> 26.613 | 23.381 -> 28.694 |
| `double_stretch_sloth` | 6.649 -> 7.925 | 15.473 -> 16.269 | 20.354 -> 17.376 | 11.475 -> 15.373 |
| `rope_double_hand` | 5.049 -> 6.556 | 15.208 -> 16.539 | 12.976 -> 14.898 | 16.887 -> 17.769 |
| `single_lift_dinosor` | 15.186 -> 17.070 | 21.039 -> 23.326 | 22.689 -> 21.593 | 19.391 -> 23.992 |
| `single_lift_rope` | 6.514 -> 8.144 | 14.187 -> 13.757 | 9.836 -> 11.325 | 17.668 -> 15.703 |
| `single_push_rope_4` | 3.404 -> 3.404 | 8.288 -> 8.288 | 7.411 -> 7.411 | 9.001 -> 9.001 |
| `weird_package` | 6.022 -> 7.570 | 13.862 -> 17.515 | 14.251 -> 19.138 | 13.244 -> 16.325 |

`single_push_rope_4` failed the previously frozen provider gate and therefore
used the required bit-exact dense fallback. It remains in every aggregate.

## Post-Open Association Diagnosis

The tracker observes benchmark material queries accurately during the prefix,
but the assimilation arm associates each query to nearby graph geometry at the
deformed source frame. The benchmark instead keeps a frame-zero nearest-node
material identity fixed through time.

A separately checksummed post-open audit compared these two node identities.
Only 2 of 27 scored provider identities, 7.41%, had the same source-frame MAP
node and benchmark frame-zero material node. One additional provider identity
had no finite frame-zero manual identity, and the failed-provider case had no
sparse association.

This explains the arm pattern without claiming that association is the only
error source:

- the direct update usually changes nodes that the manual-track metric does
  not follow, so it is nearly neutral;
- graph smoothing reaches a wider region, but spreads corrections originating
  from mostly mismatched material attachments and harms CD and hidden tracks;
- the mismatch is not only a long-prefix staleness effect: it also occurs in
  zero-gap cases.

The result therefore rejects source-frame nearest-geometry attachment as the
bridge from accurate sparse tracking to persistent physical-state correction.
It does not reject TAPNext++ as an observation provider.

## Next Method Boundary

The next credible method is **frame-zero material attachment transport**:
associate each query to the physical graph at its immutable frame-zero
material anchor, then transport only its causal observed displacement and
covariance through the permitted prefix. Association must not be recomputed
from nearby deformed geometry at the endpoint.

That method must receive a new source protocol and must not be tuned on these
opened eight outcomes. This failed arm does not authorize an independent run,
and no held-v8 artifact may be inspected while its independently owned gate is
pending.

## Technical Amendments

Two staging attempts stopped before any case prediction or future outcome was
opened:

1. commit `7986e8ce` corrected the registered physical-parameter path;
2. commit `4c495f66` bound tracker source-frame provenance from the frozen
   tracker protocol rather than inferring it implicitly.

Neither amendment changed a method parameter or used future source evidence.

## Provenance

- protocol SHA-256:
  `303654f662fb0852b2c02dcfe5d7235992a252c3bc3b8798c8e05c80e216b0c0`
- source manifest canonical SHA-256:
  `881e725d7728bf54e4323240f7bf1827415a36e82f59820ab16dd333b4114dac`
- source manifest file SHA-256:
  `db27105334854a8ffa40d7b370f539f53e1dbe740fcd6216435ef62cc09bccb2`
- prediction manifest canonical SHA-256:
  `6ceadb79e43b5057f375dfd575e51399447614e36aabd39eb9ef465e8fdc0ced`
- prediction manifest file SHA-256:
  `c15a9ebaaab8c5161ba3f24d1723ed42eef9d4617a1f34d06749310ea52610f5`
- source summary canonical SHA-256:
  `dc5361e047438050c412254fa288698b41a6272e25617603c5e4a9243ba0b022`
- source summary file SHA-256:
  `d467d49c27e85865b74abc411054c3d2bf35d952d60ae784eab24354011764bd`
- post-open association audit canonical SHA-256:
  `d9845949e72f33d13bdb1b7c7b74ffb779307f02e0792d143ba3c7226ae801d5`
- post-open association audit file SHA-256:
  `61b9b6687d62477e919f7126699d5cd9fa4054067df5533bfc366039befa862e`
- prediction implementation commit: `4c495f66`
- evaluator null-serialization fix: `f6b80c22`
- no held-v8 runtime, target, query, score, barrier, or outcome artifact was
  read or modified.

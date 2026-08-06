# TAPNext++ Material Transport Assimilation Source Result

Date: 2026-08-07

## Decision

The frozen 14-case source study produced a small, broad improvement over dense
persistence but failed its preregistered advancement gate. The decision is
`stop-before-independent-evaluation`.

| Future metric | Dense persistence | Material graph | Relative change |
| --- | ---: | ---: | ---: |
| Chamfer distance | 9.265 mm | **9.124 mm** | **-1.52%** |
| all-identity track error | 21.215 mm | **20.934 mm** | **-1.33%** |
| queried-identity track error | 21.368 mm | **21.243 mm** | **-0.58%** |
| disjoint hidden-identity track error | 20.211 mm | **19.750 mm** | **-2.28%** |
| late track error | 25.181 mm | **25.100 mm** | **-0.32%** |
| conditional 90% coverage | 94.78% | 95.78% | farther from 90% |

This is a concrete repair of the earlier source-geometry assimilation arm,
which regressed 15.91% in CD and 9.17% in all-track error. Fixing material
identity changed a large negative into a modest positive. It did not produce
the effect size or consistency required for a fresh evaluation.

## Frozen Gate

| Criterion | Required | Result | Pass |
| --- | ---: | ---: | :---: |
| CD gain | at least 5% | 1.52% | no |
| all-track gain | at least 5% | 1.33% | no |
| queried-track gain | at least 10% | 0.58% | no |
| hidden-track regression | at most 2% | 2.28% improvement | yes |
| joint CD/track nonregression | at least 10/14 | 8/14 | no |
| hidden future-frame support | at least 95% | 100% | yes |
| conditional coverage | at least 80% | 95.78% | yes |
| distance from nominal 90% | no worse than dense | 5.78 vs 4.78 points | no |
| failed-update fallback | exact dense output | 4/4 exact | yes |

Ten fixed-material updates were admitted. Four cases exceeded the frozen
30 mm query-frame distance from their immutable material node and used exact
dense fallback. Of the ten admitted updates, four jointly avoided CD and
track regression. The direct-node arm was nearly neutral; graph propagation
created the hidden-identity gain but also the case-level regressions.

## Per-Case Graph Change

| Case | Update | CD | Track | Hidden track | Joint |
| --- | :---: | ---: | ---: | ---: | :---: |
| `double_lift_cloth_3` | yes | -1.20% | -5.56% | -6.54% | yes |
| `double_lift_sloth` | yes | -2.48% | -6.17% | -1.57% | yes |
| `double_stretch_zebra` | yes | +0.80% | +3.14% | +9.37% | no |
| `single_clift_cloth_1` | yes | +3.41% | +1.26% | -0.29% | no |
| `single_clift_cloth_3` | yes | +4.54% | +1.59% | -0.36% | no |
| `single_lift_cloth` | fallback | 0.00% | 0.00% | 0.00% | yes |
| `single_lift_cloth_1` | fallback | 0.00% | 0.00% | 0.00% | yes |
| `single_lift_cloth_3` | yes | +5.89% | -5.12% | -10.12% | no |
| `single_lift_cloth_4` | fallback | 0.00% | 0.00% | 0.00% | yes |
| `single_lift_sloth` | yes | -7.21% | -5.69% | -8.31% | yes |
| `single_lift_zebra` | yes | -4.19% | +0.43% | -3.61% | no |
| `single_push_rope` | yes | -9.54% | -6.45% | -5.19% | yes |
| `single_push_rope_1` | yes | -2.31% | +1.74% | -8.35% | no |
| `single_push_sloth` | fallback | 0.00% | 0.00% | 0.00% | yes |

Positive percentages in this table denote regression; negative percentages
denote error reduction.

## Interpretation

The provider itself was not the bottleneck: it passed on all 14 cases with
5.006 mm prefix identity RMSE and 99.19% support. The fixed material bridge
also removed the earlier identity-association failure. What remains is
temporal and spatial transfer: an accurate terminal-prefix relative
displacement is not reliably a persistent future graph field.

The result therefore supports three narrower conclusions:

1. immutable material identity is necessary and materially safer than
   source-frame geometry association;
2. conservative distance rejection gives exact non-worsening fallback; and
3. unconditional graph persistence is still too blunt for a state-of-the-art
   claim, even when its prefix observations are accurate.

These 14 futures are now exhausted for method selection. No cap, graph prior,
distance threshold, temporal rule, or arm may be retuned against them.

## Provenance

- method/protocol commit: `2439c05c`
- source-lock commit: `d1ec0f58`
- prediction-seal commit: `060176fe`
- protocol SHA-256:
  `ed6467b2dfe4eb8373b5ebe3fa49e32495f5704b26f0681eb22bd83319ffedbb`
- source-manifest canonical SHA-256:
  `7a068cb14b99dcd82b0a90b519b29a26c89247d4399df59746f01105910588a6`
- prediction-manifest canonical SHA-256:
  `124da54a2eeb54b772ca730ebfc9be9d49d19e042f8678c1c9a7397b7db62dbb`
- result-summary file SHA-256:
  `d16104dd43a9ebfa015b027f36dcb6d05a006ebc1772508f511ef1968aa83ba7`
- result-summary canonical SHA-256:
  `d8f42f19bef80ec00efc97d7bdda3517f5b512c504f73c589e76a550fca08315`

All 14 predictions sealed before any source future was opened. No case was
replaced. No independent target or held-v8 runtime, target, query, score,
barrier, or outcome artifact was accessed.

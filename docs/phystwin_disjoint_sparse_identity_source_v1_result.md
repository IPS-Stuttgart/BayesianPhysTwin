# Disjoint Sparse-Identity Source Result

Date: 2026-07-26

Status: source gate failed; stop the disjoint sparse-identity route.

## Question

The earlier opened-cohort ceiling reached 7.892 mm Chamfer distance and
13.429 mm manual-track error by observing the same manual identity family that
was later scored. This audit tests a stricter interface: observe 1, 2, or 4
deterministically selected material identities during the released prefix, then
score future track error only on identities that were never assimilated.

This is post-open mechanism evidence on the released PhysTwin-22 cohort. It is
not confirmation, a deployable observation method, or an open-loop
state-of-the-art result.

## Result

The analyzer reads only the preregistered
`causal_selected_dense_relative_cap_temporal` candidate. Values are equal-case
future means. Lower errors are better.

| Observed prefix identities | Baseline CD | Candidate CD | CD gain | Baseline hidden track | Candidate hidden track | Track gain | Joint wins | Trackless future frames |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11.389 mm | 7.898 mm | 30.65% | 21.774 mm | 20.037 mm | 7.98% | 15/22 | 3 |
| 2 | 11.389 mm | 7.927 mm | 30.40% | 21.367 mm | 20.022 mm | 6.29% | 12/22 | 4 |
| **4 (primary)** | **11.389 mm** | **7.927 mm** | **30.40%** | **22.703 mm** | **21.415 mm** | **5.67%** | **13/22** | **4** |

The hidden-track baseline changes with the sensor budget because each arm
scores a different, disjoint complement of identities. The Chamfer baseline is
unchanged.

## Frozen Gate

| Primary four-identity criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Relative CD improvement | at least 5% | 30.40% | yes |
| Relative hidden-track improvement | at least 5% | 5.67% | yes |
| Joint case wins | at least 16/22 | 13/22 | no |
| Minimum hidden future-frame support | 100% | 84.62% | no |
| Trackless future frames | 0 | 4 | no |

The four unsupported frames all occur in `single_lift_dinosor`. Removing that
support failure would not rescue the method: the independent 16/22 joint-win
gate still fails.

## Interpretation

Sparse prefix identity evidence transfers weakly to unseen identities, but it
does not reproduce the earlier same-identity ceiling. The dense released
pseudo-track channel still supplies a large Chamfer gain, whereas the
four-identity update reduces hidden-track error by only 1.288 mm and remains
far above the published 15 mm context value.

Increasing the observed budget from one to four identities does not improve
the hidden transfer monotonically. The hidden scoring set becomes smaller and
the joint-win count falls from 15 to 13. This argues against treating a few
accurate material anchors as a sufficient solution to the identity problem.

The result supports the narrower conclusion that the next observation method
must infer transferable material correspondence and shared bias, not merely
provide a handful of accurate prefix anchors. Per the frozen decision rule, do
not tune identity counts, caps, graph settings, temporal gains, or support
masks against these opened hidden outcomes. The registered noise/dropout stage
is not authorized.

## Technical Amendments

Two runs aborted before writing any report or hidden score:

1. The first exposed a solver regression: the current branch lacked the
   previously tested, residual-verified sparse-direct fallback used by the
   parent opened-22 evaluator. Commit `d1853bd` restored that exact fix.
2. The second exposed a zero denominator when a fixed sparse identity had no
   valid prefix-validation observation. Commit `a668a5d` preregistered a
   support rule that renormalizes selection to prefix Chamfer in that case.

Neither amendment changes the fixed primary candidate, reads a hidden outcome,
or allows a non-primary candidate to affect the decision.

## Provenance

- evaluator commit:
  `a668a5ddce80a5681a205f98f101f40cbd7d0a0a`
- protocol SHA-256:
  `65e44d59e2493dc859163cd0bc9555cbd5645ad7bb6e980893fd90e3d9f122d8`
- compact summary SHA-256:
  `b570d073607220cb159db137c7a6fb9adb4337d0fdf3eb1c3ed7ef47c67f93e7`
- one-identity raw report SHA-256:
  `149ee446c90cea4e84f362aa5e46608618d016569b8d5c658dfff1ae1f97cfa1`
- two-identity raw report SHA-256:
  `d26361d695c80bc82f0269d48495250d767dffb0bbe0f869be147ea89e076f83`
- four-identity raw report SHA-256:
  `45bfd665a736c35ec375ebc714fe0cab3ed34dde28af32b892df769e0dff7bf0`
- runtime: Python 3.12.3, NumPy 2.0.2, SciPy 1.17.1
- archived evidence:
  `results/sota/diagnostics/phystwin_disjoint_sparse_identity_source_v1/`
- no held-v8 artifact or process was inspected or modified.

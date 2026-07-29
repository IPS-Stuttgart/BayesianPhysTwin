# Deform360 Causal-Response V12 Query-Feasibility Result

## Decision

**The frozen source gate fails at 2/8 admitted cases and V12 is closed.**
The registered gate required at least six admissions and no technical source
failure.

| Target-free disposition | Count |
| --- | ---: |
| Complete 16-query schedule | 2 / 8 |
| Geometric abstention | 2 / 8 |
| Exact-panel source failure | 4 / 8 |
| Required admissions | 6 / 8 |

No tactile stream, tracker output, state update, future identity, future object
observation, future metric, V1 target, or held-v8 artifact or process was read.

## Frozen Evidence

- Parent V12 method commit:
  `d5eab1b1dcf8bb77cd7a37f9716f5846559e930c`
- Pre-disposition protocol commit:
  `995b769a4b4f8c7b4dcf447196bfd0b38ab2d417`
- Protocol canonical SHA-256:
  `631f8df95befdf7095c6acd791a2285dfe205480a1eb70ce728f8449dd534b5d`
- Result canonical SHA-256:
  `03ae0a299f24a84fcc8e4ae5d808d9bbc935e4f14d9df30017f1bf463c990bfa`
- Result file SHA-256:
  `603cc67208a0da4d12b5072d5f46102bbf279019e4ce91efcd2d18ae2710b60c`

The complete successful and abstained query artifacts are under
`results/sota/diagnostics/deform360_causal_response_query_feasibility_v12_source/`.
Missing per-case reports are retained as technical failures by the frozen
cohort evaluator and are not replaced.

## What Failed

Two cases had zero graph identities satisfying all three simultaneous
requirements:

1. physical action support of at least 0.1;
2. association probability of at least 0.5 in three proposal cameras; and
3. the same support in three disjoint validation cameras.

Four additional cases had only 8--10 complete streams in the exact registered
12-camera panel. The missing-stream counts were two, four, two, and three.
Consequently, the full-panel source contract rejected them before query
association.

The two admitted cases retained 64 and 291 eligible identities and both filled
the complete 16-query budget. The method is therefore executable when its
carrier exists, but the carrier is not broad enough for the intended source
distribution.

## Interpretation

V12 combined action support, independent camera panels, tactile gating, robust
innovation handling, and exact fallback. Its failure occurs earlier: the hard
camera and query-support contract prevents the observation channel from
materializing in most source cases.

Weakening V12 after seeing this result is not allowed. In particular, this
result does not authorize changing the query count, camera minimum,
association threshold, action-support threshold, or source gate.

A distinct follow-up may use a target-free adaptive panel with a preregistered
two-view fallback, covariance inflation, an explicit shared-bias nuisance, and
the same action/tactile support and exact fallback. Such a method must be
versioned and gated independently; this V12 cohort cannot confirm it.

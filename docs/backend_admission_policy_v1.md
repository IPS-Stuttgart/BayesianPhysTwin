# Evidence-first backend admission policy v1

**Status:** active from 2026-08-18  
**Scope:** material-trajectory backend families and their public recommendation
status

## Purpose

A backend profile can be useful for contract compatibility without providing
native physical evidence. This policy prevents adapter breadth from being
mistaken for scientific progress and concentrates qualification effort on the
backends already selected in
[issue #664](https://github.com/IPS-Stuttgart/BayesianPhysTwin/issues/664).

This policy governs the interchangeable material-backend registry. Specialized
benchmark pipelines are recorded separately. In particular, the confirmed
[DEFORM DLO2 result](deform_dlo2_local_residual_official_v7.md) does not appear
as a canonical material family and does not lift the registry admission freeze;
it establishes a narrow DLO2 result under its own frozen source and official
evaluation gates.

The existing `preferred`, `supported`, and `experimental` labels describe
implementation maturity only. They do not authorize a physical-performance,
calibration, transfer, or downstream-value claim.

## Evidence stages

| Stage | Meaning | Minimum retained evidence |
| --- | --- | --- |
| `registered-adapter` | The profile is discoverable and satisfies the portable contract, possibly through an analytic fallback. | Registry and contract tests; exact provenance. |
| `native-smoke-passed` | The native dependency executes a deterministic minimal case. | Installed-version record, deterministic replay, finite outputs, and explicit native-path proof. |
| `source-physics-qualified` | The native backend passes the frozen source-physics checks. | Zero-action drift, equivariance, time-step sensitivity, topology identity, parameter sensitivity, physical sanity, and exact fallback. |
| `source-value-qualified` | The backend improves a preregistered source endpoint without unacceptable regressions. | Proper predictive score, uncertainty width/calibration, harmful-group probability, incumbent parity, and exact fallback identity. |
| `fresh-object-qualified` | The frozen candidate passes an independently grouped unseen-object decision. | Sealed source decision, disjoint confirmation units, complete provenance, and retained negative outcomes. |
| `downstream-causal-qualified` | The qualified belief improves a registered Causal4D task. | Factual-prefix/intervention protocol, counterfactual endpoint, subgroup checks, and fallback accounting. |

Stages are monotone: a backend cannot skip a lower stage, and a failed or
terminal experiment remains part of its evidence record.

## Admission freeze

Until at least one currently registered external backend reaches
`source-value-qualified`:

1. no additional backend family is admitted to the main registry;
2. fixes that improve an existing adapter, qualification harness, provenance,
   deterministic fallback, or retained evidence remain admissible;
3. new families may be explored on draft branches, but must remain explicitly
   adapter-only and are not recommended to users; and
4. user-facing backend selection must not imply that implementation maturity is
   evidence maturity.

The freeze is lifted only by a retained qualification and source-value bundle,
not by optional-dependency import success, registry coverage, or an analytic
contract fallback.

## Current qualification focus

Issue #664 identifies Genesis MPM and JAX-FEM as the current candidates. The
next qualification should select one candidate and run one funnel:

1. native execution and provenance;
2. source-physics qualification;
3. failure-localization arms separating simulator, parameter, observation,
   query, covariance, and guard effects; and
4. a matched value test against persistence, the released PhysTwin/Warp
   baseline, `last_residual`, the unguarded candidate, the guarded candidate,
   and an oracle diagnostic.

Promotion requires a proper-score improvement, no unacceptable grouped
regression or width cost, and an exact caller-owned fallback whenever the guard
rejects the candidate.

## Relationship to releases

A release may ship registered adapters below `source-value-qualified` for
reproducibility and interface testing. Release notes and `bpt backend list`
output must identify their evidence stage and state that adapter compatibility
is not native physical evidence. Only qualified stages may be used to justify
backend recommendations or scientific claims.

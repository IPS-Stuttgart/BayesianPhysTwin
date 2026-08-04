# Prob4D cross-window material-identity development study

## Scientific question

Prob4D causal track IDs are persistent inside one decoded prediction window. They
do not currently claim that two IDs from different windows denote the same material
point. The controlled Prob4D-to-BayesianPhysTwin study therefore granted persistent
identity across the complete prefix. This development study isolates the missing
mechanism:

> Can source-only association across overlapping windows recover enough material
> identity to improve a guarded Bayesian physical-state query over an update that
> uses only the newest internally persistent window?

The study executes the merged `prob4d.cross_window_tracklets` algorithm rather than
an in-study surrogate.

## Methods

All visual methods use the same explicit joint cross-window gauge prior, robust
nominal/outlier likelihood, source-reliability precision scaling, provider-final
likelihood powers, state prior, and target-blind BayesianPhysTwin guard.

| ID | Observation identity semantics |
| --- | --- |
| `B0_physical_fallback` | No visual update |
| `B1_newest_frame_explicit_gauge` | Newest factor frame only |
| `P0_newest_window_persistent` | Both frames of the newest window; no cross-window identity |
| `P1_naive_local_id_cross_window_merge` | Treat window-local IDs as globally stable despite a hidden permutation |
| `P2_source_linked_cross_window_identity` | Merge only unambiguous mutual-best Prob4D links |
| `P3_oracle_cross_window_identity` | Evaluation-only true material links |

The primary development method is `P2`; the scientific reference is `P0`. The
physical fallback, naive merge, and oracle merge expose safety and mechanism bounds.

## Information order

The protocol contains three disjoint development partitions:

1. **Association configuration:** true synthetic material labels may select one
   source-only association configuration using the registered precision/recall/F1
   ordering.
2. **Pilot guard calibration:** physical query truth may calibrate only the common
   BayesianPhysTwin risk threshold for each method.
3. **Pilot evaluation:** reports association and downstream endpoints without
   changing the selected association rule or guard.

No confirmatory target seed is present in the development protocol. A target
protocol may be committed only after the registered development gates pass. It must
use new guard-calibration and target seeds, and its workflow must verify the frozen
protocol before execution.

## Synthetic identity challenge

Each physical group retains the six stress scenarios from the completed controlled
Prob4D-to-BayesianPhysTwin study. Two additional source windows overlap on two
absolute prefix frames. Their local points are represented in independently perturbed
`Sim(3)` gauges, and the second window receives a random local-ID permutation.
Scenario-specific source corruption adds incomplete support, common gauge error,
track outliers, or near-crossing ambiguity.

The newest-window reference is granted correct identity inside that window. Thus the
comparison does not reward `P2` for solving an artificial within-window problem. Its
only additional opportunity is conservative reuse of older observations whose
material correspondence was recovered from the overlap.

## Development decision

The registered development gates require:

- micro association precision and recall above their frozen minima;
- a positive source-linked improvement over newest-window persistence on the
  disjoint pilot evaluation;
- a paired object-level upper 95% bound below zero;
- a bounded harmful accepted-update rate; and
- exact physical fallback on every rejection.

Exit code `3` is a valid completed negative development result. It does not authorize
changing the opened development endpoints into a confirmatory claim.

## Workflow

`.github/workflows/prob4d-cross-window-identity-development.yml` runs on
`workstation2` with labels `[self-hosted, Linux, X64, nvidia-smi]`. It resolves the
exact Prob4D source revision, verifies the target-free protocol, runs focused tests,
executes every development group, independently checks the evidence hashes, and
uploads the complete result.

## Claim boundary

This is a development-only, calibration-separated synthetic mechanism study. A pass
can justify freezing one disjoint confirmatory protocol. It cannot establish target
performance, real Prob4D observation competence, physical-object benefit, deployment
calibration, or Causal4D intervention benefit.

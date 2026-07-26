# Experiment and evidence index

This page is the navigation layer for research workflows in Bayesian-PhysTwin.
The root README documents only the stable project identity, interfaces, and
onboarding path. Claim-bearing numbers belong in frozen evidence reports, and
current project status belongs in the canonical paper repository.

Legacy `bpt-*` entry points remain available for compatibility, but they are not
all stable API promises. Prefer the grouped `bpt` commands for reusable
workflows.

## Stable workflows

| Workflow | Stable command | Documentation |
| --- | --- | --- |
| Provider capability and provenance | `bpt provider manifest` | [Causal4D provider v1](causal4d_provider_v1.md) |
| Observation-belief validation | `bpt observation validate` | [ObservationBeliefV1](observation_belief_contract.md) |
| Robust residual replay | `bpt residual replay` | [Residual replay](residual_replay.md) |
| Controlled synthetic benchmark | `bpt benchmark synthetic` | [Synthetic benchmark](synthetic_benchmark.md) |
| Content-addressed run provenance | `bpt run manifest` | Run `bpt run manifest --help` for the current schema and operations. |

The observation-to-state boundary is described in
[gauge-aware observation update](gauge_aware_observation_update.md) and
[prior-aware guarded update](prior_aware_guarded_update.md). These documents
state the identifiability, covariance, causal-cutoff, and exact-fallback rules
that apply independently of any one experiment.

## Frozen and claim-bearing evidence

### Official PhysTwin full-22 evaluation

The [PhysTwin full-22 evidence report](phystwin_sota_22_v1.md) is the frozen
source for the released-cohort comparison, uncertainty intervals, render
reproduction, provenance, ownership boundary, and permitted claim language.
It supports improvement over released PhysTwin under the recorded protocol; it
does not support an overall state-of-the-art claim or calibrated raw posterior
covariance.

### Deform360 prospective validation

The [Deform360 bias-aware prospective v2 protocol](deform360_bias_aware_prospective_v2.md)
and its [frozen result](deform360_bias_aware_prospective_v2_result.md) record a
prospective calibration gate. The target-free support gate passed, the fresh
accuracy gate failed, and reserved targets remained unopened. This is a
negative prospective result, not a target evaluation.

## PhysTwin experiment families

| Area | Primary document | Status and purpose |
| --- | --- | --- |
| Released artifact and cue integration | [PhysTwin integration](phystwin_integration.md) | Pinned upstream contracts, residual export, cue sidecars, and likelihood boundary. |
| Hierarchical, graph, spatial, and discrepancy studies | [Advanced PhysTwin inference](phystwin_advanced_inference.md) | Experimental workflows, causal splits, controls, and interpretation limits. |
| Official Warp parameter refits | [PhysTwin refit](phystwin_refit.md) | Checkpoint restoration, matched baselines, provenance, and simulator requirements. |
| Causal MatPhys graph-part residual | [MatPhys graph-part residual](matphys_graph_parts_v1.md) | Corrected causal audit; source gate failed, so the family is frozen without a wider run. |
| Legacy MatPhys backbone experiments | [Causal MatPhys backbone](matphys_causal_backbone_v1.md) | Engineering history only; the original causal audit was invalidated by a frame-ordering defect. |
| Bias-aware guarded updates | [Bias-aware guarded belief](bias_aware_guarded_belief_v1.md) | Generic guarded-update method and source-only acceptance logic. |

Detailed commands live with the protocol they implement. New experiment
commands should be documented in their dedicated report rather than appended
to the root README.

## Cross-repository interfaces

- [Prob4D](https://github.com/FlorianPfaff/Prob4D) may emit the versioned,
  content-addressed `ObservationBeliefV1` artifact consumed here.
- [Causal4D](https://github.com/FlorianPfaff/Causal4D) owns Bayesian
  abduction-intervention-prediction and consumes the versioned provider and
  belief artifacts exported by Bayesian-PhysTwin.
- [Causal4D migration](causal4d_migration.md) records the historical command and
  tag boundary after Causal4D moved into its own repository.
- [BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
  is the canonical location for current scope, evidence status, figures, result
  artifacts, and paper-facing claims.

## Compute and repository hygiene

GPU host conventions and remote-run policy are documented in
[compute](compute.md). Large datasets, checkpoints, rendered videos, and raw
runs belong in ignored local paths such as `data/`, `checkpoints/`, `runs/`,
and `outputs/`, not in the source repository.

## Documentation placement policy

Use the following placement rule when adding work:

1. Put stable installation, architecture, and public interface information in
   the root README.
2. Put each experimental protocol, controls, status, and command sequence in a
   dedicated document under `docs/`.
3. Put frozen machine-readable evidence under `results/` or the corresponding
   evidence location named by the protocol.
4. Put current paper scope and claim status in `BayesianPhysTwin-Paper`.
5. Put Causal4D-specific methods and acquisition status in the Causal4D
   repository.

This separation keeps onboarding stable while preserving negative results and
complete experimental provenance.
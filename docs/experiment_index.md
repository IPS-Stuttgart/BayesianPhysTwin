# Experiment and evidence index

This page is the navigation layer for research workflows in Bayesian-PhysTwin.
The root README documents only the stable project identity, interfaces, and
onboarding path. Claim-bearing numbers belong in frozen evidence reports, and
current project status belongs in the canonical paper repository.

The package installs only `bpt`. Stable operations use direct grouped routes;
research workflows are classified as current experiments, diagnostics, or
archived reproduction paths in the command registry.

## Stable workflows

<!-- bpt-stable-commands:begin -->
| Command | Purpose | Documentation |
| --- | --- | --- |
| `bpt provider manifest` | Print the Causal4D provider capability manifest. | [Guide](causal4d_provider_v1.md) |
| `bpt observation validate` | Validate or summarize an ObservationBeliefV1 artifact. | [Guide](observation_belief_contract.md) |
| `bpt residual replay` | Replay exported residuals through the robust likelihood. | [Guide](residual_replay.md) |
| `bpt benchmark synthetic` | Run the controlled synthetic benchmark. | [Guide](synthetic_benchmark.md) |
| `bpt evidence summarize` | Summarize matched guarded prospective evidence. | [Guide](decisive_evidence_protocol.md) |
| `bpt evidence bundle` | Build or validate a content-addressed claim bundle. | [Guide](claim_bundle_v1.md) |
| `bpt run manifest` | Create or validate a content-addressed run manifest. | [Guide](reproducible_runs.md) |
<!-- bpt-stable-commands:end -->

The observation-to-state boundary is described in
[gauge-aware observation update](gauge_aware_observation_update.md) and
[prior-aware guarded update](prior_aware_guarded_update.md). These documents
state the identifiability, covariance, causal-cutoff, and exact-fallback rules
that apply independently of any one experiment.

## Command lifecycle

Use the registry rather than maintaining a static list here:

```bash
bpt experiment list
bpt diagnostic list
bpt archive list
bpt commands list --json
```

- `experiment` contains active research protocols.
- `diagnostic` contains audits and analyses that are not promotable methods by
  themselves.
- `archived` contains frozen historical and negative-result paths.
- `stable` contains reusable operational interfaces.

Each entry records an owner, optional dependency extras, and the removed
historical `bpt-*` name when one existed. Historical aliases are inspection and
migration metadata only; use `bpt commands migrate LEGACY_ALIAS` to obtain the
current grouped route.

## Frozen and claim-bearing evidence

### Official PhysTwin full-22 evaluation

The [PhysTwin full-22 evidence report](phystwin_sota_22_v1.md) is the frozen
source for the released-cohort comparison, uncertainty intervals, render
reproduction, provenance, ownership boundary, and permitted claim language.
It supports improvement over released PhysTwin under the recorded protocol; it
does not support an overall state-of-the-art claim or calibrated raw posterior
covariance.

The
[frozen full-22 reproduction capsule](../reproductions/full22_anchor_v1/README.md)
records the exact source revision, protocol and data identities, two-stage
source command, expected paper-facing metrics, fail-closed verification, and a
strict `RunManifestV2` bundle. The capsule has been reproduced independently on
`workstation2` and a fresh GitHub-hosted runtime and is the current portable
reproduction record.

### Deform360 prospective validation

The
[Deform360 bias-aware prospective v2 protocol](deform360_bias_aware_prospective_v2.md)
and its
[frozen result](deform360_bias_aware_prospective_v2_result.md) record a
prospective calibration gate. The target-free support gate passed, the fresh
accuracy gate failed, and reserved targets remained unopened. This is a negative
prospective result, not a target evaluation.

The
[official-Hub Deform360 visuotactile v1 protocol](deform360_official_hub_visuotactile_v1.md)
is the next independent-object gate. It replaces contaminated mounted-cache
trajectories with an exact official raw-data revision and adds an independent
contact anchor to the existing explicit-gauge solver.

The [visual-provider freeze](deform360_visual_provider_freeze.md) is the next
data-free execution step. It resolves the exact cached MotionCrafter model
revisions, Prob4D provider manifest and attestation, metric-frame policy, and
portable model-set identity before any of the ten selected calibration-object
payloads may be opened.

The
[prepared-source inventory](deform360_calibration_prepared_inventory.md)
binds the successful ten-object calibration-source run to the retained aligned
RGB, tactile, and robot bytes on the protected runner. It records portable
media and array contracts and rejects every confirmation object before real
visual/contact observability inputs are produced.

The
[atomic calibration-observability batch](deform360_calibration_observability_batch.md)
turns the ten calibration-object contact-versus-visual case builds and their
object-balanced report into one fail-closed publication. It retains technical
failures without replacement and publishes a valid negative result when the
frozen support gate is not met.

The
[confirmation-opening authorization](deform360_confirmation_opening_authorization_v1.md)
then binds a successful calibration-source terminal record, the supported
observability report, the complete Stage-1 evidence-use ledger, and the exact
frozen cohort before any confirmation payload may be opened.

## PhysTwin experiment families

| Area | Primary document | Status and purpose |
| --- | --- | --- |
| Released artifact and cue integration | [PhysTwin integration](phystwin_integration.md) | Pinned upstream contracts, residual export, cue sidecars, and likelihood boundary. |
| Hierarchical, graph, spatial, and discrepancy studies | [Advanced PhysTwin inference](phystwin_advanced_inference.md) | Experimental workflows, causal splits, controls, and interpretation limits. |
| Official Warp parameter refits | [PhysTwin refit](phystwin_refit.md) | Checkpoint restoration, matched baselines, provenance, and simulator requirements. |
| Causal MatPhys graph-part residual | [MatPhys graph-part residual](matphys_graph_parts_v1.md) | Corrected causal audit; source gate failed, so the family is frozen without a wider run. |
| Legacy MatPhys backbone experiments | [Causal MatPhys backbone](matphys_causal_backbone_v1.md) | Engineering history only; the original causal audit was invalidated by a frame-ordering defect. |
| Bias-aware guarded updates | [Bias-aware guarded belief](bias_aware_guarded_belief_v1.md) | Generic guarded-update method and source-only acceptance logic. |

Detailed commands live with the protocol they implement. New commands should be
registered under `bpt` and documented in their dedicated report rather than
appended to the root README.

## Cross-repository interfaces

- [Prob4D](https://github.com/IPS-Stuttgart/Prob4D) may emit the versioned,
  content-addressed `ObservationBeliefV1` artifact consumed here.
- [Causal4D](https://github.com/IPS-Stuttgart/Causal4D) owns Bayesian
  abduction-intervention-prediction and consumes the versioned provider and
  belief artifacts exported by Bayesian-PhysTwin.
- [Causal4D migration](causal4d_migration.md) records the historical command and
  tag boundary after Causal4D moved into its own repository.
- [BayesianPhysTwin-Paper](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper)
  is the canonical location for current scope, evidence status, figures, result
  artifacts, and paper-facing claims.

## Compute and repository hygiene

GPU host conventions and remote-run policy are documented in
[compute](compute.md). Pull-request source and automation rules are documented in
the [reviewable pull-request policy](reviewable_pull_request_policy.md). Large
datasets, checkpoints, rendered videos, and raw runs belong in ignored local
paths such as `data/`, `checkpoints/`, `runs/`, and `outputs/`, not in the source
repository.

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
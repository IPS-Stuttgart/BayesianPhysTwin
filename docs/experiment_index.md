# Experiment and evidence index

This page is the navigation layer for research workflows in BayesianPhysTwin.
The root README documents the stable project identity, interfaces, onboarding
path, and bounded current evidence. Claim-bearing numbers belong in frozen
evidence reports, and current ecosystem status belongs in the canonical paper
repository.

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
state the identifiability, covariance, causal-cutoff, convergence-admission, and
exact-fallback rules that apply independently of any one experiment.

## Command lifecycle

Use the registry rather than maintaining a second static command list:

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

### Provider failure decomposition

[`bpt diagnostic run diagnose-provider-failures`](provider_failure_decomposition.md)
attributes frozen source-only provider and guarded-update failures to a fixed,
equal-case taxonomy. It preserves multiple observed causes, leaves unresolved
rejections explicit, and never changes an acceptance decision or authorizes
target access.

## Frozen and claim-bearing evidence

### Official PhysTwin full-22 evaluation

The [PhysTwin full-22 evidence report](phystwin_sota_22_v1.md) is the frozen
source for the released-cohort comparison, confirmation intervals, render
reproduction, provenance, ownership boundary, and permitted within-contract
claim.

The [release-facing claim contract](phystwin_release_claim_v1.md) binds the
positive result to its required companion evidence:

- the Bayesian anchor improves equal-case Chamfer distance and track error by
  `12.09%` and `12.78%` versus re-evaluated released PhysTwin;
- last residual is the principal matched deterministic comparator and is
  marginally better on equal-case track error;
- raw posterior covariance is severely undercalibrated; and
- conformal coverage is a separate width-bearing result under stated
  assumptions.

The result supports improvement over released PhysTwin under the recorded
contract. It does not support a unique deterministic-winner claim, calibrated
raw covariance, dynamically identified state correction, independent-object
transfer, or overall state of the art.

The
[frozen full-22 reproduction capsule](../reproductions/full22_anchor_v1/README.md)
records the exact source revision, protocol and data identities, two-stage
source command, expected paper-facing metrics, fail-closed verification, and a
strict `RunManifestV2` bundle. The capsule has been reproduced independently on
`workstation2` and a fresh GitHub-hosted runtime.

### Controlled Prob4D-to-BayesianPhysTwin mechanism

The controlled explicit-joint-gauge benchmark is positive on a disjoint
synthetic calibration/target split. Persistent explicit-gauge deployment changes
RMSE from `6.166` to `0.534 mm`, accepts `373/384` target groups, has zero
harmful accepted updates, and retains exact fallback for every rejection.

This is controlled mechanism evidence. It does not establish real provider
competence, fresh-object physical benefit, deployment calibration, or Causal4D
benefit. The current real-provider evidence below remains decisive for physical
escalation.

### Real-provider and Deform360 boundary

A retrospective MotionCrafter transfer on 19 already-open PhysTwin interactions
was negative: physical fallback is `6.899 mm`, marginal-gauge deployment is
`6.942 mm`, and the explicit-persistent guard accepts no updates and reproduces
fallback exactly. That cohort may not be tuned into replacement confirmation.

The later
[official-Hub Deform360 visuotactile v1 protocol](deform360_official_hub_visuotactile_v1.md)
froze ten calibration objects, twelve distinct confirmation objects, every
admitted camera stream, and no replacement after source-outcome access.

Its completed execution chain is:

1. all `10/10` calibration objects were prepared;
2. all `324/324` admitted visual-production jobs succeeded;
3. the frozen released robot/camera support gate retained `11` support-negative
   streams, with `313/324` supported and `0` technical failures; and
4. no source covariance was fitted, no leave-one-object-out source gate was
   evaluated, and confirmation access was not authorized.

The provider version is terminal at this source-support boundary. Deleting
cameras, fitting only the supported streams, changing the fixed prefix,
replacing objects, or opening the twelve confirmation objects would violate the
frozen information order. This is a valid real-data source-support negative, not
a fitted-covariance failure.

The implementation documents remain useful as frozen contracts and execution
history:

- [visual-provider freeze](deform360_visual_provider_freeze.md);
- [prepared-source inventory](deform360_calibration_prepared_inventory.md);
- [atomic calibration-observability batch](deform360_calibration_observability_batch.md);
  and
- [confirmation-opening authorization](deform360_confirmation_opening_authorization_v1.md).

They must not be read as an active authorization chain for the completed provider
version. Exact run, artifact, per-stream, and unopened-confirmation evidence is
maintained in the canonical paper repository's
[Deform360 source-support result](https://github.com/FlorianPfaff/BayesianPhysTwin-Paper/blob/main/docs/deform360_prob4d_source_support_negative_2026-08-09.md).

A future Deform360 attempt requires a separately versioned provider or protocol
whose complete camera/robot support feasibility is frozen before source residual
outcomes are opened. The ten opened calibration objects cannot become a fresh
replacement confirmation cohort.

### Prospective PokeFlex transfer

A conservative action-local correction has positive prospective new-interaction
evidence on previously studied PokeFlex objects. The fresh12 and disjoint fresh6
panels improve their frozen object-balanced metrics, while the retrospective
public official13 panel corroborates the direction without replacing unavailable
official takes.

This supports a bounded same-object new-interaction claim, not unseen-object
generalization, per-frame safety, general uncertainty calibration, or a
full-split state-of-the-art claim. See the dedicated frozen result documents and
the canonical paper-side manuscript for exact numbers and provenance.

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
registered under `bpt` and documented in a dedicated report rather than appended
to the root README.

## Cross-repository interfaces

- [Prob4D](https://github.com/IPS-Stuttgart/Prob4D) may emit the versioned,
  content-addressed `ObservationBeliefV1` artifact consumed here.
- [Causal4D](https://github.com/IPS-Stuttgart/Causal4D) owns Bayesian
  abduction–intervention–prediction and consumes versioned provider and belief
  artifacts exported by BayesianPhysTwin.
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
3. Put frozen machine-readable evidence under `results/` or the evidence
   location named by the protocol.
4. Put current paper scope and claim status in `BayesianPhysTwin-Paper`.
5. Put Causal4D-specific methods and acquisition status in the Causal4D
   repository.

This separation keeps onboarding stable while preserving negative results,
complete experimental provenance, and the distinction between engineering
hardening and empirical evidence.

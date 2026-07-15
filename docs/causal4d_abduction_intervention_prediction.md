# Causal4D Abduction, Intervention, and Prediction

Software status: implemented and audited on 2026-07-12. Paper status: not yet
claim-ready because the locked multi-action real protocol remains unexecuted.

This track is independent of the Bayesian-PhysTwin estimation paper. It turns
the earlier rollout-bank pilot into an explicit causal architecture in which a
commanded action is not assumed to equal the intervention realized by the
object:

```text
u_t -> z_t = (phi, kappa_t) -> realized contact/forces -> x_{t+1}
```

`phi` contains persistent or slowly varying actuation variables: gain, delay,
and controller-frame rotation. `kappa_t` contains event variables: graph
attachment and slip. The physical belief also retains model discrepancy
`delta` separately from simulator state.

## First-paper claim

The first paper makes one claim:

> **Bayesian abduction of realized interventions for counterfactual prediction
> of deformable-object dynamics.**

Its evidence chain is the command/realization distinction, joint posterior
inference, explicit abduction-intervention-prediction, controlled held-out
contact/action gains, same-object multi-action real validation, and calibrated
uncertainty or a bounded calibration limitation. The locked hierarchy and
current readiness decision are in `docs/causal4d_paper_scope.md`.

MolmoMotion is not part of the core method or main experiment matrix. The
closed-loop runner is an application, not a robotics contribution without
genuine robot execution.

## Optional semantic posterior boundary

The implementation never uses language as evidence about the present twin.
It maintains two distinct distributions:

```text
p_phys(X_cf | D, do(u_cf))

p_task(X_cf | D, do(u_cf), language)
  proportional to
p_phys(X_cf | D, do(u_cf)) q_MM(H_Q(X_cf) | I, language)^beta
```

`H_Q` reads only the sparse MolmoMotion query nodes from a dense physical
rollout. The semantic factor cannot modify state, physical parameters, model
discrepancy, or the physical posterior artifact.

## Typed artifact contract

Every artifact identifies all four causal inputs: pre-intervention
observations `O-`, post-intervention observations `O+`, factual command
`u_obs`, and counterfactual command `u_cf`. Frame intervals are half-open and
array payloads are hashed.

| Artifact | Contents |
| --- | --- |
| `TwinBelief` | particles `(x_t, v_t, theta, delta, weight)` |
| `FactualIntervention` | posterior over `(theta, phi, kappa_obs)` after an `O+` prefix |
| `CounterfactualQuery` | explicit `do(u_cf)` and same/new-contact policy |
| `PhysicalPosterior` | dense state rollouts, discrepancy-aware readouts, and conditional variance |
| `TaskPosterior` | separate semantic scores and task weights over immutable physical support |

NPZ serialization is non-pickled and checksummed. Tests change every withheld
future value and verify that prefix-only beliefs remain byte-identical.
`TaskPosterior(beta=0)` is required to preserve physical weights byte for byte.

## Full Bayesian-PhysTwin belief

The old pilot crossed spring parameters with future rollouts but restarted all
particles from one released endpoint. That shortcut is removed from the public
backend API.

For each retained spring particle, the exporter now:

1. replays the official Warp simulator through `O-` only;
2. retains that particle's endpoint position and velocity;
3. filters its tracked residual history with the fixed robust random-walk
   discrepancy model;
4. lifts discrepancy mean and variance to the complete object graph;
5. stores discrepancy as a readout/process field, never as a state injection.

On `single_lift_sloth`, all four retained particles produced distinct endpoint
states. The maximum pairwise endpoint RMSE was `1.152 mm`; the particles retain
`42.33%` of the original 9 by 9 profile mass.

## Factual abduction

The factual action bank is scored against only the first six `O+` frames. The
likelihood combines each Warp rollout with its particle-specific discrepancy
mean and variance. It updates the complete joint support over physical
particles and intervention hypotheses.

For `single_lift_sloth`, the untouched remainder gives:

| Method | Coordinate RMSE | Track error |
| --- | ---: | ---: |
| BPT, nominal `z` | 22.494 mm | 32.130 mm |
| BPT + Causal4D `z` | **22.260 mm** | **31.694 mm** |

The track improvement is `1.36%`. Nominal contact remains the MAP hypothesis
with probability `25.29%`; the gain comes from marginalization, not a claim
that the real attachment was recovered. Controlled tests with known latent
interventions verify recovery directly.

## Counterfactual operator

The operator implements the three causal steps explicitly:

1. **Abduction:** infer `(theta, phi, kappa_obs)` from the factual response.
2. **Action:** replace the command mechanism with `do(u_cf)`.
3. **Prediction:** transfer `(theta, phi)` and either retain `kappa_obs` for the
   same grasp or sample a fresh `kappa_cf` for a new contact.

A real `history_reverse` query produced 36 official Warp components with
effective support `26.06` and retained essentially all factual `(theta, phi)`
mass. The new-contact and same-grasp branches have different contact
marginals, confirming that factual contact is not silently reused.

## Physical-only validation

The five-seed controlled benchmark remains the causal validation result. It
uses held-out actions and leave-one-topology-out contact-model fitting, with
MolmoMotion absent (`beta=0`). The 2026-07-12 rerun passed every registered
gate:

| Metric | Nominal physics | Latent contact |
| --- | ---: | ---: |
| Shifted-contact RMSE | 4.132 mm | **0.805 mm** |
| Shifted-contact 90% coverage | 77.9% | **90.8%** |
| Shifted oracle-gap closure | - | **80.6%** |
| Matched-contact RMSE | 2.463 mm | **2.046 mm** |

All three excluded topologies have positive oracle-gap closure; the minimum is
`60.8%`.

The real typed physical posterior improves mean prediction but is not
calibrated: nominal 90% coordinate coverage is only `50.6%`, with NEES `7.23`.
This is recorded as a limitation, not repaired post hoc.

## Real oracle-gap diagnosis

A leakage-explicit audit freezes the six-frame `O+` evidence boundary and
compares the current 9-state intervention bank with the complete nested
108-state grid. All nine current trajectories are bit-identical in the
expanded bank.

On the untouched future, current Causal4D track error is `31.694 mm`, the
current-bank component oracle is `29.378 mm`, the expanded-bank oracle is
`29.071 mm`, and an expanded component plus an in-sample constant per-node
discrepancy ceiling reaches `8.399 mm`. The resulting headroom is `9.94%`
inference, `1.32%` proposal, and `88.74%` model discrepancy. With every point
correction capped at `10 mm`, model discrepancy remains dominant at `76.29%`.

The current posterior variance is dominated by conditional discrepancy
(`60.66%`) and the configured conditional floor (`22.92%`). Shapley-allocated
state uncertainty contributes `10.97%` from `kappa`, `3.82%` from `theta`, and
`2.15%` from `phi`. Empirical residual MSE is 4.54 times total predictive
variance and the ratio worsens across the horizon.

This rules out wider handcrafted intervention enumeration as the next modeling
priority. A graph-regularized rest-geometry/frame correction remains the first
model-discrepancy test. The main evidence work package is now the preregistered
same-object multi-action real protocol in
`docs/causal4d_same_object_multi_action_protocol.md`, so model quality,
intervention transfer, and held-out calibration can be measured separately.
Full oracle-audit methods and commands remain in
`docs/causal4d_real_oracle_audit.md`.

The subsequent undercoverage audit confirms that conclusion. Full 81-particle
support reaches only `55.05%` coverage. A rank-16 graph-persistent discrepancy
reduces track error to `23.105 mm` and raises coverage to `67.78%`, while learned
AR residual dynamics transfers poorly. A prelocked affine calibration fitted on
`double_lift_sloth` and calibrated on `double_stretch_sloth` is harmful on the
single-lift target (`43.03%` coverage), so it is rejected rather than retuned.
See `docs/causal4d_real_undercoverage.md` for the complete metrics and claim
gate.

## Semantic posterior and trust

MolmoMotion is applied only through `H_Q`. On the real `history_reverse`
posterior:

- `beta=0` gives KL `0` and byte-identical physical/task weights;
- `beta=12` changes only task weights (KL `1.24e-4`);
- the physical posterior checksum is unchanged.

The trust layer selects beta on source validation futures and applies
label-free target OOD checks for static motion, motion-scale mismatch, distance
from physical support, and anchor misalignment. Unit tests prove that static
and physically implausible forecasts fall back exactly to the physical
posterior.

On the real source validation action, the strongest beta improves RMSE by only
`0.12%`, below the locked `0.5%` minimum. The selected beta is therefore zero.
The hidden-action query is rejected with byte-identical fallback weights. This
formalizes the earlier MolmoMotion null instead of accepting a harmful prior.

Positive semantic weighting also has an independent runtime admission gate.
`causal4d.semantic_freshness` requires strict timing metadata on a trusted,
named monotonic clock; it rejects missing or mismatched clock identity,
overlong inference, an aged query snapshot, a missed planning deadline, and
malformed telemetry. Every rejection reconstructs the task posterior with
`beta=0` and byte-identical physical/task weights. The expected clock identity
must come from the planner runtime, never from the semantic payload. The locked
limits and failure behavior are in
`configs/causal4d/hardware_execution_gate_v1.json`.

A subsequent direct competence audit corrects the original 30/15 fps temporal
mismatch and evaluates Molmo before beta selection. The corrected instruction
still reaches only `0.0164` times the real motion scale, does not beat zero or
constant velocity, and ranks the true lift fifth of five for all three
paraphrases. See `docs/causal4d_molmo_acceptance.md`. The semantic branch remains
optional and disabled; no beta tuning is warranted for this checkpoint.

## Closed-loop planning

`causal4d.closed_loop` provides a constrained receding-horizon runner. Each
cycle:

1. obtains freshly simulated candidate plans from the current endpoint state;
2. rejects control-step, state-displacement, or predictive-risk violations;
3. ranks feasible plans with optional semantic evidence plus effort/risk cost;
4. executes only a short control segment;
5. updates physical component, `theta`, `phi`, and `kappa` weights from new
   observations, starting from the physical rather than task posterior;
6. optionally propagates and robustly updates a separate low-rank graph
   discrepancy mean/covariance from partial node/coordinate observations;
7. passes particle endpoint position and velocity to the next simulator call;
8. moment-matches the graph coefficients onto matching
   `(particle, phi, kappa)` support in each new plan and replans.

Graph persistence is the default coefficient transition because it is the
supported real-data baseline; learned graph AR dynamics require an explicit
opt-in. The transported field corrects readout moments and predictive risk but
is never injected into simulator position or velocity. These are software and
unit-test contracts, not evidence of a new real-data accuracy or calibration
gain.

The controlled closed-loop test rejects an unreachable action, completes a
language-conditioned task with two replans, and updates the correct physical
particle. A real-artifact replay also completes two update/replan cycles. This
is software validation, not a real-robot success claim.

Physical closed-loop execution is blocked until the source-only calibration and
safety criteria in `configs/causal4d/hardware_execution_gate_v1.json` pass. A
positive semantic beta is an additional optional gate, not a substitute for
calibrated physical risk.

## Commands

Export a complete endpoint belief:

```bash
causal4d-export-bpt-belief \
  PHYSTWIN_REPO CASE parameter_profile.npz refit_checkpoint.pt belief.npz
```

Build an observed-action bank and abduce the factual intervention:

```bash
causal4d-phystwin-rollout-bank \
  PHYSTWIN_REPO CASE parameter_profile.npz refit_checkpoint.pt known.npz \
  --action-setting known --twin-belief belief.npz

causal4d-abduct-phystwin-intervention \
  known.npz belief.npz CASE/final_data.pkl factual.npz factual_eval.json
```

Apply a counterfactual and evaluate a physical holdout:

```bash
causal4d-counterfactual-phystwin \
  PHYSTWIN_REPO CASE parameter_profile.npz refit_checkpoint.pt \
  belief.npz factual.npz physical.npz \
  --counterfactual-action-id history_reverse --contact-policy new_contact

causal4d-evaluate-physical-counterfactual \
  physical.npz CASE/final_data.pkl beta0_eval.json
```

Create and gate a separate MolmoMotion task posterior:

```bash
causal4d-build-molmo-task-posterior \
  physical.npz molmo.npz instruction task.npz --beta 0

causal4d-fit-semantic-trust source_manifest.json semantic_trust.json \
  --minimum-relative-improvement 0.005 \
  --molmo-acceptance-json molmo_acceptance_result.json

causal4d-adaptive-molmo-task-posterior \
  physical.npz molmo.npz instruction semantic_trust.json \
  adaptive_task.npz trust_decision.json
```

The real belief, abduction, counterfactual, and beta-zero sequence is also
available as `scripts/remote/run_causal4d_abduction_pipeline.sh` for the two
configured GPU servers.

The expanded-bank diagnostic is available as
`scripts/remote/run_causal4d_real_oracle_audit.sh`.

## Claim boundary

The controlled causal result supports the core mechanism. The complete
first-paper claim is not ready until the same-object multi-action real protocol
is executed and independent-execution calibration is evaluated. Current real
evidence is one interaction, a truncated parameter posterior, dominant
simulator/state discrepancy, and undercovered uncertainty. The complete
intervention grid closes only 1.32% of diagnostic headroom, so beam width is not
the primary real limitation.

MolmoMotion remains rejected and cannot enlarge the claim. Closed-loop replay
establishes software behavior only. These components stay optional/application
material, and neither expands the Bayesian-PhysTwin estimation paper's claim
set.

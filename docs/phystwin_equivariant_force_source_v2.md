# Equivariant generalized-force source gate

Lock date: 2026-07-24

Status: v2 source protocol amended and locked before Stage 1; no source GPU run
or target access has occurred.

## Why v1 was stopped

The first target-only preflight found that PhysTwin assigns every released node
simulation mass `1.0`. These values are not kilograms: interpreting them that
way would make the released objects weigh 855 to 8,582 kg. Consequently, the
v1 labels in Newtons were invalid, and its fixed `0.5` cap saturated 43% to 92%
of supported targets. Stage 1 was not run.

V2 keeps all values in native Warp generalized-force units. It estimates one
robust scale per case from the allowed prefix only, trains and scores normalized
forces, and multiplies the bounded equivariant field by that frozen scale during
rollout. The scale is part of the single robust innovation channel; it is not a
perception-reliability cue.

## Why this branch exists

Static spring fields, topology variants, pooled residual MLP/GNN models, PGRD,
and canonical triplane residual dynamics did not transfer on the registered
PhysTwin source cases. The matched discrepancy-localization audit also rejected
a constant low-rank force. Those failures leave a narrower hypothesis:
the missing term may be a bounded, state- and regime-dependent generalized
force that must be propagated by the official nonlinear simulator.

The candidate predicts scalar coefficients over physical vector bases. Relative
edge displacement, spring strain, relative velocity, controller displacement,
controller velocity, support, gravity, and a small case latent determine those
coefficients. Internal edge messages are antisymmetric. External terms are
support-gated. The resulting field is E(3)-equivariant and capped per node.

This is not a readout residual model and is not a reproduction of NeuSpring,
DeformMaster, or MatPhys.

## Two-stage gate

Before Stage 1, target QA requires the native simulator-unit contract, released
unit masses, prefix-only scale estimation, and no more than 10% cap saturation
on every allowed prefix. A failure blocks training.

Stage 1 tests source-only inverse-dynamics competence. Complete outcomes from
the non-held-out interactions supervise shared weights. For a held-out
interaction, only its latent is adapted from `[0, fit_end)`, and force targets
are scored on `[fit_end, train_end)`. The result is diagnostic: passing force
RMSE does not authorize a simulator claim.

Stage 2 is the promotion gate. Candidate and reference start from the same
released state at the allowed prefix endpoint. Each receives a separately
prefix-fitted graph-persistence readout. The only primary-arm difference is the
learned force versus exact zero force inside the pinned official Warp simulator.
Prefix state injection remains a diagnostic control. The candidate must:

1. pass Stage 1;
2. improve equal-case CD and manual-track error by at least 3% jointly;
3. improve both aggregate metrics in at least two of three whole-case folds;
4. keep every case metric ratio at or below 1.05;
5. improve late-horizon error;
6. shrink the graph-persistence correction by at least 10% in 11 of 17 cases;
7. preserve bitwise identity when force admission is zero.

The force candidate is evaluated with graph persistence refitted on top. This
factorial comparison measures whether the physical mechanism explains residual
variance rather than merely coexisting with the existing patch.

## Causal boundary

Force targets are derived from the residual acceleration of trusted source
observations relative to the registered baseline. Visibility and other
residual-independent perception cues set prior reliability. The state
innovation then enters once through a robust local-polynomial fit. It is never
reused to lower prior reliability.

Dense targets are graph-smoothed with metric acceleration variance before being
converted to native simulator generalized-force units. They are never presented
as Newtons or material-force measurements. This preserves heteroscedastic
uncertainty and avoids treating correlated points as independent evidence.

For each held-out source case:

- global weights are trained without that case;
- only the latent sees `[0, fit_end)`;
- rollout initialization and residual refitting use the same permitted prefix;
- scoring uses `[fit_end, train_end)`;
- no target cohort artifact is read.

The five historical target interactions have already been examined by earlier
branches. Even if a source pass later permits an exploratory run there, that
run cannot establish an independent state-of-the-art claim. A fresh locked
cohort is required for confirmation.

## Exact fallback

`admission_weight = 0` bypasses the force policy and delegates to the existing
official-Warp rollout after clearing the external-force buffer. Unit tests
verify exact zero model output and delegation. The native Warp run must also
establish bitwise trajectory parity before Stage 2 can start.

## Current verification

Local CPU tests cover:

- translation invariance and rotation equivariance;
- antisymmetric internal-force conservation;
- support gating and force bounds;
- typed/checksummed model and episode artifacts;
- robust target recovery, uncertainty, graph lifting, and future mutation
  invariance;
- exact zero admission;
- synthetic transfer to an unseen rotated interaction;
- disjoint complete cross-fitting;
- the rule that source force competence never authorizes Warp promotion.

GPU source training is intentionally deferred while both configured compute
servers are reserved by independent registered experiments.

The immutable v2 source build has since passed target QA on all 17 cases:
prefix cap fractions are 1.66% to 4.96% against the frozen 10% limit. Native
verification passes 35 focused tests and 1,055 full-suite tests with 4 skips.
Stage 1 remains unstarted.

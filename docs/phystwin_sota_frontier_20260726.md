# PhysTwin SOTA frontier audit

Audit date: 2026-07-26

Latest public-release recheck: 2026-07-27

Status: source-only research decision. No held-v8 barrier, query, target,
outcome, or score artifact was opened or modified.

## Question

This audit asks whether a newly released public method provides a credible,
future-blind route past the current PhysTwin state of the art. A route is
actionable only when its public artifact exposes enough provenance to enforce
the PhysTwin observation/future boundary and enough implementation to run a
matched source competence test.

## Public asset audit

| Method | Public state | Local decision |
|---|---|---|
| [PGRD](https://github.com/shivanshpatel35/pgrd) | Training code, data, and checkpoints; current `main` is `e294d96723054f77a1cfdd3c2c052de7b7cd9ce3` | No new run. This is the exact commit already used by the rejected zero-shot, temporal-head, and unrolled adapters. |
| [DeformMaster](https://github.com/CAN-Lee/DeformMaster) | Inference/playground code at `c7b3510a38b3fccbfe12cc6557aaf58d9ea823dc`; 20 PhysTwin checkpoints in the official asset bundle; full training code is explicitly pending | Do not score the released checkpoints as causal predictors. The public artifact does not encode a reproducible train/test boundary. |
| [NeuSpring](https://github.com/GhiXu/NeuSpring) | Repository `main` at `51d94f67ed1e2557fca29c1e86b418506e3d51ca`, containing only a two-line README | No executable reproduction or checkpoint overlay is available. |
| [EgoPhys](https://hjhyunjinkim.github.io/EgoPhys/) | Project page labels both code and dataset as coming soon | Revisit only after a versioned implementation and data release. |
| [BoxTwin](https://arxiv.org/abs/2607.17132) | Paper released 2026-07-19; no public implementation or matched PhysTwin benchmark artifact was found | Adjacent mechanism evidence only. Its nonlinear hinge elasticity, plastic rest-angle state, and damage evolution target elastoplastic articulated objects rather than the elastic rope, cloth, package, and stuffed-object full-22 contract. |

The DeformMaster artifact deserves a precise boundary. Its official
`playground_assets.zip` has SHA-256
`9a10ba5149685007b838363989675ba20ff8ee6e33aa829e1268ce68892746c6`.
For the released `double_lift_sloth` case, the bundle contains 62 object-track
frames, while the released configuration sets `mpm.max_frames: 300` and
contains no split field. The public dataset loader returns all available
tracks up to that maximum. The paper describes unseen future evaluation, but
the missing training loop and absent split metadata prevent an independent
check that the released checkpoint obeys that boundary. This is an artifact
provenance limitation, not evidence that the reported paper result is invalid.

DeformMaster reports `0.0114` m Chamfer and `0.0240` m track error over a
20-case PhysTwin subset. Those values use a different cohort from the
published full-22 MatPhys point (`0.0080` m / `0.0150` m) and do not constitute
a new matched numerical target for the present 22-case ledger.

BoxTwin is scientifically relevant because it makes internal material state
explicit instead of forcing every persistent error into a static spring field.
It does not currently authorize a PhysTwin experiment: its public evaluation
uses articulated hinge coordinates and elastoplastic objects, not the full-22
node-trajectory metrics, and no released code or checkpoint provides a
future-blind adapter. A future release would justify a mechanism study only on
cases with independently observable yielding, hysteresis, or permanent set.

## Closed local families

The following source-gated experiments already cover the obvious public-model
adapters and increasingly expressive residual variants:

| Family | Source result | Consequence |
|---|---|---|
| PGRD zero-shot, temporal-head, and five-step unrolled transfer | All untouched development gates failed | The unchanged upstream release does not justify another adapter run. |
| Per-object zero-order topology and spring field | Transfer gate failed; `0/4` two-metric wins | Static topology/field search is not a reliable selector. |
| Canonical triplane residual dynamics | Best blend regressed CD by `0.521%` and track by `0.071%` | Canonical local feature planes do not rescue pooled residual transfer. |
| Equivariant generalized force | Stage-1 source competence improved normalized force RMSE by only `0.97%`, below the frozen `10%` gate | Official-Warp trajectory Stage 2 was correctly not authorized. |
| Shared graph-spectral discrepancy dynamics | Best nonzero arm changed CD by `+1.06%` and track by `-0.58%`; no whole fold won both | Exact endpoint persistence remains selected. |

The equivariant-force record is on branch `equivariant-force-v1` at commit
`2a8a675b0324f93cb4462e92540ecf7bfcb25cc0`.

The graph-spectral record is on branch
`phystwin-spectral-discrepancy-dev-v1` at commit
`7e84e27131f9b4e53c775a2df3754802de3e421f`. Its source summary SHA-256 is
`83ce90e6414aa5a41805e5a6c07e169d00735d76a9ddf89751dd5b4d1019f574`.

These failures do not reject learned dynamics in general. They reject the
available transferable model families under the registered source panels.
Their shared lesson is that a correction learned without independently
supported information about the realized action or current physical state is
not reliably better than persistence.

## Current research decision

No additional public-backbone GPU experiment is authorized by this audit. The
only currently executable route with demonstrated open-source headroom is the
separately owned, baseline-relative guarded Bayesian online belief update:

1. retain the selected physical/persistence trajectory as an exact fallback;
2. admit an update only with physical/action support and structurally
   redundant observation evidence;
3. model shared camera/time/spatial bias rather than treating internally
   consistent views as independent truth;
4. calibrate an upper confidence bound on regret against the unchanged
   baseline using source data only; and
5. preserve bit-exact fallback when the admission or regret gate fails.

This note does not report or authorize any held-v8 result. That evaluation has
separate ownership and must remain sealed until its registered operator
reports the independent gate.

## Revisit triggers

Reopen an external family only when at least one concrete boundary changes:

- DeformMaster releases full training code and a versioned per-case
  observation/future split, or future trajectories with equivalent provenance.
- NeuSpring releases its implementation, checkpoints, and evaluation split.
- EgoPhys releases code and data with a transferable PhysTwin interface.
- BoxTwin releases code and a constitutive-state interface applicable to
  non-articulated PhysTwin graphs, together with a reproducible observation and
  prediction split.
- PGRD releases a materially different checkpoint or graph/action interface,
  rather than a new wrapper around commit `e294d967`.
- A genuinely independent modality, such as sparse depth, LiDAR, or tactile
  contact evidence, supplies calibrated information that can identify
  common-mode camera bias.

Until then, another camera-only residual, fixed blend, spring-field search, or
PGRD adapter would duplicate a closed family rather than advance the evidence.

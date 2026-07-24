# Equivariant-force official-Warp Stage-2 execution lock

Lock date: 2026-07-24

Status: locked after v2 source-target QA and before any Stage-1 training
outcome. No historical target artifact was opened.

## Why this amendment exists

The source-v2 protocol fixed the correct model, target, folds, seeds, and gates,
but its short Stage-2 description left two implementation choices open:

1. whether the candidate force acts during the observed prefix or only after a
   released prefix endpoint;
2. whether the three seeds produce three scored rollouts or one deployable
   force policy.

Those choices can materially change correction shrinkage and trajectory error.
They are now fixed before Stage 1 in
`configs/sota/phystwin_equivariant_force_stage2_v1.json`. The amendment is
bound to source protocol SHA-256
`1178ffe1545158225818723c700991f76d730c3627ab09644b73f2a14f53a171`.
It does not alter the already-archived v2 force episodes or their target QA.
The Stage-2 JSON SHA-256 is
`4a9f5730e8b14071b14013dab20bf6c71a26a3a923abf430c0b4113699db31c4`.

The CPU-only preflight, still before Stage 1, also closed a provenance gap
without changing any execution semantics. The lock now binds upstream
PhysTwin commit `2b66305`, tree `c52a322`, and the exact imported
`spring_mass_warp.py` SHA-256
`7deab9a25f4b8b8772f7df45c35571caf3767d014dd353cad151fe8eddceca1c`.
The 17-case source manifest at
`configs/sota/phystwin_equivariant_force_stage2_source_manifest_v1.json`
has SHA-256
`e1c0ff0171291342540227cb2cbeac024c8a9b7e13e0921cf37738a95e83a40a`.
It binds every source replay, observation archive, parameter file, checkpoint,
manual-track file, split, and force-episode identity. It contains no target
path, hash, or outcome. The Stage-2 evaluator package is independently locked
by implementation SHA-256
`b9d6e51eb2d89193ecf0af97e28a42a91acae4700de3d043d0cbce0ca89a98fd`;
the protocol loader, every case record, and the aggregate gate verify this
identity.

## Frame contract

Frame zero is the common released initial state. For a source case with
exclusive split boundaries `fit_end_frame` and `train_end_frame`:

- official Warp steps use controller frames
  `[1, train_end_frame)`;
- the candidate ensemble force acts on every one of those steps;
- the reference external force is exactly zero on every step;
- each arm fits its own graph-persistence correction from
  `[0, fit_end_frame)`;
- metrics use only `[fit_end_frame, train_end_frame)`.

Thus the learned force may change the state during the allowed prefix. The
candidate correction measures the residual still unexplained after that
physical propagation. Resetting both arms to the same released prefix endpoint
would erase this explanatory pathway, while fitting a candidate correction
from a replay that is not used for continuation would create a discontinuous
hybrid trajectory.

Both arms use the same released optimal PhysTwin parameters, initial state,
controller trajectory, graph, solver settings, and readout procedure. Before
readout refitting, their only difference is the learned generalized force.

## Seed contract

The three frozen seed models remain paired with their separately adapted
held-out latents. At every Warp frame they evaluate the same current state and
controller conditioning. Their force fields are averaged in float64 and cast
to float32 once before injection:

\[
f_t^{\mathrm{ens}} =
\frac{1}{3}\sum_{s\in\{17,43,101\}} f_t^{(s)}.
\]

There is no seed selection. Convex averaging preserves equivariance and the
per-node force bound. Admission weight zero bypasses every model and retains
the exact zero-force fallback.

## Readout refit

For each physical arm, the frozen robust random-walk endpoint filter is applied
to its own prefix innovation. The endpoint is smoothed over the normalized
released spring Laplacian with fixed prior strength `0.3` and clipped at
`10 mm`. The resulting field is held constant over the scoring interval.

Correction shrinkage is an amplitude statistic:

\[
S = 1 -
\frac{\operatorname{RMS}_i\lVert c_i^{\mathrm{candidate}}\rVert_2}
     {\operatorname{RMS}_i\lVert c_i^{\mathrm{reference}}\rVert_2}.
\]

A reference correction below `1 micrometre` has insufficient signal and cannot
count toward the shrinkage-case gate. Laplacian energy is reported only as a
diagnostic and does not replace the amplitude gate.

Late-horizon track error uses the repository's existing count-balanced
early/middle/late split (`numpy.array_split`) and reports the final group. This
is the same horizon convention used by the prior PhysTwin horizon analysis.

## Mechanical evaluator

After Stage 1 passes, one registered source case is run with:

```bash
bpt-gate-phystwin-equivariant-force official-warp-case \
  /path/to/PhysTwin \
  /path/to/source-data \
  /path/to/v2-episodes \
  /path/to/source_competence_record.json \
  configs/sota/phystwin_equivariant_force_source_v2.json \
  configs/sota/phystwin_equivariant_force_stage2_v1.json \
  CASE_ID \
  /path/to/output/CASE_ID \
  --device cuda:0
```

The evaluator refuses a failed Stage-1 record before resolving case data. It
verifies model, latent, episode, source-manifest, upstream simulator,
source-data, protocol, checkpoint, manual-track, and split hashes;
replays exact zero force and the learned ensemble from the same frame-zero
state; refits both readouts; writes checksum-bound trajectory arrays; and emits
the case record consumed by the mechanical source gate.

CPU verification with GPUs hidden passes 8 focused source-manifest and
official-Warp tests and the complete suite: 1,064 passed and 4 skipped.
The immutable deployment-readiness record is
`results/sota/phystwin_equivariant_force_stage2_v1/preflight.json`, SHA-256
`5e057c03c687fba56a66e64594b4170c02e88e5d6dcb194dd4e35bc35db95524`.
It leaves Stage 2 blocked until the registered Stage-1 gate passes and an
explicit GPU release is observed.

## Claim boundary

Stage 1 remains an inverse-dynamics competence test. Stage 2 may run only if
Stage 1 passes. A Stage-2 source pass authorizes a fresh preregistered
evaluation; it does not authorize opening the five historical targets as
confirmatory evidence and is not itself a state-of-the-art result.

The source-v2 protocol and episode archive remain immutable. This document and
the separate machine-readable amendment supersede only the three ambiguous
Stage-2 prose fields named in the amendment.

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
`b0378a6157c9e731da19a2dceed61287a57c9d1527a14890a3abc1e0ecc21b55`.

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

## Claim boundary

Stage 1 remains an inverse-dynamics competence test. Stage 2 may run only if
Stage 1 passes. A Stage-2 source pass authorizes a fresh preregistered
evaluation; it does not authorize opening the five historical targets as
confirmatory evidence and is not itself a state-of-the-art result.

The source-v2 protocol and episode archive remain immutable. This document and
the separate machine-readable amendment supersede only the three ambiguous
Stage-2 prose fields named in the amendment.

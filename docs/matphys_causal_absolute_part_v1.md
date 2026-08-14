# Causal absolute part-aware MatPhys adapter

The positive all-frame reconstruction control showed that the DINO graph-part
adapter can express a strongly spatial spring field, but that checkpoint used
future observations and is unusable for prediction. This successor tests the
same architectural capacity under a causal prefix boundary.

This is a BayesianPhysTwin adapter, not the published MatPhys method. The
public MatPhys release does not provide its final semantic part bundle, so the
input is the existing deterministic DINO graph partition. Unlike the rejected
teacher-centered family, this mode predicts the complete positive spring field
directly. It does not make a small residual around released PhysTwin stiffness.

## Fail-closed contract

The `--absolute-part-field` flag is accepted only when all of these hold:

- exactly one case is trained under `causal-prefix-only`;
- graph parts and a positive part-feature scale are enabled;
- initialization is fresh from the generic VideoMAE backbone;
- teacher residuals, teacher proximity, and checkpoint initialization are off;
- the transactional finite optimizer guard is on;
- the fixed terminal checkpoint is used.

The causal training audit stores these choices under
`matphys-causal-absolute-part-field-v1`. Export keeps this record separate from
the older teacher `parameterization`, writes the complete applied spring field,
and reports absolute per-part stiffness rather than a teacher-relative ratio.
The audit also rehashes the exact runner, adapter, bridge, and registered
protocol bytes; changing any of them invalidates export.

## Frozen competence sequence

The exact protocol is
`configs/sota/matphys_causal_absolute_part_competence_v1.json`. It uses the
already-open `single_lift_zebra` case, whose fit evidence ends at frame 34.
Frames 34--45 remain unused prefix holdout and frames 46--65 are the released
future interval.

First run one epoch without reading future metrics. This stage checks only
finiteness, accepted optimizer steps, audit validity, export validity, and a
positive complete spring field. If it passes, run exactly one 200-epoch fit and
seal the checkpoint, audit, trajectory, spring field, and manifest before
opening the already-released future metrics.

A 10% improvement in both future Chamfer and track error over exact released
PhysTwin is the competence gate for a separately frozen five-case source panel.
The `8/15 mm` published MatPhys point is reported only as a headroom diagnostic:
one already-open case cannot establish a cohort-level state-of-the-art result.

Prob4D is unused and MolmoMotion remains at zero weight. No artifact from this
experiment may modify the frozen Causal4D acquisition candidate or authorize
fresh target access.

## Stage-0 import amendment

The first one-epoch mechanical attempt stopped while importing pinned MatPhys,
before model or optimizer construction, because Warp 1.15 no longer exposes the
private `warp._src.utils.warn` helper MatPhys wraps. It produced no checkpoint,
export, or scientific metric. The exact invocation and failure log are bound by
`configs/sota/matphys_causal_absolute_part_competence_v1_amendment.json`.

Version 1.1 permits exactly one new one-epoch run in `stage0-v1.1` after adding
the signature-compatible warning adapter already exercised by the all-frame
control. The adapter only restores warning dispatch, including MatPhys's
`once` argument; it does not alter data, simulation, model, optimization, or
any gate. Stage 1 remains unauthorized until this retry passes the original
mechanical gate.

The authorized retry passed. Its compact gate record is
`results/sota/matphys_causal_absolute_part_stage0_v1_1/stage0_gate.json`.
Training used 16 frames ending at frame 33, accepted all 33 optimizer steps,
and produced a finite checkpoint. The validated export contains 54,989
positive springs and five distinct part geometric means. No future metric was
opened. This mechanical result authorizes the single registered 200-epoch
competence run; it is not performance evidence.

The competence run is complete and failed its frozen gate. See
`docs/matphys_causal_absolute_part_competence_v1_result.md`. The causal
candidate regressed from `15.451/24.304 mm` to `31.057/34.789 mm` future
CD/track error, and its nominal five-part field collapsed to an effectively
uniform stiffness. The direct absolute-prefix family is closed without tuning;
the five-case panel is not authorized.

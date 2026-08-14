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

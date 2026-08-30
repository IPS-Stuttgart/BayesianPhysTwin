# DLO-Lab Separation Headroom Development Screen v1

This bounded public-simulator development screen asks whether the untouched
DLO-Lab two-rope separation task has enough state-dependent action value to
justify a later guarded-belief transfer study. It does not alter any frozen
DEFORM, wrapping, slingshot, wiring, support, coiling, target, or held-v8
evidence.

## Fixed Query

The native environment is DLO-Lab's released
`Train_Env_Separation` at revision
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`. The two released rope geometries,
robot models, Pink controllers, contacts, time step, constraints, grasp indices,
and symmetric nearest-point-distance reward remain native. The only world
variation is a rigid in-plane rotation of both released ropes about their shared
centroid before the native reset attaches the two grippers.

Nine development rotations span -35 to +35 degrees. The action bank contains a
shared two-macro 30 mm lift, a hold continuation, and eight symmetric 120 mm
pulls with directions fixed from -35 to +35 degrees. Two exact duplicates test
native determinism. Actions and worlds are specified from public geometry before
any native result.

## Qualification and Gate

Each world executes all actions in one CPU/float64 batch. Every world must pass
finite-state, exact rotation, shared-prefix, duplicate, segment-length, height,
attachment, and independently reconstructed native-reward checks. All nine
worlds are sealed before value analysis.

The action-value gate requires:

- the best fixed action to beat prefix hold by at least 20 mm of native reward;
- oracle headroom over that best fixed action, after a 1 mm numerical margin,
  to be at least 10 mm;
- at least three distinct oracle actions;
- at least four worlds with at least 10 mm oracle gain over the best fixed
  action.

These thresholds ask for contribution-sized decision value, not merely a
different argmax. A pass does not authorize source transfer automatically; it
only justifies a new, separately frozen off-grid source protocol. A failure is
terminal for this exact task/world/action design and will be entered into the
staged competence atlas.

## Boundaries

This is development evidence from a public procedural simulator. It is not an
official DLO-Lab benchmark result, a real-robot certificate, a calibrated sensor
model, or a backend-wide competence claim. It uses no new recordings, GPU,
protected target, held-v8 artifact, DLO4/DLO5 data, or official DLO3 evaluation.
The registered root is write-once, there is no retry, and exact fallback remains
the unchanged best fixed action.

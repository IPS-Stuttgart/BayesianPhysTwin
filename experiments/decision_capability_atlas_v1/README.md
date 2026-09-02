# Decision capability atlas mechanism study

This deterministic study instantiates the affine-task atlas on four supported
physical hypotheses, two registered quotient classes, and three actions. The
task varies a target displacement and a physical-risk weight. The complete
physical state remains ambiguous within both quotient classes.

The exact zero-regret regions for `pull_left`, `hold`, and `pull_right` are
computed as continuous polygons by enumerating the exact classwise-support
half-spaces. The uncovered region returns fallback. A canonical point belief
always selects one action, including inside that uncovered region, and therefore
illustrates unsupported decisiveness rather than a guarantee.

The extended study also treats the task objective as uncertain. For a fixed
axis-aligned objective box with half-width `(0.1, 0.2)`, exact support-function
erosion constructs the region of centers whose *entire* task box admits the same
action. On the valid center domain, nominal capability covers 83.18%, whereas
objective-robust capability covers 66.33%; the robust fallback region is 33.67%.
The task `(-0.6, 0.1)` is nominally certified for `pull_left`, but the box with
half-width `(0.04, 0.05)` has no certified action. This is the intended
strictness result: objective uncertainty can invalidate a nominal decision even
when the physical quotient is unchanged.

Reproduce with:

```bash
python experiments/decision_capability_atlas_v1/run.py \
  --output build/decision-capability-atlas/result.json
```

The result is controlled mechanism evidence. It does not validate a real
provider, quotient, task family, task-uncertainty set, physical-model
misspecification, or safety claim.

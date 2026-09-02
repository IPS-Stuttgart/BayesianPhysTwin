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

Reproduce with:

```bash
python experiments/decision_capability_atlas_v1/run.py \
  --output build/decision-capability-atlas/result.json
```

The result is controlled mechanism evidence. It does not validate a real
provider, quotient, task family, or physical safety claim.

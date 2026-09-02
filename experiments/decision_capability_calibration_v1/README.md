# Controlled finite-group calibration of a capability atlas

This deterministic mechanism study augments the controlled two-parameter
pull-left/hold/pull-right capability atlas with a finite-group statistical
correction.

The study deliberately separates three erosions of task-space capability:

1. the original exact atlas under registered latent-state ambiguity;
2. an inward half-space shift from one split-conformal order statistic over 19
   synthetic calibration groups;
3. the combination of that shift with an uncertain task-objective box.

Each synthetic group score is recomputed as an exact continuous-task maximum of
realized pairwise loss-gap undercoverage over the complete registered task box.
The maximization uses the vertex representation of an equivalent linear
program, not a plotted grid. The plotted/evaluated grid is used only to check
that action regions do not have positive-area overlap in this controlled case.

Reproduce with:

```bash
python experiments/decision_capability_calibration_v1/run.py \
  --output build/decision-capability-calibration/result.json
```

The result is controlled mechanism evidence. The synthetic groups do not
constitute real-data calibration, unseen-object validation, pointwise safety,
or deployment evidence. A real experiment must freeze the model, quotient,
action set, affine task family, task domain, group split, and score before
opening one outcome bundle per independent calibration object or trajectory.

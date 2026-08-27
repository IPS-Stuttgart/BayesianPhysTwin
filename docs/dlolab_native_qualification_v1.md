# Public-Simulator Decision Route: Native Qualification

The DEFT topology source gate failed. Do not tune or extend that frozen state
correction on its opened outcomes. A distinct possible contribution is to test
whether a physical belief improves action choice under uncertainty, with actual
branch-at-prefix simulator futures available for every action. This does not
require new physical recordings. It does require an honest simulation-only claim.

## First Gate

Before any method comparison, qualify the public DLO-Lab native rod reset and
control interface. Pin upstream commit
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`. Use an isolated CPU, float64 runtime,
one Torch thread, a procedural 16-node rod, and no external dataset, checkpoint,
benchmark asset, render, or policy. All physical configuration is fixed in
`DloLabConfig`; its canonical identity and every upstream Python source hash are
recorded. The native solver is not modified or replaced by a hand-built simulator.

After 25 native steps with a fixed two-point root, save all 15 native rod memory
fields. Compare a 40-step cubic 20 mm lateral clamp action with its exact replay
from the snapshot and with a monolithic run from the initial full state. Restore
the same snapshot and run the opposite action. Only the two prescribed clamp
positions enter the control API; the free-node positions and velocities are
preserved before each native step. Padding and outcome-based state injection do
not occur. A coupled active solver is rejected by this initial adapter.

Required checks: all trajectories finite; bit-identical replay and monolithic
suffix positions and all memory fields; clamp error at most 1e-10 m; opposite
actions change a free point by more than 1e-5 m; maximum segment-length error at
most 10%; captured snapshots unchanged. The last two are basic sanity checks,
not material validation or numerical-convergence evidence. Physical parameters
are simulator settings, not inferred real material properties.

The separate `--world-bank` qualification repeats these checks with three
parallel environments: bending settings [0.5, 1, 2] times nominal and initial
lateral velocities [-0.15, 0, 0.15] m/s in a smooth free-node mode. Both root
velocities remain exactly zero. The runtime reads back the native material and
velocity arrays and requires exact equality with their registered values before
the prefix. Every snapshot binds the complete material/velocity bank identity;
it cannot be replayed in a different bank. This tests the interface used by the
decision study without generating or comparing any decision-study outcomes.

The source must be clean and committed before native qualification. An exclusive
output directory and attempt record precede runtime initialization; every failure
is retained. Runtime or source repairs require a separate recorded revision and
output root. The existing DEFORM environment, forecasts, and point result remain
untouched. Qualification is not a method outcome and does not automatically
authorize a decision study.

## Candidate Scientific Test, Not Yet Authorized

If qualification passes, freeze a separate decision experiment before producing
its outcomes. It should compare a nominal point-model controller, a simple
measurement-conditioned controller, and a bias-aware physical belief under equal
observations, action banks, and computation accounting. Score action regret,
task error, safety violations, and abstention against simulator branch outcomes;
include a full-state oracle only as a ceiling. Keep model uncertainty distinct
from shared observation bias, and test misspecification rather than only drawing
truth from the exact inference prior. Preserve exact fallback and retain failures.

This is a prospective source experiment, not a claim that Bayesian deformable
control is new. JIGGLE already combines a differentiable soft-body simulator,
EKF estimation, information-seeking control, and safety costs. DLO-Lab already
provides deformable-manipulation tasks and control/optimization baselines. A
useful contribution must earn a measured advantage over relevant controls; a
new adapter or standard filtering formula alone is insufficient.

No DEFORM DLO4/DLO5 or official DLO3 evaluation, held-v8, protected Deform360
target, physical Causal4D data, or prior failed-study target is accessed. Nothing
from this lane changes the frozen DEFORM implementation. Code and evidence remain
local/private until an explicit publication decision.

Primary sources:

- [DLO-Lab public code](https://github.com/UMass-Embodied-AGI/DLO-Lab).
- [JIGGLE, RSS 2024](https://www.roboticsproceedings.org/rss20/p007.html).

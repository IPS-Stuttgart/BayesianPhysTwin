# Native Support Task: Decision-Sensitivity Qualification

The frozen contact-free DLO-Lab experiment did not establish Bayesian decision
value: the nominal model already selected the oracle action for every episode.
Do not retune that task or its gates. This separate source-only qualification
adds an actual native unilateral support and a longer support/lift interaction.
It asks whether hidden physical conditions can change the best action for the
same known goal. No Bayesian method result is claimed at this stage.

## Native Scene

Pin DLO-Lab `c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`. Use the native ROD-ROD
capsule contact implementation documented in its `examples/quick_example.py`
and used by `experiments/envs/env_wiring_post.py`. No contact solver is written
or patched here. No robot mesh, external task asset, or new recording is used.
This is not the official DLO-Lab robot benchmark or a reproduction of its scores.

The object is a 25-node, 0.6 m rod, initially horizontal at z=0.6 m, with the
first two nodes prescribed. All settings inherit `DloLabConfig` except its node
count. Gravity is (0,0,-9.81) and the native floor is explicitly at z=-5 m, away
from the workspace. A seven-node fixed native capsule, radius 20 mm and total
length 0.36 m, lies along y at z=0.48 m. Source worlds cross bending settings
[20000,100000,500000] with support x positions [0.20,0.40] m, giving six fixed
worlds. Friction uses the pinned native material defaults. Material settings are
simulator parameters, not calibrated real-world material properties.

The runtime reads back all material and support coordinates before stepping.
It inherits the already qualified state/control methods without changing them:
only the two root positions are prescribed; free-node state is preserved before
each step. Snapshots include all 15 native rod memory fields for both object and
fixed support, and bind the full world/configuration identity.

## Frozen Qualification

Hold the root for 125 steps (0.25 s). From that complete state generate twelve
250-step (0.5 s) candidate lift/side motions, each with a cubic displacement
ramp. Hold is action zero; other offsets are the lexicographic nonzero pairs
y in [-0.06,0,0.06] m and z in [0,0.05,0.10,0.15] m. The x offset is zero.
All actions, all six worlds, and all nine goals remain in the denominator.

Before task analysis, require finite trajectories, fixed-support byte identity,
root error <=1e-10 m, maximum segment-length relative error <=10%, and geometric
support penetration <=3 mm. At least two worlds must approach the support
surface within 2 mm. Read-only finite-segment geometry checks contact distance;
all collision forces and trajectories are computed by native DLO-Lab.

Actions 0 and 11 must replay bit-identically in positions and all 15 memory
fields; action 11 must also match a monolithic prefix-plus-action run. Both
snapshots must remain unchanged. A failure is retained and blocks task analysis.
These are interface/geometry checks, not a proof of converged physical accuracy.

Goals are all nine combinations x in [0.35,0.50,0.60] m and z in
[0.35,0.45,0.55] m, with y=0. Loss is the mean squared tip-goal distance over
the last 50 action frames plus 0.02 times squared root displacement, in m^2.
At each goal compare the best action chosen without knowing the world,
min_a mean_w L(w,a), with a world-informed oracle, mean_w min_a L(w,a).
This comparator is optimistic for the world-blind policy; an arbitrary weak
nominal action is not enough to pass.

A goal passes only if it has at least two oracle action indices, a world-
information gap at least 10% of mean hold loss, and an absolute gap at least
25 mm^2. At least three of the nine goals must pass. The fixed goal/world grid
is a source-design qualification, not independent statistical replication.
There are no significance tests, population-calibration claims, or outcomes
selected for later confirmation. All nine goals must remain in any subsequent
method design, including the uninformative ones.

## Custody And Next Stage

Freeze clean source, source-byte hashes, native runtime, upstream Python files,
and output root before any native run. Generate and seal every trajectory and
native check before computing task losses. A separate loop recomputes all
648 goal/world/action losses with explicit units and time averaging. Retain
failures without replacements, retry, or source-gate relaxation. The previous
contact-free failure and successful DEFORM remain immutable.

Passing only shows that a decision problem has nontrivial information value.
It does not show that sparse observations reveal that information or that a
Bayesian method uses it effectively. A later, separately frozen method test
would need matched nominal, MAP, posterior-mean, calibrated mean-only, and
joint-regret controls; shared sensor bias; source-only calibration; fresh draws;
and prediction sealing before evaluation futures. Active probing, if studied,
must compete with simple and information-gain probes at equal action budgets.

Bayesian physical inference and active boundary sensing already exist. Relevant
primary work includes [A Bayesian Treatment of Real-to-Sim for Deformable Object
Manipulation](https://arxiv.org/abs/2112.05068) and
[JIGGLE](https://www.roboticsproceedings.org/rss20/p007.html).
The paper would need measured value beyond those standard ingredients, not a
claim that a probabilistic simulator or active probe is new.

All work stays on an isolated local/private branch. No DEFORM DLO4/DLO5,
official DLO3 evaluation, held-v8, reserved Deform360 target, real Causal4D
record, GPU job, or new physical recording is used. No push, merge, or downstream
evaluation is automatically authorized by the qualification.

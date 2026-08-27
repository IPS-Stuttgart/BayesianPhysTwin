# Coupled Action Regret: Frozen Public-Simulator Source Test

## Question And Claim Boundary

Does preserving a shared physical world across alternative actions improve a
calibrated baseline-relative decision over identical action marginals without
that coupling, and over a cheaper calibrated mean-only guard? This is a
single-decision, branch-at-prefix simulation study. It is not closed-loop robot
control, real-world safety, official DLO-Lab benchmark parity, SOTA, or fresh
physical validation. No new recordings are needed. Successful DEFORM artifacts
and all previous negative experiments remain unchanged.

Gaussian nuisance marginalization, weighted posterior quantiles, and
split-conformal order statistics are established tools, not claimed inventions.
The proposed evidence is the matched decision comparison and its exact fallback,
not a new theorem or a new physics backend. Related prior work includes
[JIGGLE](https://www.roboticsproceedings.org/rss20/p007.html),
[Conformal Risk Control](https://arxiv.org/abs/2208.02814), and
[conformal decision theory](https://conformal-decision.github.io/).

## Native World And Observations

Use unmodified [DLO-Lab](https://github.com/UMass-Embodied-AGI/DLO-Lab) commit
`c5026a9416b03c6bc5186eba13cd4ffd4c0e7796`, CPU float64, one Torch thread, and a
procedural 16-node rod with two prescribed root nodes. `DloLabConfig` freezes
every simulator setting. The prefix has 25 steps (50 ms); each action has 40
steps (80 ms), with 10 solver substeps per step. This short horizon and the
single procedural topology explicitly limit external validity.

The native three-world qualification must pass before the study lock. All 15
native memory fields are captured; exact replay and monolithic continuation
must be bit-identical. Material and initial-velocity arrays are read back from
the native solver and hash-bound to the snapshot. Each candidate action starts
from the same complete state, not only a copied position array. All trajectories
must be finite, root tracking error at most 1e-10 m, and maximum relative segment
length error at most 10%. These are implementation checks, not material-property
or numerical-convergence validation.

The inference bank contains 15 worlds: three bending settings [0.5, 1, 2] times
nominal, crossed with five initial lateral velocities [-0.30, -0.15, 0, 0.15,
0.30] m/s in a smooth root-zero mode. Its prior is uniform. The nominal point
model is the center world. True worlds draw bending log-uniformly from [0.5, 2]
times nominal and lateral velocity uniformly from [-0.25, 0.25] m/s. They are
not restricted to the particle grid, but use the same solver family. This is
parameter/state interpolation, not a model-class misspecification test.

Each episode supplies 12 known-identity 3D positions: nodes [3, 6, 10, 15] at
prefix indices [4, 14, 24]. Observation noise is 3 mm iid Gaussian plus one
episode-shared xyz Gaussian offset of 12 mm standard deviation. The shared-bias
likelihood marginalizes that offset exactly; the iid control sets its variance
to zero. Sensor scales are known generative settings, not estimated calibration
on real cameras. There is no residual-based prior reliability, clipping, or
second use of an innovation. Known 3D identities are assumed, not automatically
tracked. Observing the tip in the prefix and scoring its future task error is
intentional; this is not a hidden-identity tracking study.

## Actions, Loss, And Arms

There are nine actions: unchanged hold, followed by lexicographic nonzero
(y,z) pairs in {-25, 0, 25} mm squared. The root translates through a cubic
zero-endpoint-velocity ramp; x displacement is zero. Goals are independent of
the noisy observation and are sampled around the initial tip: x +/-5 mm,
y +/-20 mm, z in [-50, -20] mm. The method receives the task goal, but not true
bending, true initial velocity, full true prefix, or future outcomes.

Task loss in m^2 is squared terminal tip-to-goal distance plus 0.02 times
squared root displacement. Harm means strictly larger task loss than hold
(1e-12 m^2 numerical tolerance). It is not collision, force, or hardware safety.

1. Exact hold.
2. Nominal point-model loss minimization.
3. Posterior expected loss with an iid-only sensor likelihood.
4. Posterior expected loss with the shared-bias likelihood.
5. Calibrated mean-regret guard.
6. Calibrated independent-marginal regret guard.
7. Calibrated coupled-regret guard (primary).

The last three share the same bias-aware posterior and expected action losses.
The mean arm starts with expected regret. The coupled arm takes the 90th
weighted percentile of L(p,a)-L(p,hold) using the same world p. The independent
arm uses all p,q pairs with product weights: L(p,a)-L(q,hold). Thus its action
marginals are exactly matched; only their coupling changes. All three undergo
the same separate calibration procedure. Hold regret is exactly zero even for
the independent arm; fallback returns the original hold command object and
preserves its dtype, shape, and C-order bytes.

The oracle uses all realized action losses only after decision sealing and is
reported as a ceiling, never supplied to a policy. Nominal inference needs one
world, posterior methods 15. Each has nine actions. Independent quantiles use
225 pairs versus 15 paired differences per non-hold action. The source runner
records stage wall times; it does not claim matched inference cost or latency.

## Calibration, Decision Seal, And Scoring

The canonical `protocol()` values are frozen in a clean-commit lock before any
method outcome. Calibration uses 39 independent synthetic world/noise/goal
draws (seed 260829); evaluation uses 64 different draws (seed 260830). One
calibration score per episode is the maximum true-minus-predicted regret over
all eight non-hold actions. The nominal 90% rank is ceil(40*0.9)=36. Its offset
is clamped below at zero, conservatively. A candidate is admitted only if its
upper regret plus offset is strictly negative. Among admitted actions and hold,
choose the smallest posterior expected loss, breaking ties by action index.

The bound concerns all finite-bank actions within one new exchangeable episode,
marginal over calibration and new-episode sampling. It is not conditional
coverage given acceptance, simultaneous coverage of every future episode,
physical safety, or a guarantee under arbitrary distribution shift. Episodes
share a simulator family and procedural object; they are not 64 physical-object
replicates. The exact numerical harm bound uses all 64 episodes, not only
accepted actions.

The executable stages are explicit and write-once:

```text
native qualification -> freeze -> model bank -> calibration outcomes
 -> 64 prefix-only decision seals -> verify full barrier/replay
 -> evaluation action futures -> fixed scorer -> independent arithmetic replay
```

No future generation occurs in the decision stage. Model-bank predictions are
not evaluation outcomes. The scorer validates all dependencies, recomputes
prefix-only inference/decisions, and verifies byte-identical native prefix
replay and command hashes before generating evaluation futures. Any failure
closes the attempt, retains the full denominator, and authorizes no retry,
replacement, imputation, or target. A missing stage cannot be bypassed with a
pass boolean. Complete evidence is bound by file/array/content SHA-256 values,
exact implementation revision, upstream Python source bytes, runtime versions,
and the local software-rendering library.

Equal-episode aggregation uses paired 10,000-replicate bootstrap intervals,
seed 260831, for mean loss gains. The primary gate requires all of:

- All 64 ordinary evaluation episodes, no replacements.
- At least 16 non-hold decisions and at least 10% mean task-loss gain over hold.
- One-sided 95% Clopper-Pearson task-harm upper bound at most 10%.
- At least 85% observed simultaneous-action bound coverage.
- At least three distinct oracle action indices.
- At least 1.1 times each calibrated control's mean gain, with negative control
  gains conservatively replaced by zero for this comparison.
- Positive paired mean-gain 95% lower bounds against hold and both calibrated
  controls (mean-only and independent-marginal).

Only this joint arm has a promotion gate. All other results are descriptive
controls, not a post-outcome search for a replacement winner. A failed gate is
archived without retuning this task, seeds, noise, horizon, actions, or thresholds.
Even a pass would justify only broader separately registered simulation and
public-data studies, not an automatic real-world or target claim.

Before outcomes, unit tests exercise positive/placebo decisions, full calibrated
controls, Gaussian covariance equality, exact fallback, missing-stage rejection,
one-attempt directories, and a complete fake-runtime staged dry run. A separate
scorer recomputes the Gaussian likelihood with a dense inverse, weighted
quantiles with an independent implementation, calibration offsets, actions,
losses, bootstrap intervals, binomial bounds, and gates. That is independent
arithmetic, not an independent human reviewer.

No DEFORM DLO4/DLO5 or official DLO3 evaluation, held-v8, protected Deform360
target, or physical Causal4D record is accessed. This branch stays local/private;
no automatic push, merge, or publication is authorized.

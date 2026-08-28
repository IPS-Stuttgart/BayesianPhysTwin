# Native Slingshot Grip-Force Screen: Small Source Gain, Gate Failed

The frozen source implementation was
`06a54b43bbd92a16e411c5c60988b93e456165d3`. All three CPU worlds completed
with no technical failures, retries, replacements, or denominator changes.
This is development evidence under a finite simulated world/observation model,
not an independent control evaluation or a real-sensor experiment.

## Matched Result

| Arm | Expected native reward | Gain over new best fixed action |
|---|---:|---:|
| Prior -3 N fallback | 7.001163165 | -0.012763341 |
| Best fixed force-plus-motion choice | 7.013926506 | 0 |
| Bias-aware posterior mean | **7.018174008** | **0.004247502** |
| Bias-aware MAP | 7.013971472 | 0.000044966 |
| Posterior ignoring shared bias | 7.009888309 | -0.004038197 |
| Perfect-information oracle | 7.024264971 | 0.010338465 |

The posterior mean gains 0.004202535 over MAP and 0.008285699 over ignored-bias
inference. Its Monte Carlo standard error is 0.000063603. These are numerical
integration errors over the registered sensing model, not experimental
confidence intervals or uncertainty across independent physical objects.

The result **fails the frozen advancement gate**. Gain over the strongest
fixed action is below 0.005 and is only about 3.73% of that action's excess
reward above zero control, below the required 10%. The MAP and ignored-bias
comparison gates pass; they do not override the two failed gates.

There is also a useful upper limit: the oracle gain is only 0.010338465,
below the required 0.011392641 for the 10% criterion. Even perfect inference
cannot make this bank pass that criterion. The low-coupling world ties zero
control for every candidate and remains in the three-world denominator.
No threshold is relaxed and no model or controller is promoted.

## Native and Causal Checks

All native QA, prefix replay, and fallback checks pass. The maximum fallback
position difference from the prior native contact run is 8.9148e-8 m, below
1 micrometre; fallback rewards match exactly. The maximum entire-prefix
difference is 2.2063e-13 m. Both fallback slots are retained.

The actual solver records verify all 93 finger-force commands and three
unchanged releases. Forces are identical before step 300; the final allowed
observation is frame 299. The native +/-30 N limits, release at step 700,
controller, arm motions, contact worlds, and reward remain registered.
The separate force-command record, not the upstream nominal finger entries
in `joint_targets`, is authoritative for replay.

The contact coefficient is the pinned simulator's tangential velocity
coupling coefficient, not a measured Coulomb friction parameter. The
posterior receives noisy causal positions, not the true coefficient or
contact/force state. This screen tests one particular finite prior and
sensing model, not calibrated camera reliability or real robotic safety.

## Verification and Custody

Before native execution, 202 relevant tests, Ruff, focused MyPy, and source
preflight passed. A separate arithmetic implementation (`verify_grip.py` in
the local evidence archive) rehashed the arrays and source blobs, recomputed
all 24 native rewards, checked force schedules and causal/fallback replay,
and reproduced the source integration with a dense covariance inverse.
This is a second implementation by the same agent, not independent human
review. It did not execute the simulator again.

- Lock ID: `5a6e51e361289121f3b07ac6a257ce5fc522689df63d288172c0e60623b1935b`.
- Lock file SHA-256: `bf5c100bdc6d1f5508c0dc75a33ea61c7f3f3fff837325b8b036e62df6e39844`.
- Result ID: `b97ff64f95db0cb76aecde90dc1eec504e9e5e8422fd95c280c0caac909651c2`.
- Result file SHA-256: `00b09c2fd7a5ae8a27c2b5f0aca292aac397df812a9db1614688c95383207f57`.

The write-once native root remains
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/grip-recovery-source-v1`.
Compact artifacts accompany this note under
`results/source/dlolab_slingshot_grip_recovery_v1/`.
All results remain local/private. No GPU, new recording, robot, protected
target, held-v8, DLO4/DLO5, public push, or main merge was used. Existing
DEFORM results and every earlier failed study are unchanged.

## Interpretation

Adding a realizable grip-force action creates some value for uncertainty-aware
decision making that the Cartesian-only contact screen lacked. Its magnitude
is insufficient under the locked rule. The next design would need genuinely
greater decision-relevant control authority, not only a sharper posterior,
a weaker comparator, a removed difficult world, or a relaxed gate.

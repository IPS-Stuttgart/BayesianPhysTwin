# Native Slingshot: Post-Prefix Grip-Force Source Screen

The prior contact screen found observable coupling differences but zero oracle
headroom in its Cartesian-only action bank. This new source-development screen
adds one actionable variable: native finger force after the causal prefix. It
does not change that negative result, the material/probing studies, or DEFORM.

Use the same three native gripper-coupling worlds (0.3, 0.6, 0.9), nominal
material and placement, source observation model, and task reward. Run one
fresh eight-environment CPU batch per world, in order 0.9, 0.3, 0.6. The seven
fixed choices and the fallback duplicate are:

| Index | Prior Cartesian action | Post-prefix force per finger (N) |
|---:|---:|---:|
| 0 | 6 | -6 |
| 1 | 6 | -12 |
| 2 | 6 | -24 |
| 3 | 5 | -6 |
| 4 | 5 | -12 |
| 5 (fallback) | 6 | -3 |
| 6 | 5 | -24 |
| 7 (duplicate fallback) | 6 | -3 |

Every Cartesian sequence is copied without modification from the already-open
source bank. Native reset is still -1 N; frames 100-299 still use -3 N. The
first changed force command occurs before native step 300, after the final
permitted observation at frame 299. There are exactly 31 force commands: reset
at step 0 and native microstep commands at 100,120,...,680. Each command is
checked against the solver's actual control force. The original +/-30 N
actuator limits must match exactly. Release remains native position control
at 0.08 m at step 700, followed by 200 native steps. Arm control, physics,
contact coefficients, reward, horizon, and observation budget are unchanged.
The separate force-command record is authoritative for fingers: the upstream
`joint_targets` archive retains its nominal finger placeholders and must not
be mistaken for a complete replay input for this new policy.

Force changes are a new control policy, not a claim of modified simulator
mechanics or measured friction. The native coupling law remains the pinned
tangential velocity law, not calibrated Coulomb friction. No force or contact
ground truth is fed into the posterior. Observations are the original twelve
causal 3D positions with 2 mm independent noise and 5 mm shared xyz bias.

Each batch must pass the existing native QA plus replay both fallback slots
against the prior contact world's action 6 within 1 micrometre with exact
reward. Its entire prefix must also replay within 1 micrometre. A failure in
the first nominal batch blocks later batches; every failure is retained,
without retry or replacement. The locked 3-world denominator is not reduced.

The same source-only finite-model calculation is reused: equal world prior,
8192 Gaussian draws, seed 260909, posterior mean versus MAP, ignored shared
bias, best fixed action, and perfect information. Reusing these source worlds
and noise draws is deliberate development, not independent confirmation.
Advancement still requires at least 0.005 gain over the best fixed action,
at least 10% of its excess reward above zero, at least 0.002 gain over MAP,
and no loss to ignored-bias inference. The best fixed force-plus-motion choice,
not only the older -3 N action, is the comparator. Numerical integration
standard errors are not experimental confidence intervals.

The write-once output is
`/home/fpfaff/source-only/dlolab-benchmark-source-v1/grip-recovery-source-v1`.
The lock binds clean Git revision, code/tests/doc hashes, contact source lock,
three reference seals, exact Cartesian and force schedules, and qualified
native runtime/assets. Task claims precede initialization; workers rederive
earlier QA before another batch. No GPU, new recording, robot, protected data,
calibration/evaluation world, public push, or main merge is authorized. Even a
passing screen would require a new prospective control evaluation.

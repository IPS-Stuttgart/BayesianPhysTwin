# Native Slingshot Decision Value: New Source Screen

This is a post-development source diagnostic, not a preregistered independent
confirmation. The original strict memory and 1 micrometre replay gates remain
failed. Their observed numerical jitter is explicitly used to choose a new,
task-level diagnostic envelope before any of the new world/action outcomes:
0.5 mm for position traces and 0.001 for native cumulative reward. These are
engineering envelopes, not statistical coverage guarantees. No hidden-state
snapshot is reused; every world starts in a fresh native process.

Freeze the nominal source world plus eight corners: sphere and cube lateral
offset +/-20 mm, bending E at 0.5/2 times nominal, and stretching K at 0.5/2
times nominal. Use the native rod parameter setters and the same rigid-position
setter as public domain randomization. All other physics, robot, release,
geometry, and native reward are unchanged; realized parameters are recorded.
The nominal point controller is the unchanged CMA-ES source candidate 29.

There are seven distinct commands, all sharing its complete first stage:
incumbent; second-stage x +/-30 mm; later-stage y multiplied by 0.8/1.2; and
final-stage yaw +/-0.2 rad. Official component and per-stage translation bounds
are applied to modified suffixes only. A duplicate incumbent fills the eighth
batch slot and measures numerical variation. Thus the fixed design has nine
worlds, 63 distinct action/world trajectories, and nine diagnostic duplicates.
The causal branch boundary is after native frame 299. The first 300 observed
frames must agree across actions within the new numerical envelope.

All 72 trajectories are generated and sealed before the aggregate oracle
screen. There are no retries, replacements, additional actions, or world
selection from outcomes. A failed native world terminates with a retained
technical failure; failed numerical QA prevents passing the screen.

The gate compares a world-specific oracle against the strongest single action
chosen retrospectively for this source bank, not against zero control. It
requires at least 0.01 native reward gain after subtracting twice the 0.001
numerical envelope, at least 10% of the best blind action's excess reward over
zero (denominator floored at 0.01), at least three worlds with raw oracle gain
above 0.01, and at least two different optimal actions. The best blind action
must itself exceed zero reward by 0.01. This is an upper-bound headroom test,
not evidence that a Bayesian controller attains the oracle.

A positive result permits designing a separately frozen matched belief test;
it does not itself fit or promote one. Such a test must keep this incumbent,
point-estimate and predictive-mean controls, joint versus independent regret
coupling, source-only calibration, and exact command fallback. The uncertainty
benefit must not be credited to a better open-loop optimizer or extra sensing.
No target objects, held-v8, protected DEFORM data, GPU work, new recordings,
public pushes, or main-branch changes are involved.

# Slingshot Source Mechanism And Readout Audit

This new diagnostic uses the already-open, fixed CMA-ES candidate 29. Its
original all-state replay gate remains failed. We do not restart a hidden
state, revise that result, retune the controller, or claim Bayesian improvement.
Fresh native rollouts instead test two specific questions: is its task readout
reproducible, and is rod-to-projectile coupling necessary for its observed task
response? Final internal acceleration/director agreement is not substituted
for either question and remains separately reported.

Freeze three fresh CPU processes: two unchanged native repeats and one arm
with only the sphere collision geometry's `needs_coup` set false before scene
build. The pinned native field means contact with non-rigid entities; rigid
collision `contype`/`conaffinity`, sphere mass/geometry, all other coupling
flags, rod material, robot commands, native reward, and release are unchanged.
The public `construct_extra_cameras` pre-build hook performs the intervention;
compiled coupling flags are checked after build. No upstream file is edited.

The unchanged repeats must match all four position traces to the saved isolated
source reference within 1 micrometre and native reward within 1e-5, and each
must retain at least 10 mm of sphere and cube forward progress. The disabled
arm must reduce both sphere and cube progress by at least 80%, relative to the
smaller unchanged-repeat progress. All three arms must be complete and finite;
failures remain failures without retries or replacements. Readout replay does
not authorize hidden-state snapshot restart. All arrays are sealed before
the final comparison, and the failed parent result is exact-hash bound.

A pass supports rod-to-projectile causal dependence for this source controller,
not a complete energy accounting, real-world validity, learned uncertainty,
published-budget parity, or SOTA. A separate prospective source experiment is
still required to show that uncertain worlds change useful actions and that a
belief-aware decision beats strong matched point controls. Public simulator
assets only; no new recordings, GPUs, targets, held-v8, or protected DEFORM data.

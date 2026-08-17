# DEFORM DLO2 local-residual all-train v7

This stage is a target-blind refit of the DLO2 method that passed the frozen v6
source gate. It trains the official DEFORM physical model from scratch on all 56
official DLO2 training trajectories for the already fixed 6,400-update budget,
then fits the causal local-residual model on the same 56 trajectories.

The transferred arm is fixed at ridge `1.0` and shrinkage `0.25`. Its query uses
only two observed states, the known future clamped-node action, and the physical
baseline rollout. Validation, source, and target reselection are all forbidden.
Prob4D is unused. The exact physical checkpoint remains the fallback.

The runner installs read guards for both official DEFORM evaluation partitions,
validates the passed v6 result and all 56 training identities, and writes a
preflight artifact before training. A smoke run performs one update and cannot
authorize official evaluation. Only a complete registered run can emit the
checkpoint, pickle-free local-residual model, final-method artifact, and
authorization for the separate one-shot evaluator.

Protocol: `configs/sota/deform_dlo2_local_residual_alltrain_v7.json`.

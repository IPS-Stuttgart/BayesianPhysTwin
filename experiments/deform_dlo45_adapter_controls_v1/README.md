# DEFORM DLO4/DLO5 post-hoc adapter controls

This experiment evaluates predeclared controls for the completed DLO4/DLO5
BayesianPhysTwin result from workflow run `33361441865`.

The target outcomes were already open before this study was created. The result
is therefore **retrospective post-open evidence**, not fresh confirmation. The
workflow may fail on parent-identity, data-boundary, or numerical-parity errors;
it does not fail because a scientific control wins or loses.

## Questions

1. Does the frozen local adapter outperform an equal-wall-time continuation of
   the DEFORM optimizer?
2. Is its gain explained by a global, per-node, or time-indexed mean residual?
3. Does action-frame alignment, explicit action information, or the simulated
   baseline trajectory contribute to the result?
4. How does target error vary with deterministic nested subsets of the 56 public
   training trajectories?

## Frozen controls

- `primary_full_adapter`: the reported action-local Bayesian ridge adapter;
- `compute_matched_physical`: the retained one-update DEFORM continuation;
- `global_bias`, `node_bias`, and `time_node_mean`: source-only trivial residuals
  with seven-fold source-only shrinkage selection;
- `global_frame_linear`: the full linear adapter without rotational action-frame
  alignment;
- `no_explicit_action_linear`: removes explicit prescribed-action and
  endpoint-relative features while retaining the action-conditioned simulator
  rollout;
- `initial_action_only_linear`: keeps initial state and prescribed action but
  removes simulated future geometry and dynamics features; and
- deterministic source-size fits at `1, 2, 4, 8, 16, 32, 56` trajectories.

Complete trajectories are the statistical units. DLO4 and DLO5 are summarized
separately and with an equal-DLO stratified bootstrap.

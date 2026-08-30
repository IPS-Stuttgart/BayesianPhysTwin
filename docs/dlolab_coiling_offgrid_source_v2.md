# DLO-Lab Coiling Off-Grid Source Screen v2

This source-only study asks whether a guarded posterior over public DLO-Lab
coiling worlds can improve action selection over the best fixed source action.
It is a new question on twelve off-grid state/material worlds. It does not
complete or retry the terminal v1.1 material-grid study.

The seven unique actions, exact duplicate, common prefix, observed nodes,
native reward, horizon, and correlated synthetic-noise model are unchanged
from v1.1. The worlds jointly vary bending stiffness, twisting stiffness, and a
registered initial rope translation inside the public simulator's documented
randomization bounds. Every source world is disjoint from the one opened v1.1
nominal world.

All twelve native worlds must seal and independently requalify before any value
analysis. Evaluation is leave-one-world-out. The training fold chooses its best
fixed action as baseline. A posterior-selected action is admitted only when its
expected gain is at least `0.003` and posterior probability of losing by more
than the `0.002` numerical margin is at most `0.05`; otherwise selection falls
back exactly to that fixed action.

The source gate requires nontrivial fixed-action and oracle headroom, multiple
oracle actions, positive guarded cross-fit gain, recurring admissions and
gains, and bounded observation-draw harm. Passing would authorize only a new,
separately locked prospective public-simulator protocol. It does not select
prospective worlds, open protected data, or authorize a target run. Failure is
terminal and no retry or post-outcome tuning is permitted.

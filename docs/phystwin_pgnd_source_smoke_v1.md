# PGND source-backbone competence smoke

## Question

Can the public Particle-Grid Neural Dynamics (PGND) plush checkpoint improve
the future of a released PhysTwin state without using future object
observations?

This is a single, already-open `single_lift_sloth` source experiment. It is a
competence test, not an independent comparison or a state-of-the-art claim.

## Why PGND

PGND is a distinct public learned-dynamics backbone. It predicts a dense
velocity field on a particle-grid representation and propagates particles
through its released Warp transition. Unlike the previously tested PGRD
residual, this test runs PGND's complete learned state transition rather than
adding a learned correction to PhysTwin.

The official repository exposes episode-disjoint training and evaluation
ranges for six categories, including plush objects. Its `--state_only` entry
point still imports the Gaussian renderer unconditionally, but state
prediction itself does not require rendering. The frozen adapter imports only
the material model and simulator.

## Frozen interface

The candidate receives:

- the unchanged PhysTwin state at the last 10 Hz frame fully inside the
  released training prefix;
- two earlier PhysTwin states and finite-difference velocities;
- the released future controller trajectory as a known action;
- the public PGND plush checkpoint.

The candidate does not receive future object points or manual tracks.

Controller trajectories contain 30 hand points rather than one robot gripper.
The frozen bridge selects, at each model step, the action point nearest the
unchanged PhysTwin trajectory. This is a physical-prior-supported contact
readout. It cannot use the candidate outcome or future observations.

PGND retains metric scale. Its released plush axis transform is fitted with a
translation from the current prefix state only. There is no fitted scale, yaw,
blend, cap, or trust parameter.

## Execution order

1. Extract a prediction carrier containing only the physical trajectory,
   controller actions, and released split.
2. Reject dirty adapter or PGND checkouts and bind both commits.
3. Run PGND twice and require bit-exact state trajectories.
4. Seal the prediction archive and its provenance.
5. Open future object points and manual tracks for evaluation.

The candidate and an equal-support PhysTwin comparator use the same
deterministic 1,000-particle subset. Sampled nodes preserve the released
surface/interior distinction, so interior particles are not silently counted
as surface support in Chamfer distance. The unchanged full PhysTwin trajectory
remains the primary comparator. Endpoint persistence uses the exact final
allowed prefix frame and is reported as a control.

## Gate

Advance to a wider opened-source panel only if PGND improves both aggregate
future Chamfer distance and aggregate future manual-track error by at least 2%
relative to unchanged full PhysTwin. Otherwise this raw replacement is closed.

No result from this smoke authorizes held-v8, PokeFlex, or any fresh target.

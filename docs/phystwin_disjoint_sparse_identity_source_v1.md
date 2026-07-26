# Disjoint sparse-identity source protocol

## Question

The opened PhysTwin-22 headroom audit reached 7.892 mm Chamfer distance and
13.429 mm manual-track error when released manual identities were observed
during the prefix. That number is not a fair state-of-the-art result because
the same identities also formed the future track metric.

This protocol asks the harder question:

> Does a small set of prefix material-identity observations improve future
> trajectories at identities that were never assimilated?

## Causal split

For each case, finite frame-zero manual identities are partitioned using only
their frame-zero geometry. Deterministic farthest-point sampling selects 1, 2,
or 4 observed identities. All remaining identities are hidden.

- Observed identities are available only before `train_end_frame`.
- Hidden identities are unavailable to fitting and prefix selection.
- Future track error is evaluated only on hidden identities.
- Assimilated identities never enter the future track metric.
- Every future frame must contain at least one valid hidden identity.

The four-identity arm is primary. One and two identities are declared
sensor-budget ablations. The physical baseline, dense released pseudo-track
channel, graph method, temporal selector, caps, and all numerical settings are
unchanged from the earlier headroom audit.

If the fixed sparse identity set has no valid observation in a case's prefix
validation interval, the selector renormalizes to the supported prefix Chamfer
term rather than assigning a zero track denominator. This causal support rule
was added after a technical abort and before any report or hidden score was
written.

## Gate

Only `causal_selected_dense_relative_cap_temporal` is analyzed. Other broad
runner candidates cannot affect the decision. The primary arm must:

1. improve equal-case Chamfer distance by at least 5%;
2. improve disjoint hidden-identity track error by at least 5%;
3. improve both metrics in at least 16 of 22 cases; and
4. have no future frame without a valid hidden identity.

Passing authorizes only a separate, preregistered source stress test with
observation noise and dropout. Failure stops this route without tuning on the
hidden outcomes.

## Claim boundary

This is retrospective mechanism evidence on an opened cohort. Manual tracks
emulate sparse material sensors and are not a deployable observation source.
Even a numerical crossing of the published 8/15 mm point would not establish
open-loop state of the art because the input information differs. No held-v8
artifact or sealed target is authorized.

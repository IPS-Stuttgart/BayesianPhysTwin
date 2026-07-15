# Deform360 cardinality source execution lock v1

This operational addendum closes two details that were underspecified in the
independent `002-rope-silk` preregistration: the exact train/tail frame boundary
and which physical-parameter arm can open calibration and target evaluation.
It was locked after source-only frame and mask staging began, but before dense
object trajectories, source-tail metrics, calibration episodes, or the sealed
target were read.

The staged interval contains 81 frames beginning at raw frame 110. Frames
`[0,64)` are the only frames permitted for physical-parameter and trust
selection. Frames `[64,81)` are untouched source tails. The boundary applies
the same `floor(0.8 T)` rule as the 60/76 discovery split.

The source-pooled physical grid is the primary arm. In each of six outer
leave-one-action-out folds, the physical tuple and cardinality-normalized trust
weights are selected jointly using only the other five episodes' train frames.
The held episode's tail is then evaluated once. The inherited `081-stripe-rope`
physical tuple is a transfer control and cannot by itself open calibration or
the target.

All source gates in the parent protocol are evaluated on the six outer held
tails. In particular, each joint win is an episode that improves both track
RMSE and symmetric Chamfer, and the maximum-degradation gate applies to each
metric in every held episode. Calibration episodes remain inaccessible unless
the primary arm passes every parent gate; the sealed target remains inaccessible
until the subsequent calibration gate passes.

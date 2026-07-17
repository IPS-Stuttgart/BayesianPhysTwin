# Deform360 reusable-PhysTwin action-window addendum v1

This addendum binds the reusable-PhysTwin protocol to an 81-frame action-rich
slice before any held development outcome is evaluated. Full-episode 3DGS
reconstruction is unnecessarily expensive for the first method-selection
panel. The slice rule is inherited byte-for-byte from the previously locked
Deform360 dense-panel protocol rather than selected from the new objects.

The start maximizes mean gripper-centre path weighted by closure confidence.
It reads only the released robot action, gripper aperture, and episode length.
Object images, masks, geometry, tracks, tactile signals, and evaluation errors
cannot affect the selected start. Ties select the earliest candidate.

Each raw slice contains 81 frames. Five terminal frames are reserved for the
tracking overlap, leaving 76 processed frames. Evaluation uses frames `[1,76)`
with fixed early, middle, and late intervals `[1,26)`, `[26,51)`, and `[51,76)`.

For a held episode, the complete known action may select the window because it
is conditioning input. Before the prediction is checksummed, the model may read
only the selected first object frame; future geometry, tracks, video, and
tactile data remain sealed. The addendum does not establish parity with the
unreleased Deform360 evaluator and cannot by itself support a direct Table 4
claim.

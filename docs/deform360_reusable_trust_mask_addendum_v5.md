# Deform360 reusable-twin source-trained camera panel v5

V4 tested the appealing idea that all 32 calibrated cameras should improve
frame-zero reconstruction. It did not: on source cable episode 1, accepted views
fell from 6 to 3. Additional cameras introduced correlated occlusions and false
grounded masks into an objective that was not robust to a majority of weak views.

V5 treats camera trust as a learned source quantity. It freezes the views accepted
by leave-one-view multiview consistency on fit episode 1, without object outcomes:

- cable: 6 cameras;
- scarf: 9 cameras;
- penguin: 9 cameras.

The minimum confirmatory panel contains four independent calibrated views. This is
not justified by episode 1 alone. The exact camera names, Grounding DINO/SAM2
configuration, four-view consensus gate, visual-hull gate, and 3 cm geometry
contact envelope are frozen before validation. Episodes 3, 4, 6, 7, and 9 are an
untouched fit-side transfer panel. If any object arm does not transfer under these
unchanged settings, that arm is rejected and returns exact persistence; its camera
set and thresholds are not retuned.

A frozen camera that yields no grounding box is recorded as unavailable rather
than aborting the episode. It contributes no evidence, and the unchanged four-view
multiview gate still decides admission. This is the intended missing-view behavior,
not an imputed mask or a relaxed gate.

The delayed-contact policy from v4 is retained. It uses only the known controller
trajectory and frame-zero hull, requires two consecutive frames inside 3 cm, and
latches contact thereafter. It uses neither source nor target tactile and never
uses post-initial object observations.

This is source training, not a held benchmark result. Held episodes remain sealed
until the source camera panel and official Warp reference smoke pass.

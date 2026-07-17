# Deform360 reusable-twin observation/contact addendum v4

This source-only addendum replaces the v3 admission rule before any held media or
outcome is opened. V3 established that text-grounded SAM2 recovered the correct
object identity for cable, scarf, and penguin, but exposed two protocol defects:

- the borrowed 12-camera subsets admitted only 6 cable views even though each
  episode releases 32 calibrated views;
- the penguin was correctly reconstructed at frame zero but the grippers reached
  it only 15 and 20 frames later, so frame-zero contact was the wrong requirement.

V4 freezes every common calibrated camera in lexicographic order. Grounding DINO
and SAM2 remain proposal mechanisms only; cross-view geometry decides admission.
The camera set is determined by calibration availability, not mask quality.

Contact is evaluated separately from object reconstruction. For each known
controller group, the policy measures distance to the frame-zero visual hull over
the known action. Two consecutive frames inside the unchanged 3 cm envelope start
contact, which is then latched. Latching is required because distance to the
*initial* object cannot identify release after the object has moved. The same
policy conditions controller springs inside Warp.

The information boundary is unchanged:

- allowed: frame-zero RGB, calibration, and known future robot action;
- forbidden: post-initial object observations, target tactile, object outcomes,
  and simulator residuals;
- failure: exact persistence, never a wider contact radius.

The v4 source run is an admission diagnostic. It does not establish benchmark
performance, and the 76-frame panel remains insufficient for a state-of-the-art
claim until the frozen held episodes and official full-horizon protocol are run.

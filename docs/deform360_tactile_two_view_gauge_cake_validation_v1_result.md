# Conservative two-view tactile gauge: cake validation result

## Result

The independently locked `153-cake` source case stopped at the first source
gate. The all-camera robot-prefix artifact was valid and replay-verifiable, but
the frozen quality gate rejected it:

- contact-ready frames by gripper: `0/1`, versus at least 4 each;
- both-finger support fractions: `0.000/0.0238`, versus at least 0.5 each;
- maximum inferred opening: `157.55 mm`, outside the released 40--112 mm
  interface;
- wrist support, rigid-transform validity, rotation steps, and translation
  steps otherwise passed.

The registered disposition is therefore exact baseline fallback. No tactile
value, MotionCrafter NPZ member, prediction score, future frame, confirmation
payload, target outcome, or held-v8 artifact was opened.

## Interpretation

This is not a negative result for the conservative two-view metric gauge. That
gauge was never evaluated because its required robot/contact parent was not
admitted. The result instead shows that the current released UMI robot-prefix
contract does not cover this interaction reliably enough to support tactile
contact geometry.

The gate will not be widened retrospectively. In particular, treating a
157.55 mm opening as valid after seeing this case would alter the physical
interface and compromise the prospective test. The frozen protocol does not
authorize replacement of this object.

## Compatibility control

Before the fresh case was opened, the previous three-view gauge was replayed
under revision `4f7a1572`. Its complete scientific payload was exactly equal to
the frozen result after removing only runtime provenance fields and the derived
artifact ID. The two-view implementation therefore preserves old-schema
behavior.

## Provenance

- validation lock ID: `e11efef16fe0d42039e2bb6f707a00e420a7fdecc1315bce8efa26343b0b95b5`;
- robot lock ID: `994a056a440ab5e7c2021436559e8f27c4d1479cc0309fd138fc07c1c79ebfe1`;
- robot artifact ID: `d78cd59b726be5831525d89fe445920536f9690037e7ee9418d877d0ccfa46fe`;
- robot manifest SHA-256: `7c9387ee76f8c45d2336ceeea4813fa6ee3bc15db4b8f37de12278543680ebf7`;
- robot archive SHA-256: `3254b545203118e2fd538029d8eafd6cc9c240f20aedffe1d0f9eccc6c7a1d08`.

The two-view gauge remains a source-locked but independently unvalidated
method. A later evaluation requires a new, preregistered fixed cohort or an
independently justified robot-interface model; this case cannot be silently
replaced.

# Deform360 official-Hub causal schedule v2

## Motivation and information order

The frozen v1 schedule supported only 4 of 10 calibration objects because six
contacts occurred before a 42-frame prefix ending six frames after contact could
be placed. This result was opened without running MotionCrafter, decoding camera
images for provider inference, computing a prediction score, fitting a policy, or
accessing confirmation payloads.

V1 remains closed and reproducible. V2 is a calibration-derived schedule recovery,
identified separately by
`protocols/locks/deform360_official_hub_visuotactile_v2_causal_schedule_recovery.json`.
It binds the v1 failure manifest, the unchanged visual execution lock, and the
unchanged Prob4D/MotionCrafter provider.

## Frozen rule

For first tactile contact frame `t_c`, v2 uses:

```text
causal cutoff (exclusive) = max(t_c + 6, 42)
observed source interval  = [cutoff - 42, cutoff)
future evaluation         = [cutoff, cutoff + 24)
```

The observed interval still yields exactly two independent 25-frame windows with
eight-frame overlap. Every arm receives the same object-specific prefix and future.
No synthetic left padding is introduced.

When `t_c < 36`, v2 deliberately waits beyond six post-contact frames until all
42 source frames exist. Thus the amount of observed post-contact response varies
with event time. That is a declared consequence of preserving the two-window
provider, not an unreported boundary adjustment.

An episode lacking contact or 24 untouched future frames remains a retained
technical failure. It is not replaced. At least 8 of the 10 locked calibration
objects must remain supported before provider inference may proceed.

## Claim boundary

V2 was selected after calibration schedule feasibility was known, so it is not a
pre-payload preregistration. It is locked before provider output or endpoint score
inspection and can still support a prospective test on the separately sealed
confirmation objects if all calibration gates later pass. The schedule itself is
not evidence of provider competence, uncertainty calibration, physical-state
improvement, tactile benefit, or state-of-the-art performance.

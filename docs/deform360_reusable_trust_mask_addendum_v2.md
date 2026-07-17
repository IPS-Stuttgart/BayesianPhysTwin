# Deform360 reusable-twin mask addendum v2

This addendum repairs a source-preprocessing failure without changing the
physical model, trust rule, fit grid, held cohort, or admission thresholds.
The lock is
`configs/causal4d_public/deform360_reusable_trust_mask_addendum_v2.json`.

## Why a new addendum is necessary

The v1 fallback selected a generic SAM2 candidate independently in every
camera. It failed before any Warp outcome was produced:

- `003-cable`: the reconstructed hull was 221.2 mm from the controller;
- `086-cotton-scarf-cloth`: the hull was 121.8 mm from the controller;
- only 22.8% of the reconstructed scarf projected inside the median selected
  camera mask.

The 30 mm controller radius is therefore not widened. A source-only diagnostic
using the existing appearance and calibrated joint-multiview selector retained
10 of 12 scarf cameras and produced a 2,461-point hull 4.3 mm from the
controller.

## Frozen mask rule

Episode 1 supplies one object-level appearance reference. Its mask is the top
generic SAM2 candidate in the fixed reference camera on the exact start frame
selected from the known robot action. No manual point, box, or mask is used.

For every fit or held episode:

1. read only the exact action-window start RGB from 12 frozen cameras;
2. rank at most four SAM2 masks per camera using source-reference appearance;
3. select candidates jointly using calibrated 3D consistency;
4. carve a frame-zero visual hull;
5. require at least eight accepted cameras, 512 hull points, and a hull point
   within 30 mm of the known controller surface.

Failure rejects the episode to exact persistence. It never changes the
attachment radius or uses a simulator residual to choose a mask.

## Information boundary

The selector may use camera calibration, the source reference, the current
episode's frame-zero RGB, and the known robot trajectory. It may not use
post-initial object frames, tactile, object outcomes, or simulator residuals.
The same source reference is reused for fit and held episodes. All held
predictions must still be hashed before any held outcome is opened.

The helper command is
`scripts/remote/build_deform360_reusable_trust_masks.py`. Its packed
`sampled_masks.npz` is directly consumable by
`scripts/remote/stage_deform360_dense_source_smoke.py`.

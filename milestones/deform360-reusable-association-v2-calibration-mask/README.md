# Deform360 reusable association v2: calibration mask gate

This milestone freezes the independent frame-zero mask-association result for
the three calibration episodes declared by
`deform360_reusable_association_v2.json`.

The method and source-only evidence were frozen before any of these calibration
frames were read. The execution runner hard-rejected every episode outside
`081-stripe-rope` episodes 0, 2, and 8. Each episode passed the preregistered
minimum of 10 calibrated-consistent cameras:

| Episode | Accepted cameras | Gate |
| ---: | ---: | :---: |
| 0 | 11 / 12 | pass |
| 2 | 12 / 12 | pass |
| 8 | 11 / 12 | pass |

All selected masks were the highest-ranked source-appearance candidate. The
calibrated 3D selector therefore acted as a feasibility gate and did not replace
an already valid appearance identity in this calibration set.

Only frame zero was decoded. No future geometry, dynamics metric, or target
episode was read. Passing this milestone permits the separately locked
six-frame temporal-identity gate; it is not a future-prediction or state-of-the-
art result.

The exact checksummed JSON records and selected initial-mask archives are under
`artifacts/`.

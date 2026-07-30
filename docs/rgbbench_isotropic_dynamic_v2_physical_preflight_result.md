# RGBench Isotropic Dynamic v2 Physical Preflight

## Status

The target-free physical gate passed. This authorizes source-only development
of the frozen temporal-discrepancy bank. It does not authorize calibration or
target outcomes.

The operator used Bayesian-PhysTwin commit
`f78230eac49806d9055979148940928705af2eea` and upstream RGBench commit
`eddae2f28f388b4706d65d626f67bc9e34b14c68`.

## Mesh artifacts

The final seven-garment mesh manifest has:

- canonical artifact SHA-256
  `54f337666cc1b9f51b4bd01ff3a4ebdca9ebee4f82779e386c6681d3966a9418`;
- file SHA-256
  `30c1adf910afbbb24a99b94f3282ad0bad1b0a797c77102bac568bbb753da250`.

Five garments use the first geometry-admissible isotropic remesh, the original
green T-shirt remains byte-preserved, and the 20,279-node pleated skirt uses
the clean released source mesh after every under-cap isotropic candidate
introduced self-intersections.

## Physical replay gate

Sample `01` was run twice for all seven garments and all three actions in
PyBullet DIRECT mode:

```text
7 garments x 3 actions x 2 replays = 42 replays
```

All 42 runs completed. Every one of the 21 replay pairs produced a
byte-identical compressed trajectory archive. The exact gate artifact is
`results/sota/rgbbench_isotropic_dynamic_v2/physical_preflight_gate.json`,
with file SHA-256
`d252a6cf0531f8361da965cbd06d3c447aa2cda7964304b05da444598c16af93`.

The gate read frozen point-cloud filenames to reproduce timestamps and used
the known actuator trajectories as interventions. It did not parse any real
point-cloud coordinate or future object outcome.

The byte-preserved green-shirt grasp replay retained the predecessor hash
`c62d7ffdb07f09bcd996d83084c64caa70b0d1cb547db0d3a0ea0681114a7d38`,
providing an additional parity check against RGBench v1.

## Decision

The public-backend failure that closed RGBench v1 is resolved without dropping
a garment or counting a technical failure as a prediction. Source work may
now proceed on exactly the 27 frozen captures from `white_cakeskirt`,
`brown_coat`, and `green_tshirt`.

Calibration and target garments remain sealed until their separately frozen
gates authorize them.

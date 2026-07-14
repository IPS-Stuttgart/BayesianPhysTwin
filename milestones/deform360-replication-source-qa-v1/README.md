# Deform360 Replication Source Geometry QA V1

This milestone freezes the source-only camera and first-frame mask boundary for
the six-object public replication. It does not contain target media, target
metrics, or a replication result.

## Locked Inputs

- Replication config SHA-256:
  `f0aab308345807b2183f653306a062d4ad0295584b6b283deb99d29b3c247934`
- Source-QA policy SHA-256:
  `f5c3f4577bab648bf3da2537c28b7a77e87c055b32e1284c179a192e7569909b`
- Dataset revision:
  `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`
- Exploratory source artifact SHA-256:
  `5d2880c286cedb558dd7b36e929ce95091eab0c3c8587b66af516ed9020b26c9`

## Result

The executable QA passed all six objects:

| Object | Appearance views | Cross-view views | Peak votes | Core voxels |
| --- | ---: | ---: | ---: | ---: |
| `002-rope-silk` | 30 | 27 | 26 | 178 |
| `081-stripe-rope` | 30 | 23 | 22 | 577 |
| `085-scarf-cloth` | 29 | 23 | 25 | 29,520 |
| `083-blanket-cloth` | 31 | 27 | 28 | 46,933 |
| `092-squirrel` | 31 | 26 | 27 | 609 |
| `170-spider` | 31 | 26 | 26 | 6,836 |

Each object therefore exceeded the 12-camera gate. The artifact stores the
deterministically selected 12-camera subset for downstream processing.

- Internal result SHA-256:
  `2d7f0f4be5d27af1c2d6abb87168e0bf1a07c335287a6d15541c266adf22290f`
- Artifact file SHA-256:
  `5cba6655ba3714f949a0342813e94c2c72fee0112ab9a0f6734b1d5b44b0501c`

The raw `001` camera mask is rectified with nearest-neighbor interpolation and
used only as a source appearance anchor. Other cameras must pass both
reference-conditioned SAM2 selection and calibrated 3D leave-one-view
consistency. No camera was selected using downstream prediction accuracy.

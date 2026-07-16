# Deform360 reusable association v2: calibration prefix gate

This milestone seals the independent six-frame material-identity calibration
for `081-stripe-rope` episodes 0, 2, and 8. The method, thresholds, source
evidence, and frame-zero masks were frozen before these prefixes were
reconstructed.

| Episode | Gaussians per frame | Min. match | Min. effective support | Gate |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 695 | 100.00% | 96.27% | pass |
| 2 | 557 | 100.00% | 96.78% | pass |
| 8 | 653 | 100.00% | 96.75% | pass |

All frames stay below the frozen cardinality cap of 2,238 Gaussians. Every
transition exceeds the 95% match and 80% effective-reliable-support gates.
Together with the previously sealed 11/12, 12/12, and 11/12 multiview mask
results, all three calibration episodes pass the conjunctive reusable-
association gate.

The test used raw frames `[0,6)` exactly as authorized by the protocol. It did
not compute future Chamfer distance, track error, or any physical rollout, and
it did not read a target episode.

Three operational attempts on episode 0 stopped before any splat or gate
statistic was produced: missing staged timestamps, an unset CUDA toolkit path,
and a missing Ninja path. Their directories remain archived on `gpuserver6000`.
The successful execution used the corrected runner at commit `43e5b92`, the
existing `/usr/local/cuda` toolkit, the pinned Deform360 Ninja binary, and
`TORCH_CUDA_ARCH_LIST=8.9`. No scientific setting changed.

This is an independent association-transfer result. It permits calibration-
episode dynamics evaluation under a separate frozen protocol; it is not yet a
future-prediction or state-of-the-art result.

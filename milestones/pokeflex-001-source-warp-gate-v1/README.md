# PokeFlex Source-Warp Gate V1

This milestone records the locked development-only source QA and official-Warp
admission test for the first PokeFlex object. The numerical backend is rejected
by its prospective transfer gates. The negative result is retained without
expanding or retuning the candidate grid.

## Information Boundary

Only the five metadata-selected development takes were opened:

```text
3dPrintedBunny_T1
3dPrintedBunny_T3
3dPrintedBunny_T4
3dPrintedBunny_T6
3dPrintedBunny_T7
```

`3dPrintedBunny_T5` remains the unopened calibration take and
`3dPrintedBunny_T2` remains the unopened target take. No calibration or target
mesh, robot record, prediction metric, or outcome was read.

The source QA established that:

- measured tool poses and the locked force threshold agree with surface
  proximity in all five development takes;
- the reconstructed meshes are sufficiently closed and have no nonmanifold
  edges under the locked checks;
- independently reconstructed meshes do not establish persistent material
  vertex identity;
- cross-take shape variation rejects one shared canonical sparse graph;
- take-specific graphs with shared simulator settings remain technically
  admissible.

Source-QA result SHA-256:
`e09d36db4e1ba8a38c70e112c3af9ab95516ee245302f71a853f36cd2dd0e0e7`.

## Backend

Each take uses a 128-node farthest-point surface sample and an eight-neighbor
take-specific spring graph. The first six frames initialize the graph, and the
official PhysTwin `spring_mass_warp.py` simulator rolls forward at the public
30 Hz cadence using 64 substeps. The locked 50-candidate grid shares object
spring setting, controller spring setting, and ground friction across takes.

This source adequacy test uses the released future tool trajectory and the
force-derived contact schedule. It is therefore favorable to the simulator and
is not a deployable contact predictor. The evaluated values are simulator
configuration parameters, not identified material constants.

The official dependencies are pinned to:

```text
PhysTwin commit: 2b6630528141b9cba5a7677c8b88b2129b4a8390
simulator SHA-256: 7deab9a25f4b8b8772f7df45c35571caf3767d014dd353cad151fe8eddceca1c
PokeFlex commit: aaa8726072834a95bbe97e1a113588968c36e185
Warp policy SHA-256: bf534da2543116f486472581e842786f55b1fa538b0f06d5cb9b5d98c6904c26
```

## Result

The backend failed both predictive admission gates while passing numerical
determinism and strain plausibility.

| Quantity | Observed | Required |
| --- | ---: | ---: |
| Pooled source mean Chamfer | 23.771 mm | descriptive |
| Persistence source mean Chamfer | 10.093 mm | baseline |
| Leave-one-take wins over persistence | 0/5, 0% | at least 3/5, 60% |
| Pooled wins over median single-source selection | 2/5, 40% | at least 3/5, 60% |
| Maximum repeat-rollout RMSE | 0.000 mm | at most 0.100 mm |
| Maximum selected p99 edge strain | 19.37% | at most 50% |

The pooled selector chose object spring setting 10000, controller spring
setting 300, and ground friction 0.3. Its mean error is 135.5% higher than
persistence.

More decisively, the best candidate chosen separately with each held-out
development take's own outcome also loses to persistence:

| Take | Persistence | Leave-one-out pooled | Per-take oracle |
| --- | ---: | ---: | ---: |
| T1 | 9.467 mm | 33.875 mm | 22.718 mm |
| T3 | 11.401 mm | 25.609 mm | 22.311 mm |
| T4 | 10.463 mm | 19.083 mm | 19.083 mm |
| T6 | 10.403 mm | 27.943 mm | 21.743 mm |
| T7 | 8.734 mm | 23.499 mm | 23.499 mm |

Warp artifact result SHA-256:
`08e6a57b7ac3b16d25b590947fcb6cea0f5a3a703f956f69a0dbe13343dfd2cf`.

## Interpretation

This rejects the tested sparse take-specific surface-spring model as a useful
PokeFlex backend for this object. It does not reject PokeFlex, the full PhysTwin
reconstruction pipeline, volumetric or bending-aware graph construction, or
Bayesian-PhysTwin. In particular, the sparse graph lacks an identified
volumetric interior and persistent material correspondences.

The target must remain sealed. A richer backend should be attempted only as a
separately specified method-family test, with a volumetric/canonical twin and
source-only acceptance gate. The failed backend cannot be rescued by selecting
on `T2` or `T5`.

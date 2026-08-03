# PokeFlex Public Paired v1 Result

## Outcome

The unchanged source-selected Bayesian state update improves the released
PokeFlex checkpoint on the object-balanced mean, but it does not pass the full
registered transfer gate. CD_UL1 falls from 4.7030 mm to 4.6565 mm, a 0.9886%
improvement. Thirteen objects improve, one unsupported object uses exact
fallback, and one object regresses. The 97.5% object-bootstrap upper bound for
candidate minus baseline is -0.0293 mm.

| Take | Checkpoint (mm) | Candidate (mm) | Improvement | Supported/scored |
| --- | ---: | ---: | ---: | ---: |
| `3dPrintedCylinder_T4` | 3.081 | 2.998 | 2.689% | 91/96 |
| `3dPrintedHeart_T2` | 3.330 | 3.247 | 2.520% | 72/76 |
| `3dPrintedPizza_T3` | 4.136 | 4.103 | 0.792% | 48/52 |
| `3dPrintedPyramid_T3` | 2.351 | 2.301 | 2.137% | 59/63 |
| `Beanbag_T5` | 4.766 | 4.711 | 1.151% | 60/65 |
| `FoamCylinder_T5` | 5.234 | 5.152 | 1.570% | 41/44 |
| `FoamHalfSphere_T1` | 2.579 | 2.521 | 2.268% | 81/86 |
| `Pillow_T1` | 6.819 | 6.807 | 0.173% | 67/72 |
| `PlushDice_T4` | 6.895 | 6.798 | 1.409% | 67/72 |
| `PlushMoon_T6` | 6.112 | 6.075 | 0.599% | 105/110 |
| `PlushOctopus_T2` | 4.615 | 4.557 | 1.267% | 65/70 |
| `PlushTurtle_T4` | 6.852 | 6.794 | 0.845% | 78/83 |
| `PlushVolleyball_T1` | 5.188 | 5.188 | 0.000% | 0/88 |
| `Sponge_T1` | 4.285 | 4.304 | -0.459% | 99/106 |
| `ToiletPaperRoll_T2` | 4.301 | 4.291 | 0.227% | 77/82 |

## Registered Gates

The positive mean, bootstrap, minimum-win, and support-breadth criteria pass.
The frozen no-object-regression criterion fails solely because `Sponge_T1`
regresses by 0.4585%. Consequently, `paired_transfer_passed` and
`all_target_gates_passed` are both false. The result is a prospective near-pass
and development evidence for a target-free guard, not a positive confirmation.

The candidate's 4.6565 mm is numerically 28.34% below the published 6.498 mm
Kinect value. The published value uses an unavailable internal split, so this
cross-split number remains context only and is not an official-table SOTA claim.
Jaccard is also non-gating: 868 of 1,165 frame evaluations are valid, for a
74.51% valid fraction and a valid-object mean of 0.8958.

## Custody

All 15 predictions were sealed at clean pre-outcome commit
`c015534497c2a37aacd5c059ad48bed3721b6e3f`. Causal input staging read zero
future mesh members. The input archive moved directly from `gpuserver6000` to
`gpuserver4090` over their LAN; the jump server did not relay the payload.

The first batch environment lacked `pyvista`, causing nine technical failures
before prediction output for those cases. Existing successful predictions were
preserved, and only those nine cases were rerun in the correct frozen
environment. All succeeded. The complete 15-case barrier then validated with
canonical digest
`d37ea9d533ab8e5ac47f5df387b8ffb21acb6300d0378213c1160a259fd9ba35`.
Only after that barrier were the 1,165 registered target meshes staged and
scored. The exact target result has SHA-256
`d3a03ca0c5f834cb5ae8fff840027e4ae919d3f94451a11847ab0b319221bb3c`.

## Next Experiment

`Sponge_T1` is now opened development evidence. It must not be used to alter
the claim on this cohort. The next method must learn a target-free,
baseline-relative abstention rule from source data plus this opened cohort,
lock that rule before outcomes, and evaluate it on newly selected public takes
that remain absent from all tracked development history.

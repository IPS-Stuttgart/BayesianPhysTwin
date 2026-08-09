# PokeFlex All-18 Robust-Scale Source V4 Result

## Outcome

The frozen source gate passed. The unchanged repeated-action maximin rule
selected a non-global correction multiplier for all six previously uncovered
public PokeFlex object identities. Every selected multiplier improved both
source actions relative to multiplier one.

| Object | Selected multiplier | Mean source gain | Minimum source gain |
| --- | ---: | ---: | ---: |
| `3dPrintedBunny` | 3.0 | 2.208% | 1.297% |
| `3dPrintedHeart` | 1.5 | 0.269% | 0.195% |
| `FoamDice` | 4.0 | 2.830% | 2.296% |
| `MemoryFoam` | 1.5 | 0.509% | 0.340% |
| `PlushOctopus` | 3.0 | 1.449% | 0.471% |
| `ToiletPaperRoll` | 3.0 | 1.459% | 0.925% |

Across the 12 source actions, the mean relative improvement is 1.454%, the
minimum is 0.195%, and the regression count is zero. All six objects exceed
the preregistered requirement that at least three select a non-global scale.
The unchanged synthetic controls also pass: 12/12 positive detections and
0/12 placebo deviations.

The resulting calibration completes a source-derived scale map for all 18
public object identities. It is stored at
`configs/sota/pokeflex_action_robust_scale_all18_v4.json`.

## Interpretation

This source result strengthens the mechanism behind the earlier prospective
fresh-six result: the useful correction magnitude is object dependent, and a
maximin choice across repeated actions can adapt that magnitude without
regressing either source action. The strongest new source gains occur for
`FoamDice` and `3dPrintedBunny`; the smallest occurs for `3dPrintedHeart`, but
it remains positive on both actions.

This is advancement evidence, not a state-of-the-art evaluation. The source
actions are opened development data. The 13 publicly reproducible official
targets were already opened for v3 and therefore cannot independently confirm
v4. Five records from the published 18-object validation split still lack a
reproducible mapping to the public release. No official-18 run is authorized
until the PokeFlex authors provide those mappings or the processed validation
set with checksums.

## Provenance

The source protocol canonical digest is
`7a7b291418964ed7ccaf54f2eb4e2db25badf35edc3cb68d4ca484e0b0a6ed03`.
The 12 source predictions were generated from frozen commit
`d19e97618fa5fa3af7118a99cfe43a070fd25032`. A result-recording fix then made
the generated calibration retain every already-verified source-artifact file
hash; it did not alter scores, selection, or the gate.

Pre-result regression testing also found that the first wrapper implementation
had changed the bytes of a legacy evidence-bound runner to add source
authorization. The final implementation restores that runner to its registered
SHA-256,
`79ba8946653a55a70dc0b990e874754397e18948b9b7ba541158c6641cfc4b43`,
and confines v4 admission to the new wrapper. Replaying
`3dPrintedBunny_T2` through the corrected wrapper reproduced the existing
source JSON exactly.

The calibration canonical digest is
`e94eeb9bdd2cc69e245b0bd48d843e5f64cb039e1eb02841e4a784cbe4dbc880`,
and its file SHA-256 is
`00cdf5732f5dbf7eb0f899ebbb536260d9e66c0a151b41eec81ffaaef4aaf110`.
All 12 source JSON hashes are embedded in that calibration.

Source payloads moved directly over the server LAN from `gpuserver6000` to
`gpuserver4090`; the jump server was not in the payload path. No Deform360
held-v8 artifact, process, identity, query, or outcome was accessed.

## Decision

Retain the all-18 robust scale map as the preferred PokeFlex guarded-update
configuration. Seek the missing official validation mapping from the authors,
then preregister and run the complete official split with the released
checkpoint and global-scale arms preserved. Do not tune this map from the
already opened public-13 outcomes.

# Deform360 author-released evaluator smoke v1

This milestone records a development-only evaluator smoke on the
author-released processed particles for Deform360 episode `001-rope/0`.
It closes the material-identity and episode-horizon contracts for this open
episode. It does not establish parity with Deform360 Table 4.

## Provenance

- Dataset: `brownu/deform360`
- Hugging Face revision:
  `93280cbb466de6b9e59927c58a99fd3b9e91900e`
- Episode: `processed/001-rope/episode_0`
- Ordered material particles: 5,426
- Released particle frames: 252, from source frame 0 through 251
- Prediction origin: source frame 173
- Evaluation frames: source frames 174--197
- Frame rate: 30 Hz
- Maximum released ordered-advection residual: `2.9802322387695312e-08 m`

The particle identity is bound to the ordered frame-zero positions. Every
loaded NPZ, the metadata, and the split file are checksummed. The released
relation

```text
x[t + 1] = x[t] + v[t] / 30
```

is checked across all 251 released frame transitions before an evaluator
contract can be created.

## Smoke result

The prediction is exact last-training-frame persistence. Two explicit metric
contracts are archived because the paper's public description does not
disambiguate all metric semantics:

| Contract | Future Chamfer | Future identity error |
| --- | ---: | ---: |
| Symmetric mean Euclidean CD; mean Euclidean track | 1.2151 mm | 1.7416 mm |
| Symmetric mean squared CD; RMSE track | 2.0876 mm^2 | 2.0324 mm |

The squared Chamfer value in the table is the archived
`2.0876304868271323e-06 m^2`, expressed as `2.0876 mm^2`.

## What this establishes

- Author-released particles provide stable ordered material identities.
- The author-released split provides an episode-specific prediction origin and
  test horizon.
- Exact persistence can be scored without any future observation entering the
  prediction.
- Chunked exact Chamfer scoring handles the released particle cardinality
  without allocating a full three-dimensional pairwise-difference tensor.

## What remains unresolved

The complete Table 4 object/episode split, exact metric implementation,
aggregation, evaluator entrypoint, ParticleFormer predictions, and reference-row
reproduction remain unavailable. The artifacts therefore have
`independent-protocol` status, and the Table 4 authorization function rejects
them.

This episode was already in the public development set. No prospective
confirmatory object or future, PokeFlex target, held-v8 artifact, or frozen
Causal4D artifact was opened or modified.

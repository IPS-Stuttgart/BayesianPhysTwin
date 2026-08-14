# Newton implicit-MPM backend compatibility result v1

## Decision

**Compatibility gate passed; empirical advancement gate not yet evaluated.**

Newton's implicit MPM solver executed on CUDA, produced a deterministic
particle rollout, and materialized the simulator-neutral Bayesian-PhysTwin
physical-rollout contract. This result authorizes source-only backend
development. It does not authorize a target evaluation or claim that MPM
improves accuracy, calibration, or state of the art.

## Frozen implementation

- BayesianPhysTwin implementation commit:
  `11800f1f4a0fdb8f40f7083e0ba2a88aee7a05cb`
- Newton: `1.5.0`
- Warp: `1.16.0`
- Python: `3.12.13`
- Device: `NVIDIA GeForce RTX 4090 Laptop GPU` on `cuda:0`

The implementation reads no dataset payload, future observation, or target
outcome. The only input is the registered synthetic beam scene and its known
boundary action.

## Smoke result

The driven and zero-action arms each simulated 76 frames and 625 persistent
material particles. Deterministic farthest-point sampling retained 64 material
queries.

| Check | Result |
| --- | ---: |
| Requested right-end displacement | 25.000 mm |
| Maximum driven-minus-zero response | 25.000002 mm |
| Median final response over retained queries | 12.579754 mm |
| Maximum zero-action drift | 0.000000 mm |
| Frame-zero identity | exact |
| Existing 76-frame physical consumer | passed |
| Same-runtime replay bytes | identical |

The second execution reproduced the raw particle archive, portable physical
archive, and artifact manifest byte for byte.

## Artifact identities

- artifact ID:
  `2da872ff8dc50d6b4c2595ebcf7f1f02b476a41059b35152a52b7e8a53ec3296`
- runtime ID:
  `f3490aca1c0447726b5d5950a91aa99aa461a4482f536e761c3bec21e5663e35`
- raw particle archive SHA-256:
  `87c6b369b03fbba9b1706e2a3f66d7886a670906c36e713a4b159bb4b7835978`
- physical archive SHA-256:
  `6c3eb92dcfb87d5368c9c084f24a1346deacb597d7f8b02e8eb4ef4d05086e1b`
- runtime manifest SHA-256:
  `4afec6f5efda84513bbc8ebaec6350c941846cd7732e7c2eae9a13f54574a6e6`
- material-query index SHA-256:
  `c234f7dd84c444d2e968bfbc8792c0233432d13edc30a8ca394776355a74ca0d`

## Verification

- focused MPM, MatPhys compatibility, command-registry, and distribution
  contract suite: 61 passed;
- changed-file Ruff: passed;
- strict MyPy on the portable adapter and CLI: passed;
- wheel and source distribution build: passed;
- Twine distribution checks: passed;
- base wheel import without the `mpm` extra: passed and did not import Newton
  or Warp.

## Next gate

The next admissible experiment is one already-open source object with a frozen
geometry-to-particle map, measured action/contact boundary, repeated numerical
replay, and exact incumbent fallback. The MPM arm must improve or complement
the incumbent source prediction and produce plausible non-degenerate ensemble
spread before any independent-object protocol is designed.

# Genesis elastic-MPM backend compatibility result v1

## Decision

**Compatibility gate passed; empirical advancement gate not evaluated.**

Genesis 1.2.2 executed its elastic MPM solver on CUDA, preserved material
particle identity, and produced Bayesian-PhysTwin's engine-neutral
`physical_rollout_v1` artifact. The result authorizes source-only backend
development. It does not authorize a target evaluation or claim that Genesis,
PhysWorld, or an MPM twin improves accuracy, calibration, or state of the art.

## Frozen implementation

- BayesianPhysTwin commit:
  `a1bbdb844dcc019d0b323fa1ad7be84af8e0cc1d`
- Genesis release: `1.2.2`
- Genesis source commit:
  `72bdd98b8d77d249e9c5296e4a6746ff680416a2`
- PyTorch: `2.13.0+cu130`
- Python: `3.12.3`
- device: `NVIDIA GeForce RTX 4090`

The implementation reads no dataset payload, future observation, or outcome.
The only input is a registered synthetic beam and its known gripper action.

## Native smoke

The driven and zero-action arms each simulated 40 frames and 690 persistent
material particles. Deterministic farthest-point sampling retained 64 material
queries. Both beam ends used compliant particle constraints; one end followed
a 10 mm action and the other remained stationary.

| Check | Result |
| --- | ---: |
| Requested gripper displacement | 10.000000 mm |
| Maximum driven-minus-zero response | 10.224522 mm |
| Response/action ratio | 1.022452 |
| Frozen stability cap | 3.000000 |
| Maximum one-frame particle step | 0.312886 mm |
| Frame-zero identity | exact |
| Zero-action replay across native runs | byte-identical |
| Fixed-input portable rematerialization | byte-identical |

## Native replay floor

Two independent GPU executions retained the same frame-zero state and query
indices but were not byte-identical. Across all driven particles, the maximum
absolute coordinate difference was `1.4901161e-7 m` and RMSE was
`2.0085949e-8 m`. In the portable 64-query prediction, the maximum difference
was `8.9406967e-8 m` and RMSE was `1.9533904e-8 m`.

This is a measured sub-micrometre CUDA replay floor, not deterministic-output
evidence. Artifact production and validation remain byte-deterministic once a
raw rollout is fixed.

## Artifact identities

- primary artifact ID:
  `726241f0d5464fb62cfa4b851a357f3419dbaead96c47db6b1c278997c1ddd60`
- primary runtime ID:
  `c24de5162986bfc8c830bcb34b611b279eb487a88ff5e3fec8edfa9343d6f8f7`
- raw particle archive SHA-256:
  `cf51834cf9816d5bba4422e5830aaeb824c156aa3f42bdd5b77577b8e4363695`
- physical archive SHA-256:
  `4394bea1251506fe7615ef12549c8b3cbe3d934a7570c45303b04c720520c821`
- runtime manifest SHA-256:
  `3e6b8ad7251a1b781fba01d2247864596ff772a5f8616f4181b21f68c69561c0`
- material-query index SHA-256:
  `c87ac5f4890c79a977d4888bfb97c0660f373a956ec60055bcf8b5b4a5920f01`

## Verification

- focused Genesis, Newton compatibility, command-registry, and stable
  distribution suite: 80 passed;
- changed-file Ruff: passed;
- strict MyPy on the adapter, runtime boundary, and CLI: passed;
- wheel and source distribution build: passed;
- Twine distribution checks: passed;
- base wheel import without the `genesis-mpm` extra: passed and did not import
  Genesis;
- portable validation without the Genesis source tree on `PYTHONPATH`: passed.

## Next gate

The next admissible experiment is an already-open source object with a frozen
geometry-to-particle map, measured action/contact boundary, a numerical replay
ensemble, and exact incumbent fallback. It must test whether volumetric MPM and
compliant contact add source accuracy or complementary ensemble spread before
any independent-object protocol is designed.

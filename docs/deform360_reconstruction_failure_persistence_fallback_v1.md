# Deform360 Reconstruction-Failure Persistence Fallback V1

## Status

This is a target-free, post-open integration result. It closes the frame-zero
coverage failure of the first bias-aware prospective protocol without claiming
that fallback geometry is a valid physical twin.

The implementation is commit `29091da`. The checksum-bound audit is
`results/sota/deform360_reconstruction_failure_persistence_fallback_v1/integration_audit.json`
(file SHA-256
`5c06156459204e4a02e1625fea13428848bcdccc3d05b5a673df60de3ae0ed51`,
canonical result SHA-256
`8da661c0452a482f9aef3c90676740bcf31a6f7cc24d5ffb63df218687b266ec`).

## Contract

The source-frozen visual hull is evaluated only after the original 128-point
Splat admission check fails. A recovered hull receives the explicit policy
`persistence_only`:

- no automatic material graph is constructed;
- no PhysTwin or Warp rollout is attempted;
- no Bayesian state update is available;
- action support is identically zero; and
- prediction, persistence, driven readout, and zero-action readout are
  bit-identical for all 76 frames.

The admitted Splat path remains the existing `automatic_twin` path. Omitting
the new opt-in argument preserves the legacy frame-zero manifest and behavior.

## Integration Result

| Case | Failed Splat points | Fallback points | Exact persistence | Physical runtime absent |
| --- | ---: | ---: | ---: | ---: |
| `160-hose-ep0001` | 25 | 2,675 | yes | yes |
| `174-chain-ep0001` | 24 | 264 | yes | yes |
| `015-airbag-cloth-ep0006` | 44 | 3,495 | yes | yes |
| `100-puppet-ep0009` | 13 | 3,479 | yes | yes |

All four known reconstruction failures pass. The admitted parity control,
`011-green-cloth-ep0000`, preserves its 1,072 point/color arrays bit exactly
and retains `automatic_twin`.

## Rejected Alternative

A separate target-free source diagnostic tested whether the already declared
zero-action limits (99th-percentile displacement at most 250 mm and maximum at
most 500 mm) should force persistence even for admitted physical backbones.
The rule rejected 15 of 27 open source episodes. On the already-open outcomes,
it worsened the object-balanced selected baseline:

The complete result is
`results/sota/deform360_zero_action_abstention_source_v1/source_audit.json`
(file SHA-256
`305e53d798cd8d1ffc096dc08ab841bd402d6b2a69838caea8a614b3fbc0e40f`,
canonical result SHA-256
`61e963d2c1f8aa89652acbc2334fd2a4407a8879767f52b383c8a79405aed220`).

| Metric | Existing selector | Zero-action abstention | Change |
| --- | ---: | ---: | ---: |
| Hidden identity RMSE | 8.807 mm | 9.218 mm | +4.66% |
| Hidden Chamfer | 7.888 mm | 8.191 mm | +3.84% |

The rejected group contained much of the useful physical-backbone gain. Raw
zero-action displacement is therefore not a valid admission statistic for an
otherwise successful twin and is not implemented as a selector.

## Scoring Boundary

Visual-hull points do not inherit the official Splat material identities.
Consequently, fallback cases do not provide absolute identity RMSE or
calibration observations under the old outcome contract.

Their paired candidate-minus-baseline regret is nevertheless exactly zero for
any deterministic metric: the complete candidate and baseline trajectories
are bit-identical. A future protocol may count these dispositions as exact
non-regression ties, but not as absolute-accuracy or calibration samples.
Variable-cardinality absolute Chamfer would require a separately locked target
contract.

## Decision

The persistence-only recovery path passes its integration gate and may be used
in a new, genuinely fresh protocol. The old failed prospective protocol and
its twelve reserved targets remain untouched. A new protocol must still
predeclare a minimum number of full Splat/physical cases for absolute accuracy
and calibration; fallback ties cannot manufacture statistical power.

This removes a pipeline-coverage blocker while preserving the scientific
claim boundary: uncertainty-bearing state updates are attempted only when a
physical response exists, and every unsupported case falls back exactly to the
unchanged baseline.

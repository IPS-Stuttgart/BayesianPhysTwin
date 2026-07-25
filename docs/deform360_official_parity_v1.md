# Deform360 official 3D parity audit v1

## Purpose

Bayesian-PhysTwin has strong open Deform360 results in an online hidden-point
setting. Those results are not yet directly comparable to the published
Deform360 world-model tables. This audit prevents a candidate metric convention
from being presented as the official benchmark merely because its numbers look
favorable.

The public Deform360 paper reports aggregate future CD and track-error values:

| Setting | Method | Future CD | Future track error |
|---|---:|---:|---:|
| Per-episode | PhysTwin | 0.014 | 0.025 |
| Per-episode | ParticleFormer | 0.044 | 0.041 |
| Per-episode | PGND | 0.073 | 0.073 |
| Multi-episode | ParticleFormer | 0.051 | 0.079 |
| Multi-episode | PGND | 0.130 | 0.144 |
| Multi-object | ParticleFormer | 0.038 | 0.048 |
| Multi-object | PGND | 0.429 | 0.320 |

These are paper results, not locally reproduced values.

## Public evidence

The audit binds:

- Deform360 arXiv source `2607.05390v1`, source-tar SHA-256
  `66d2bfecd6ec9b829cd810913238821adc143c4831393704ed4bcc4ccc09e05c`;
- public Deform360 repository commit
  `0fe36f0b7a7a917ba62b5f8cee707299a9a4a317`;
- PGRD commit `e294d96723054f77a1cfdd3c2c052de7b7cd9ce3` and its exact
  metric-source hashes.

The paper defines the three evaluation settings and says that CD and
mean-squared track error are used. The public Deform360 repository currently
contains dataset and processing code, but no benchmark evaluator or split
manifest. The PGRD source implements one-sided, unsquared Euclidean Chamfer
from prediction to ground truth and coordinate-wise MSE for aligned particles.
That is a useful candidate convention, not proof of the Deform360 table
contract.

## Missing authoritative contract

An official 3D comparison still needs all of the following from a released
evaluator or content-hashed author confirmation:

1. exact training and evaluation object/episode manifests;
2. exact future-frame indices or per-case split manifest;
3. particle identity, resampling, and alignment rules;
4. prediction and ground-truth preprocessing;
5. validity and visibility masks;
6. coordinate frame, units, and normalization;
7. Chamfer direction, distance power, and reduction;
8. track correspondence, distance, and reduction;
9. frame, episode, and object aggregation order;
10. failed, missing, and unequal-length episode policy.

The CLI emits this list as a machine-readable information request:

```bash
bpt-audit-deform360-official-parity parity-audit.json
```

It exits with status 2 under `--require-ready`, because the current public
contract is deliberately incomplete.

## Claim boundary

Until the missing fields are authoritative:

- published table values may be quoted as results reported by Deform360;
- local scores may be reported under an explicitly named candidate convention;
- metric and aggregation sensitivity may be shown;
- no local score may be called an official Deform360 reproduction, direct
  leaderboard comparison, or state-of-the-art result.

This is not clerical caution. The included deterministic examples show that
one-sided versus symmetric Chamfer and frame-, episode-, versus object-balanced
aggregation can produce materially different values from the same predictions.

## Next decision

The shortest path to a defensible SOTA comparison is to obtain the evaluator or
an author-confirmed contract, then freeze a fresh object/episode cohort before
running Bayesian-PhysTwin. If that contract cannot be obtained, the paper
should retain the stronger but distinct claim: prospective cross-object online
hidden-point improvement under the repository's own fully specified protocol.

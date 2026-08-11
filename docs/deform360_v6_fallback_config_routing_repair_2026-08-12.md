# Deform360 v6 fallback-config routing repair

## Retained failure

Protected-main run `31543722289` used source revision
`5b2aadb87466381633844af52d3014764b301980`. The repaired selector and
precompiled frame-zero CUDA runtime both activated, and Splatfacto completed
500 iterations. The first registered source case then stopped before any
physical manifest or prediction seal with `fallback source config changed`.

The bounded artifact is `9121774996`, with digest
`sha256:9b8cb6149846328939c87242f0bd39ae5728c98221d5c8545a68e5da4fe8e8fb`
and execution receipt
`238f7a42ae607991ff8a166c99bb06fd789df40604e26a07ef8c02d1db66f5a6`.
It records zero physical manifests, zero source prediction seals, and false for
every suffix, confirmation, target, replacement, and claim authorization flag.

## Diagnosis

The frozen source runner supplied
`configs/sota/deform360_reconstruction_failure_persistence_fallback_v1.json`.
That file is a post-open integration report. Its canonical configuration hash
is `bf483664...aa50`, and it has no initializer `method` block.

The frame-zero loader is intentionally pinned to
`configs/sota/deform360_frame_zero_initializer_source_v1.json`. Its canonical
configuration hash is `64f72fe9...c759`, exactly matching the existing loader
constant, and it contains the frozen initializer method. Updating the loader
digest to accept the integration report would therefore be incorrect and would
only defer failure to the missing method block.

## Repair boundary

The dual-runtime dispatcher now rewrites exactly one argument only when all of
the following match:

- the frozen physical-stage entrypoint;
- stage `frame-zero`;
- flag `--persistence-fallback-source-config`;
- the retained failure's integration-report path.

It replaces that path with the already-frozen initializer specification after
checking both files' exact byte hashes. Missing, duplicate, or changed bindings
fail closed. A separate activation marker is incorporated into the compact
execution receipt.

This changes no candidate, cohort, initializer algorithm, physical algorithm,
mean, covariance, loss, horizon, selector, suffix policy, target policy, or
loader digest. It authorizes one reviewed protected-main source execution only;
it does not authorize a scientific result or any outcome access.

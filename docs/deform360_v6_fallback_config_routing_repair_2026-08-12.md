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

## Latest protected-main corroboration

Protected-main single-runtime run `31579234343` used source revision
`ab3e5cbedea7c163b8f340d6cc7b858dc5edde53`. Its checksum-pinned GNU 11,
CUDA 12.1, gsplat, and Ninja runtime probes passed. For the first registered
source case, Splatfacto again completed all 500 iterations and the Gaussian
export completed before the frame-zero loader rejected the same integration
report with `fallback source config changed`.

The bounded artifact is `9134627670`, with digest
`sha256:de3092dc213ec04836e8856362f300f99f90a2293e2bcfda4c2c1c3e4071b625`
and execution receipt
`dcf59f6a453bf7d53512479fac5ea95aa4fd98fa7ecd16414e0c2b188a4c2b0b`.
It records zero physical manifests, zero source prediction seals, and false for
every suffix, confirmation, target, replacement, and claim authorization flag.
This independently confirms that the remaining failure is argument routing,
not reconstruction, CUDA compilation, or the frozen frame-zero initializer.

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

The latest provenance update intentionally exercises the already-reviewed
protected-main dual-runtime workflow, whose dispatcher contains this exact
routing repair. The execution must still seal all ten physical manifests and
all 100 source predictions before any downstream evidence is admissible.

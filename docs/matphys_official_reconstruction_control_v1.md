# MatPhys all-frame reconstruction control v1

## Purpose

This control tests the current public MatPhys parameterization on its advertised
per-case, all-frame objective. It uses the already-open `single_lift_sloth`
interaction and fits all 85 released frames for a fixed 200 epochs.

This is deliberately not a prediction experiment. The checkpoint sees future
RGB frames and its Warp loss sees future geometry and track targets from the
same sequence. Every artifact therefore records
`future_observations_used=true` and `predictive_use_authorized=false`.

## Public-artifact boundary

The public MatPhys repository does not contain the final per-case
`train_ready.pt` files used by its current recipe. The control uses the existing
deterministic `causal-dino-graph-voronoi-parts-v1` proxy to provide graph-part
assignments and material rows. This is a Bayesian-PhysTwin extension, not an
official MatPhys preprocessing artifact. `node_sem.npz` is also supplied by the
proxy, although the released simple model does not consume node semantics in
its forward pass.

## Frozen execution

- MatPhys commit: `c16b858dfb79bf21024ead24b45a710600de7b4f`
- case: `single_lift_sloth`
- numeric video sampling: 16 uniform frames from all 85 frames
- objective frames: `[1,85)`
- terminal checkpoint: epoch 200, selected without metric-dependent extension
- track / geometry / render weights: `1 / 1 / 0`
- acceleration smoothness weight: `0.01`
- seed: `42`
- optimizer guard: transactional finite AdamW

The export writes the full MatPhys trajectory, complete object-plus-controller
spring field, seven global contact/damping parameters, official interval
metrics, and SHA-256 identities. The causal MatPhys audit validator rejects this
checkpoint by construction.

## Interpretation

A pass establishes parameterization capacity and a usable backend export. It
does not establish forecasting ability. A subsequent predictive method must be
trained only on registered source objects or inferred from an allowed target
prefix, then pass a source gate before any fresh independent evaluation.

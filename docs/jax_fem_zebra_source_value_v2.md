# JAX-FEM finite-deformation source-value gate v2

**Status:** frozen source-physical rejection; no outcome partition opened.

## Question

Does the source-qualified stable-Neo-Hookean JAX-FEM v2 arm add predictive
value on the two registered PhysTwin source actions without changing anything
in response to their object observations?

This is an already-open source-value gate, not fresh-object or target evidence.
A pass may justify an independently registered untouched evaluation. A failure
retains the incumbent byte-for-byte.

## Exact inheritance

The protocol at SHA-256
`4614dc7e6b550321c77572dfa5b88b1e6cbb1583c7fc98d322f6887078d94785`
copies the v1 source-value roster without retuning:

- the same lift and stretch source inputs and observation hashes;
- the same incumbent and MatPhys comparators;
- Poisson ratios `[0.20, 0.35, 0.45]` with equal weights;
- Young's modulus `100 kPa`;
- the same equal-event point and marginal-energy scores;
- the same 2/3 prefix and 1/3 future split;
- the same physical and value thresholds; and
- the same byte-exact fallback and no-replacement policy.

The only model change relative to the rejected v1 arm is the finite-deformation
runtime qualified by artifact ID
`820df616afcd911af2999aa3b208f8d2da1e2acbe62521bc9d1980fc317aba50`.

## Information order

1. Generate all 768 native predictions from frame-zero geometry and the known
   controller trajectory.
2. Seal every member archive, ensemble mean, hash, and physical diagnostic with
   `prefix_outcomes_read=false` and `future_outcomes_read=false`.
3. Run the full-horizon contact, displacement, and determinant gate without
   opening either outcome partition.
4. Open the prefix observations exactly once only if that physical gate passes.
5. Freeze the source decision. Open future observations exactly once only if
   the prefix value gate authorizes them.
6. Retain the result, positive or negative, without retry or parameter change.

Target, reserve, DLO4/DLO5, and held-v8 artifacts are outside every stage.

## Frozen result

The sole registered execution used clean revision
`083890e77c50d33c8b9ec047d4ce1dd2f2591013` and exact source archive
`2f2d0a716df49abb31acb43fa5cc6fcb65790dd5abe2ff51e615137b8b1de5e0`.
It completed 217 of the registered 768 native solves. All three lift members
and their ensemble mean sealed, then the first stretch member stopped because
the finite-deformation continuation violated its predeclared hard orientation
threshold. No prediction grid was published.

This is a source-physical rejection, not a missing-dependency or scheduler
failure. Prefix and future outcomes remained unopened, so neither source-value
scoring nor an untouched evaluation is authorized. The exact incumbent
fallback is retained without retry or parameter change. The compact receipt is
[`results/sota/diagnostics/jax_fem_zebra_source_value_v2/failure.json`](../results/sota/diagnostics/jax_fem_zebra_source_value_v2/failure.json); it binds the
remote log and partial archives by SHA-256 without publishing their payloads.

## Full-horizon physical gate

Before any outcome is opened, every ensemble member must have deformation
determinants in `[0.5, 2.0]`, maximum source-node displacement at most `0.35 m`,
and contact-projection error at most `0.02 m`. The native solver also retains
its stricter fail-closed hard orientation floor during continuation.

## Value gate

On the prefix partition, the equal-group point and energy ratios versus
persistence must each be at most `0.95`; the worst-group point ratio must be at
most `1.0`. Identity and Chamfer ratios versus the incumbent must each be at
most `1.05`. Final ensemble spread must lie in `[1e-5, 0.1] m`. These are the
unchanged v1 thresholds, not values selected after the v2 qualification.

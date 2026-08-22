# Native PhysTwin MatPhys source qualification v1

## Status

The source gate has completed and failed. See the
[frozen result](matphys_native_phystwin_source_v1_result.md). No fresh target
evaluation is authorized.

## Question

The prior Deform360 endpoint could not test MatPhys covariance cleanly: graph
support was sparse and the custom surface endpoint mixed reconstruction error
with physical-model error. This source-only study instead asks the same narrow
question on the 11 released additional PhysTwin interactions, where material
point identities, the fitted simulator graph, and the held-out future split are
all explicit:

> Does target-excluded MatPhys model-family disagreement improve predictive
> NLL and calibrated coverage around an unchanged released PhysTwin mean?

All 11 interactions were already opened by earlier project work. This is a
development qualification, not fresh evidence.

## Native interface

The released checkpoint contains a heterogeneous spring field in exact
object-spring-then-controller-spring order. The first MatPhys smoke accepted
only a scalar incumbent. The native path instead:

1. reconstructs the official radius-neighbour graph from `final_data.pkl` and
   `optimal_params.pkl`;
2. requires exact agreement with the checkpoint's object and total spring
   counts;
3. exports the first `num_object_springs` float32 values as the MatPhys
   incumbent, preserving byte order;
4. keeps the controller-spring values unchanged during every replay; and
5. verifies that a zero-strength proposal returns the incumbent array without
   arithmetic.

This is opt-in. The scalar compatibility input and every existing DEFORM result
remain unchanged.

The material proposal uses four uniformly spaced causal DINO keyframes from the
released training prefix, all three calibrated RGB-D views, graph-connected
five-part Voronoi regions, and MatPhys's cloth class. No future image is decoded
by the preparer or fold producer.

## Covariance around an unchanged mean

Let `mu_0` be the mean of four incumbent Warp replays and `X_j` the single
trajectory from target-excluded fold `j`. The point forecast remains the exact
released `inference.pkl` array. The raw covariance donor is

```text
C_raw = (1 / 11) sum_j (X_j - mu_0)(X_j - mu_0)^T + C_replay.
```

This is a baseline-relative second moment, not covariance centered around the
unused MatPhys ensemble mean. Centering around the ensemble mean would erase a
physical displacement shared by all folds while still claiming that the point
mean is unchanged. `C_replay` is the empirical covariance of four incumbent
replays through the official atomic spring-force accumulation path and is used
as a shared numerical floor. A deterministic accumulation variant is not used:
it would make the registered floor identically zero and can diverge from the
released trajectory on long self-collision rollouts. This 4+11 replay design is
the registered source approximation; it does not claim that replay variance is
identical for every spring field.

## Scoring

For each interaction, 128 material identities are selected by deterministic
frame-zero farthest-point sampling seeded at node zero before future validity is
inspected. Future events are scored only where released visibility and
motion-validity flags are true. Aggregation is equal event within interaction,
then equal interaction.

For held-out interaction `s`, the other ten source interactions select

```text
Sigma_candidate = a^2 C_raw + sigma^2 I
Sigma_isotropic = sigma_iso^2 I
```

from the exact grids in
[`matphys_native_phystwin_source_v1.json`](../configs/sota/matphys_native_phystwin_source_v1.json)
by minimum equal-interaction Gaussian NLL. The held-out interaction then reports
NLL, 90% chi-square ellipsoid coverage, ellipsoid volume, and NEES. The
candidate and comparator always use the same released point mean.

## Advancement gate

Fresh evaluation is allowed only when all 11 interactions are accounted for
and scoreable with no retained native-parity failure, and the candidate
simultaneously achieves:

- at least 6/11 held-out case-level NLL wins;
- at least 0.05 nats/event equal-case NLL improvement over isotropic;
- 90% coverage between 80% and 98%; and
- mean 90% ellipsoid volume at most 95% of the isotropic comparator.

Failure closes this MatPhys covariance donor. It does not authorize another
endpoint, scale grid, feature path, or target inspection. Passing only permits a
separately locked genuinely fresh evaluation; it is not itself a paper claim.

## Evidence boundary

- no held-v8 artifact;
- no DLO4 or DLO5 artifact;
- no revoked 24-target covariance route;
- no old six-object reserve;
- no change to the selected DEFORM predictor; and
- no target outcome access before a later prediction seal.

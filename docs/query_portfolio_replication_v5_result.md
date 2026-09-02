# Prospective two-query portfolio certificate result

## Result

The complete prospective portfolio passed its registered joint gate. Both
deformable-control queries have a positive familywise-adjusted lower bound on
mean reward gain and a familywise-adjusted upper bound on baseline-relative
harm below the registered `0.05` budget. The resulting statement holds
simultaneously across the two-query portfolio at confidence at least `0.95`.

| Query | Worlds | Guard deployments | Exact fallbacks | Mean gain | 99.5% gain lower bound | Harmful worlds | 98% harm upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| DLO-Lab Wrapping | 320 | 270 | 50 | +0.00590771 | +0.00480501 | 1 | 0.01809391 |
| DLO-Lab Slingshot | 320 | 65 | 255 | +0.00633863 | +0.00382284 | 1 | 0.01809391 |

All 640 registered evaluation worlds completed ordinarily with no technical
replacement. Across the complete denominator, the policies used their
Bayesian actions in 335 worlds and returned the incumbent exactly in 305. The
aggregate counts are descriptive only: rewards, confidence bounds, and gates
were computed separately within each query and were never pooled across tasks.

## Stronger contribution

The earlier single-task result established that uncertainty could improve one
decision problem. This prospective replication establishes a stronger object:
a finite portfolio of query-specific simulator competence certificates with a
simultaneous error guarantee. The portfolio membership, fresh world seeds,
complete denominators, component confidence allocation, and fail-closed joint
rule were frozen before either portfolio outcome was opened.

The evidence supports the following claim:

> Bayesian uncertainty has measurable decision value across a preregistered
> finite portfolio of deformable-control queries when simulator authority is
> granted only by query-specific competence certificates with exact fallback.

This is not a claim that either simulator is globally accurate. It is evidence
that reward-relevant simulator competence can be certified locally enough to
improve decisions while controlling harm relative to an unchanged incumbent.

## Statistical interpretation

For each query, the registered procedure uses a one-sided `99.5%` bootstrap
lower bound for mean gain and a one-sided `98%` exact binomial upper bound for
harm probability. The total gain-family error allocation is `0.01`; the total
harm-family allocation is `0.04`. A union bound therefore gives at least `0.95`
simultaneous confidence without assuming independence between tasks or between
the value and harm statistics.

The joint claim fails if either component is partial, has a technical
replacement, has a nonpositive adjusted gain bound, or exceeds the harm budget.
No favorable component can be promoted alone under this protocol.

## Claim boundary

The result concerns two registered public-simulator decision queries. It does
not establish real-robot safety, arbitrary-query validity, globally calibrated
simulation, point-prediction state of the art, or automatic transfer to a new
task. Adding a third query requires a new preregistered family or a prospective
alpha-spending rule.

## Provenance

- Joint result artifact ID:
  `711ca7a97017a3661e16980bd64d5481e61b32e86a5db287dcae63ebf92d907f`
- Joint result file SHA-256:
  `c42a52825f74b27d51152049239099f879385f8b2a3b48e6c9e4c8739eb0f713`
- Wrapping result artifact ID:
  `6371c2c231e2ac325e1a721575ddd7f6d3c4b06e4700ce6f38359f98579da4c7`
- Wrapping result file SHA-256:
  `4ea4f11c55c6a0648ed1b932f4ab9c74bb5d706591bb301508715abcadd9c7fd`
- Wrapping component artifact ID:
  `6cb1267174929f0ecd69711ab95862a0600ffcbc1e7587cd52a890c86f76c903`
- Wrapping component file SHA-256:
  `dfcd61f2e55a74fb763826bbac4c3dfe9b7223905a035084ae93cc6d7651078b`
- Slingshot result artifact ID:
  `7f5622d544d2c8f14c054c28014686b1113af427cae4eb57ee4d92ac6f2cd52d`
- Slingshot component artifact ID:
  `c7556807bb9c0fb8a78c410be58f13a8ff40f56190a70332e41be80dd69d3c70`
- Frozen portfolio protocol ID:
  `d1c5f9a7b52281d0762b597f2cb3143891b63f0d063093a6b2706a808a9f9ed6`
- Portable Wrapping replay repair revision: `7c90238986d0e55e68b384920d1cef2a3d35eae1`

Both portable component records reproduce from their sealed world-level
decisions and rewards. The joint artifact is a deterministic assembly of those
two complete records.

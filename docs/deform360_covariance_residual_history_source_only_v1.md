# Deform360 source-only residual-history contract v1

This experimental contract isolates the reusable source-side part of the
covariance-only validation design. It contains **no target roster**, does not
select or replace target objects, and does not authorize target payload access,
prediction, scoring, or claim promotion.

## Frozen semantics

For each opened source object/session, the adapter stores an exact residual
history with shape `(T, N, 3)` and an explicit Boolean validity mask with shape
`(T, N)`. Invalid entries are stored as zero only; temporal, spatial,
nearest-neighbour, camera, and material-identity filling is forbidden.

The caller must provide the already registered `last_residual` future-mean array
as a C-contiguous `float64` object. The contract independently reconstructs the
causal last-valid residual for each material identity and requires byte-exact
agreement with that supplied mean. It never substitutes a newly synthesized
mean for the registered object. On acceptance, the covariance-only result and
its hybrid record retain the exact caller-owned mean object by identity.

Each material identity requires the frozen minimum of two valid causal prefix
observations. One unsupported material rejects the whole source unit and returns
the exact caller-owned physical mean and covariance objects. There is no
partial-material deployment or unsupported-material zero correction.

The covariance donor is hard-bound to `independent_endpoint_v1` and the
reference predictor to `last_residual`. The early/middle/late scale schedule is
hard-bound to `[8, 16, 16]`; these values are not caller-selectable.

## Camera and reconstruction provenance

Camera names are not heuristically parsed into recorder identities. A
content-addressed source-inventory map explicitly binds every camera to one
physical recorder family. The deterministic provider/scoring split assigns
complete recorder families, exhausts the frozen map, and rejects any family that
crosses roles.

Provider and scoring reconstructions use separate content-addressed manifests.
Each manifest binds its role, camera set, source inventory, implementation
revision, configuration, input source artifacts, reconstruction artifact, and
parent reconstruction lineage. The contract rejects shared input bytes and any
overlap between provider and scoring reconstruction lineages.

## Registered-study binding

The owning prospective study is issue `#461`. Its paper-side native protocol is
`preregistrations/deform360_covariance_only_confirmation_v1.json`, with protocol
ID `fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3`
and selection-artifact SHA-256
`dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82`.
The registered information order requires 100 sealed source prediction records
and a source-positive authorization before any of the twelve disjoint
confirmation outcomes may be opened. This module implements only the source-side
contract and grants no authorization token.

## Evidence boundary

Passing the contract tests establishes deterministic construction, explicit
missingness, provenance-separated reconstruction, exact registered-mean
verification and identity, hard-bound donor/scales, per-material admission, and
exact whole-case fallback. It is implementation evidence only. It does not
establish fresh-object calibration, target accuracy, provider competence,
physical-query benefit, intervention benefit, deployment safety, or state of
the art.

The registered fresh study remains governed by issue `#461`, including its
separate source-first information order and sealed target cohort. This module
must not be used to revive the abandoned 24-target draft.

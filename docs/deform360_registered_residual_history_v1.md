# Registered Deform360 residual-history path v1

## Purpose

`bayesian_phystwin.deform360_registered_residual_history_v1` is the single
source-side implementation for the frozen covariance-only candidate in the
object-disjoint Deform360 validation protocol.

It answers one implementation question on an already opened source object/session:

> Can the registered `independent_endpoint_v1` covariance be reconstructed from
> an explicit residual history and attached to the exact caller-owned
> `last_residual` mean without exposing any donor, horizon, bin, or scale
> selection surface?

The module does not contain a target roster, target path, target-opening command,
scoring routine, or claim decision.

## One execution route

The only execution function is:

```python
run_registered_residual_history_v1(...)
```

The caller supplies:

- a physical prefix with shape `(T, N, 3)`;
- provider observations with the same shape;
- an explicit Boolean validity mask `(T, N)`;
- the physical future baseline `(H, N, 3)`;
- the exact caller-owned registered `last_residual` future mean;
- the exact caller-owned zero-covariance reference; and
- one content-addressed source-provenance record.

The caller cannot supply a covariance donor, forecast horizons, horizon bins, or
covariance scales.

## Frozen construction

Invalid provider rows must be explicit `NaN` rows. The stored residual history is
zero only at those invalid rows; temporal, spatial, nearest-neighbour, camera, or
material filling is not performed.

Every track requires at least two valid causal prefix observations. The module
reconstructs the last valid residual per track and requires byte-identical
agreement between

```text
physical future + last valid residual
```

and the supplied registered mean.

On acceptance, it:

1. runs `infer_model_averaged_endpoint` with the frozen default v1 component
   family over the complete causal prefix;
2. predicts exactly the consecutive horizons `1, ..., H`;
3. assigns future frames to the canonical equal three-way early/middle/late
   partition produced by `numpy.array_split`;
4. applies the frozen scales `[8, 16, 16]`; and
5. calls `compose_covariance_only_hybrid`, which retains the exact mean object by
   identity and creates an immutable covariance.

## Exact fallback

If any track has insufficient support or the reconstructed mean differs from the
registered mean, the complete source unit falls back. The returned objects are
exactly the caller-owned registered `last_residual` mean and zero-covariance
reference. There is no partial-track deployment, donor substitution, retuning,
or copied fallback object.

Malformed arrays, non-explicit missingness, a nonzero reference covariance, or
inconsistent reconstruction provenance are contract errors rather than tunable
fallback cases.

## Source provenance

`ResidualHistorySourceProvenanceV1` binds:

- the source inventory;
- separate provider and scoring reconstruction identities;
- exact implementation revisions and configuration identities;
- disjoint recorder-family sets; and
- disjoint input-artifact byte identities.

The execution decision additionally binds residual and validity bytes, support
counts, reconstructed and registered means, the endpoint configuration and
posterior, every consecutive-horizon prediction, the unscaled donor covariance,
the canonical scale schedule, output covariance, and covariance-composition
artifact.

## Static typing and review boundary

The module is part of the repository's changed-source MyPy ratchet. Generated
NumPy horizon arrays carry explicit `np.ndarray` annotations, and digest-field
iteration uses names distinct from integer support-count variables. These are
static narrowings only; they do not change array values, ordering, content
identities, admission, or fallback behavior.

The permanent change contains only the implementation, focused contract tests,
this documentation, and its source-distribution manifest entry. It contains no
write-enabled workflow, target execution helper, or second residual-history
route.

## Boundary

Passing the focused tests establishes deterministic construction, explicit
missingness, source-reconstruction separation, exact registered-mean identity,
internal donor reproduction, canonical horizons and scales, content-addressed
lineage, and exact whole-unit fallback.

It does not establish provider competence on real data, calibrated target
uncertainty, independent transfer, physical-state identification, Causal4D
intervention benefit, deployment safety, or state of the art. The target-closed
protocol and paper-side preregistration remain authoritative for any later
execution or claim.

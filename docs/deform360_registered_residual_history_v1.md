# Registered Deform360 residual-history path v1

## Purpose

`bayesian_phystwin.deform360_registered_residual_history_v1` is the single
source-side implementation for the frozen covariance-only candidate in the
object-disjoint Deform360 validation protocol.

It answers one implementation question on an already opened source
object/session:

> Can the registered `independent_endpoint_v1` covariance be reconstructed from
> an explicit causal residual history and attached to the exact caller-owned
> `last_residual` mean without exposing any donor, horizon, bin, or scale
> selection surface?

The module contains no target roster, target path, target-opening command,
scoring routine, or claim decision.

## One execution route

The only execution function is:

```python
run_registered_residual_history_v1(...)
```

The caller supplies:

- a finite physical prefix with shape `(T, N, 3)`;
- provider observations with the same shape;
- an explicit Boolean validity mask `(T, N)`;
- the physical future baseline `(H, N, 3)`;
- the exact caller-owned registered `last_residual` future mean;
- the exact caller-owned zero-covariance reference; and
- one content-addressed source-provenance record.

The caller cannot supply a covariance donor, forecast horizons, horizon bins,
or covariance scales.

## Frozen construction

Invalid provider rows must be explicit `NaN` rows. The stored residual history
is zero only at those invalid rows; temporal, spatial, nearest-neighbour,
camera, or material filling is not performed.

Every track requires at least two valid causal observations. The module
reconstructs the last valid residual per track and requires byte-identical
agreement between

```text
physical future + last valid residual
```

and the supplied registered mean.

On admission, it:

1. runs `infer_model_averaged_endpoint` with the exact
   `DEFAULT_MODEL_AVERAGED_ENDPOINT_CONFIG_V1`;
2. predicts the consecutive horizons `1, ..., H`;
3. assigns future frames to the deterministic equal early/middle/late
   `numpy.array_split` partition;
4. applies the frozen scales `[8, 16, 16]`; and
5. calls `compose_covariance_only_hybrid`, retaining the exact mean object by
   identity.

The decision binds the endpoint-model and fixed-anchor contract versions, the
complete frozen configuration identity, the posterior identity, every
consecutive-horizon prediction identity, the unscaled donor covariance, the
canonical scale schedule, and the output covariance.

## Exact fallback

The complete source unit returns the exact caller-owned registered
`last_residual` mean and zero-covariance reference when:

- any track has insufficient causal support;
- the reconstructed mean differs from the registered mean; or
- endpoint inference, horizon propagation, or covariance composition is
  rejected.

A donor failure before a complete donor exists records no donor execution. A
composition failure after donor construction retains the complete donor
lineage, while still returning the exact zero-covariance reference. Partial
donor provenance is invalid. There is no partial-track deployment, donor
substitution, retuning, or copied fallback object.

Malformed arrays, non-explicit missingness, a nonzero reference covariance, or
inconsistent source provenance are structural contract errors rather than
fallback cases.

## Source provenance

`ResidualHistorySourceProvenanceV1` binds:

- the source inventory;
- separate provider and scoring reconstruction identities;
- exact implementation revisions and configuration identities;
- disjoint recorder-family sets;
- disjoint input-artifact byte identities; and
- disjoint reconstruction ancestry, including each reconstruction and its
  declared parent reconstruction identities.

The execution decision additionally binds residual and validity bytes, support
counts, reconstructed and registered means, the canonical horizons and scales,
and the accepted or fallback disposition. All identities are content addressed
and metadata is recursively immutable.

## Boundary

Passing the focused tests establishes deterministic construction, explicit
missingness, reconstruction separation, exact registered-mean identity,
internal donor reproduction, canonical horizons and scales, fail-closed donor
handling, content-addressed lineage, and exact whole-unit fallback.

It does not establish provider competence on real data, calibrated target
uncertainty, independent transfer, physical-state identification, Causal4D
intervention benefit, deployment safety, or state of the art. The target-closed
software protocol and paper-side preregistration remain authoritative for any
later execution or claim.

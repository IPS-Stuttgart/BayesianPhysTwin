# Provider-to-physical mapping audit v1

A probabilistic 4-D provider can be technically competent while still being
unusable for a particular physical query. Common causes include a wrong frame,
an undeclared unit scale, no spatial overlap with the physical support region,
a timestamp mismatch, or structurally invalid covariance. Those failures must
be detected before they reach Bayesian inference and must not be confused with
poor estimator performance.

`provider_physical_mapping_audit_v1` adds a target-blind, fail-closed certificate
for this boundary. The module is experimental and intentionally uses a direct
import rather than the stable package-root or `v1` API.

## Contract

`ProviderPhysicalMappingCaseV1` binds one immutable provider artifact to one
immutable physical query and mapping protocol. It records:

- exact provider, physical-query, and protocol identities;
- source and destination frame names;
- provider points and a strict validity mask;
- an explicit native-unit-to-meter scale;
- a homogeneous provider-to-physical transform;
- one physical-query axis-aligned support box;
- optional paired point timestamps and a query time window;
- optional per-point covariance; and
- finite, immutable metadata.

All arrays are copied into immutable byte-backed storage. Their shape, canonical
dtype, and SHA-256 digest participate in the case identity.

`ProviderPhysicalMappingPolicyV1` freezes support thresholds and numerical
checks. The audit verifies:

1. rigid homogeneous-transform validity;
2. finite declared-valid provider points;
3. finite transformed points in meters;
4. required timestamp availability and finiteness;
5. required covariance availability, finiteness, symmetry, eigenvalues, and an
   optional condition-number ceiling;
6. minimum provider-valid point support; and
7. minimum spatial and temporal overlap with the physical query.

The result separates three decisions:

- `technical_valid`: the frame, values, timestamps, and covariance satisfy the
  frozen numerical contract;
- `provider_support_complete`: enough provider points are declared valid; and
- `query_support_sufficient`: enough declared-valid points map into the frozen
  physical query.

`mapping_admissible` is true only when all three decisions pass. Rejection
reasons are deterministic and use a fixed precedence order. The complete audit
is content-addressed.

## Minimal reproduction

From an editable checkout:

```bash
python examples/provider_physical_mapping_audit_v1.py \
  > provider-physical-mapping-audit.json
```

The example is synthetic and source-only. It prints one canonical finite JSON
certificate with all identities, accounting, bounding boxes, time diagnostics,
covariance diagnostics, the decision, and the scientific boundary.

A real provider adapter should construct the case from its already-frozen
artifact and protocol rather than changing thresholds after inspecting physical
or target outcomes.

## Failure attribution integration

`ProviderPhysicalMappingAuditV1.provider_failure_signal_patch()` exposes only
the two existing diagnostic signals owned by this audit:

```python
signals.update(audit.provider_failure_signal_patch())
```

A technical contract failure maps to `technical_valid=False`. Insufficient
provider support or insufficient query overlap maps to
`provider_support_complete=False`. The method does not set provider competence,
query identifiability, covariance calibration, physical-guard, or target-access
signals.

## Statistical and scientific boundary

This certificate is a contract and geometry precondition, not an accuracy
result. Passing does not establish:

- provider competence or association quality;
- calibrated provider or posterior covariance;
- improved physical state, parameter, or trajectory estimation;
- unseen-object transfer;
- Causal4D intervention benefit;
- deployment safety; or
- state of the art.

Points, frames, views, and tracks remain nested observations, not independent
experimental units. Independent object or session groups are still required for
scientific inference. The audit must not open sealed target outcomes and cannot
by itself authorize a Bayesian update or replace the caller-owned exact physical
fallback.

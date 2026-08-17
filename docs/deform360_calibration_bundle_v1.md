# Deform360 calibration bundle v1

## Purpose

The official-Hub Deform360 Stage-0 lock fixes the exact public dataset revision
and a fresh metadata-only cohort before camera, tactile, robot, geometry, or
target payload access. The next information boundary is to process calibration
objects, freeze every method choice, and prove that the confirmation cohort was
still unopened when those choices were sealed.

`bayesian_phystwin.deform360_calibration_bundle` defines that boundary. It does
not download data, fit a contact mapping, or open confirmation payloads. It
provides the portable artifact that a calibration workflow must publish before a
confirmation workflow can proceed.

## Exact cohort units

Each `Deform360CohortUnitV1` binds one selected object and episode to:

- the `sheet` or `volumetric` stratum;
- the exact `raw/<object>/metadata.json` path; and
- the SHA-256 of that metadata file.

The complete bundle requires the locked Stage-0 design:

- five calibration objects per stratum;
- six confirmation objects per stratum;
- no repeated object or unit; and
- no calibration/confirmation overlap.

Input ordering does not affect the bundle identity.

## Required calibration artifacts

The bundle contains exactly one `Deform360CalibrationArtifactRefV1` for each
registered calibration decision:

1. `contact_feature_and_grouping`;
2. `contact_linearization_and_covariance`;
3. `anchor_bias_prior`;
4. `visual_reliability_and_gauge`;
5. `normalized_evidence`;
6. `physical_response_and_closure`;
7. `regret_guard`; and
8. `conformal_interval`.

Each reference retains:

- the selected artifact and exact implementation revision;
- the complete calibration-selection evidence identity;
- selected candidate and candidate count;
- every calibration object used as an independent group;
- source-file SHA-256 values; and
- explicit declarations that no target outcome or confirmation payload was used.

A bundle fails closed if a role is missing or duplicated, if one selected
artifact omits a calibration object, or if any calibration choice reports target
or confirmation access.

## Evidence ownership

The bundle binds an `EvidenceUseLedgerV1` identity. The calibration workflow
should record every visual, tactile, robot-state, force, and derived factor used
while selecting the eight artifacts. This prevents the same raw factor from
later entering BayesianPhysTwin state inference and Causal4D intervention
inference as two apparently independent measurements.

The bundle also retains SHA-256 values for the protocol, Stage-0 lock,
preprocessing code, calibration reports, selection ledgers, and other files
needed to reconstruct the seal.

## Creating the seal

```python
from bayesian_phystwin.deform360_calibration_bundle import (
    Deform360CalibrationBundleV1,
    save_deform360_calibration_bundle,
)

bundle = Deform360CalibrationBundleV1(
    selection_artifact_sha256=selection_artifact_sha256,
    content_selection_sha256=content_selection_sha256,
    dataset_revision=dataset_revision,
    processing_revision=processing_revision,
    implementation_revision=implementation_revision,
    calibration_units=calibration_units,
    confirmation_units=confirmation_units,
    calibration_artifacts=calibration_artifacts,
    evidence_use_ledger_id=calibration_evidence_ledger.ledger_id,
    source_artifacts=source_artifacts,
)
save_deform360_calibration_bundle(
    bundle,
    "deform360-calibration-bundle-v1.json",
)
```

A valid bundle always records:

```text
status = sealed-before-confirmation-payload-access
confirmation_payload_opened = false
target_outcomes_used = false
replacement_allowed = false
```

It exposes a `confirmation_opening_token` derived from the exact bundle identity
and the exact ordered confirmation-unit identities.

## Confirmation gate

The confirmation runner must pin the reviewed bundle, Stage-0 selection, and
calibration evidence ledger before downloading or opening confirmation payloads:

```python
from bayesian_phystwin.deform360_calibration_bundle import (
    load_deform360_calibration_bundle,
    verify_deform360_confirmation_gate,
)

bundle = load_deform360_calibration_bundle(bundle_path)
token = verify_deform360_confirmation_gate(
    bundle,
    expected_bundle_id=registered_bundle_id,
    expected_selection_artifact_sha256=registered_selection_sha256,
    expected_evidence_use_ledger_id=registered_ledger_id,
)
```

The runner should record `token` in its own result manifest and stop before data
access if any identity differs. A changed candidate, calibration group,
preprocessing revision, evidence ledger, cohort unit, source file, or method
choice changes the bundle ID and the opening token.

## Persistence and validation

Loading independently revalidates every nested unit and calibration-artifact
identity. It rejects duplicate JSON keys, unknown or missing fields, non-finite
JSON, path traversal, coercible booleans or integers, malformed revisions or
digests, cohort overlap, incomplete strata, changed claim boundaries, and
content-ID tampering.

Saving uses a temporary file, `fsync`, and atomic replacement. Existing bundles
are not overwritten unless explicitly requested.

## Stage order

```text
Stage-0 metadata-only selection lock
-> download and process calibration objects only
-> retain complete calibration candidate/selection evidence
-> build evidence-use ledger
-> seal Deform360CalibrationBundleV1
-> review and pin bundle ID plus confirmation opening token
-> open confirmation payloads exactly once
-> publish the object-level positive or negative result
```

## Claim boundary

A valid bundle proves that the declared calibration choices and confirmation
cohort were sealed under the implemented contract. It is not an empirical model
result. It does not establish Deform360 accuracy, tactile benefit, calibrated raw
covariance, material-parameter identification, Causal4D benefit, or overall state
of the art.

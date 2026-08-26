# Deform360 covariance source-input inventory v1

## Purpose

The frozen covariance-only independent-validation protocol requires exactly
100 prefix-only source prediction records before its source authorization
decision can be evaluated. The predictor, source reducer, exact
`last_residual` mean, `independent_endpoint_v1` covariance donor, horizon
scales `[8, 16, 16]`, observation model, ten source object-sessions, and twelve
closed confirmation object-sessions are already fixed.

The remaining operational problem is narrower: identify the exact retained
source manifests, residual histories, endpoint beliefs, physical predictions,
and related inputs available on the authorized source-data runner without
entering any confirmation data.

## Authenticated issue command

Create this exact comment on
`IPS-Stuttgart/BayesianPhysTwin#775`:

```text
/bpt-inventory-covariance-source-v1
```

The command is admitted only when all of these conditions hold:

- the event is a newly created issue comment;
- the owning issue is exactly `#775`, not a pull request;
- both the event actor and comment author are exactly `FlorianPfaff`;
- the comment body is an exact byte-for-byte command match; and
- the job runs on the labelled `workstation2` source-data runner.

The workflow file is the existing
`deform360-covariance-only-independent-validation-v1.yml`; no additional
permanent workflow is installed.

## Filesystem and information boundary

The inventory may inspect only the retained calibration-source and
calibration-processed roots:

```text
/mnt/lexar4tb/datasets/deform360-official-hub-visuotactile-v1/calibration-source
/mnt/lexar4tb/datasets/deform360-official-hub-visuotactile-v1/calibration-processed
```

It resolves and explicitly forbids the separate confirmation root:

```text
/mnt/lexar4tb/datasets/deform360/adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370
```

The implementation never enters the confirmation root, skips symlinked
directories and files, and rejects any resolved path inside the forbidden
tree. It inventories:

- relative file paths and byte counts;
- suffix counts;
- selected top-level JSON keys and scalar identity/status fields;
- `.npy` header shape, dtype, and storage order; and
- `.npz` member names and embedded `.npy` headers.

It does not read NumPy array values, deserialize pickle or Torch payloads,
score source or target outcomes, generate a candidate, select a scale, or
construct confirmation predictions.

### Retained pre-inventory failure

Workflow run `32997229588` reached the inventory command on `workstation2` and
stopped before directory traversal because the workflow supplied the legacy
symlink spelling `deform360_official_hub_visuotactile_v1`. The inventory
validator correctly rejected it as noncanonical. The run created and uploaded
no inventory, posted no result comment, read no array values, and did not
dispatch or consume the one-attempt source producer.

The repaired workflow uses the underlying canonical directory spelling
`deform360-official-hub-visuotactile-v1` for both admitted roots. This changes
only filesystem path binding. It changes no roster, source bytes, prediction,
model, covariance, score, gate, or information boundary.

## Result artifact

The job uploads one content-addressed
`covariance-source-input-inventory.json`. The implementation derives the exact
ten object-session roster from the frozen selection lock and verifies it against
the registered source reducer; the workflow contains no independent roster
literal.

The record binds:

- the software and paper protocol identities;
- the paper/code cross-repository binding;
- the exact selection Git blob;
- the implementation revision;
- both admitted source roots; and
- every retained header record and source-unit path count.

Both records bind the exact default-branch workflow revision and state that:

```text
source_roots_only=true
confirmation_root_entered=false
target_outcomes_opened=false
array_values_read=false
file_payloads_scored=false
```

The artifact is retained for 90 days. It is an operational diagnostic, not a
claim-bearing evidence bundle.

## Prefix-only source producer

After the corrected inventory implementation and producer merge to protected
`main`, create this exact comment on issue `#775`:

```text
/bpt-produce-covariance-source-v1
```

The authenticated job recomputes the header-only inventory, binds the exact
successful public Deform360 v6 prefix run and all consumed source-prefix files,
then constructs the frozen covariance candidate. It publishes:

- ten deterministic unit archives and manifests;
- exactly 100 records in canonical outer-fold/source-unit order;
- the content-addressed source-prediction batch;
- a complete panel receipt; and
- the one-attempt ledger and execution log.

The candidate mean is the exact caller-owned `last_residual` array. The only
added quantity is `independent_endpoint_v1` covariance with fixed horizon
scales `[8, 16, 16]`, `5 mm` observation noise, and a `1e-12 m²` covariance
floor. Unsupported units retain the byte-exact comparator fallback.

The producer consumes one atomic, revision-specific attempt ledger before
source values are read. A failure retains a bounded technical receipt and does
not silently permit a retry, replacement, or partial barrier. The successful
path validates all 100 records and then stops. It does not attach source
suffixes, score outcomes, execute the reducer, create confirmation predictions,
or authorize confirmation access.

## Later decision stage

Only after a complete 100-record receipt has been independently rehashed may a
separately reviewed source-scoring execution attach the already-open source
suffix and run the frozen reducer exactly once. That reducer must return one of:

- `source-positive`;
- `source-negative`; or
- `source-technical-negative`.

Only `source-positive` may authorize the subsequent construction and sealing of
the twelve prefix-only confirmation predictions. It still does not authorize
opening any confirmation payload or outcome. A separately reviewed source
scorer now exists, but no empirical source-scoring or confirmation trigger is
installed by the producer change.

## Scientific boundary

This inventory does not establish source competence, covariance value,
independent-object transfer, calibrated target uncertainty, improved point
prediction, physical-state identification, real-provider competence,
Causal4D intervention benefit, deployment safety, or state of the art. It
changes no method, cohort, donor, scale, score, fallback, information order, or
claim.

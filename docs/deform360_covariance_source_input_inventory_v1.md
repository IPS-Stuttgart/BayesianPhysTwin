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
`IPS-Stuttgart/BayesianPhysTwin#461`:

```text
/bpt-inventory-covariance-source-v1
```

The command is admitted only when all of these conditions hold:

- the event is a newly created issue comment;
- the owning issue is exactly `#461`, not a pull request;
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
/mnt/lexar4tb/datasets/deform360/bpt-runner-local-science-f804696d7a13/calibration-source
/mnt/lexar4tb/datasets/deform360/bpt-runner-local-science-f804696d7a13/calibration-processed
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

## Result artifact

The job uploads:

- `covariance-source-input-inventory.json`, containing the complete bounded
  inventory and source-unit path counts; and
- `covariance-source-input-inventory.md`, containing the concise candidate
  path summary posted to issue `#461`.

Both records bind the exact default-branch workflow revision and state that:

```text
source_roots_only=true
confirmation_root_entered=false
target_outcomes_opened=false
array_values_read=false
file_payloads_scored=false
```

The artifact is retained for 30 days. It is an operational diagnostic, not a
claim-bearing evidence bundle.

## Next admissible action

Use the inventory to identify the exact current source-side inputs required by
the already frozen predictor. Then construct and seal the 100 source prediction
records before any source suffix scoring. The existing source reducer must
return exactly one of:

- `source-positive`;
- `source-negative`; or
- `source-technical-negative`.

Only `source-positive` may authorize construction and sealing of the twelve
prefix-only confirmation predictions. It still does not authorize opening any
confirmation payload or outcome.

## Scientific boundary

This inventory does not establish source competence, covariance value,
independent-object transfer, calibrated target uncertainty, improved point
prediction, physical-state identification, real-provider competence,
Causal4D intervention benefit, deployment safety, or state of the art. It
changes no method, cohort, donor, scale, score, fallback, information order, or
claim.

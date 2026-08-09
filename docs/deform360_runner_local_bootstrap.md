# Deform360 runner-local bootstrap

## Scope

`deform360-runner-local-science.yml` is a storage-aware preparation and
reproduction bootstrap for the sole Deform360 self-hosted runner. This bootstrap
is not a claim-bearing experiment. It does not authorize confirmation access and
must not
be used to replace the registered source-gate, independent-audit, observability,
or confirmation-opening chain.

The workflow is manually dispatchable only from trusted `main`. Pull requests
call the reusable hosted contract workflow
`deform360-runner-local-contracts.yml`; they never receive a self-hosted runner
or dataset access.

## Frozen runner inputs

The bootstrap recognizes two separately bounded roots:

- official/raw snapshot:
  `/mnt/lexar4tb/datasets/deform360/data-7fea8e2`;
- adaptive-confirmation download:
  `/mnt/lexar4tb/datasets/deform360/adaptive-confirmation-download-5a9c56d593462486bdd0953dcaf6f9c643bf8370`.

The official/raw snapshot is an older revision and is therefore only an exact
byte cache for the frozen calibration plan. A file is reusable only when its
registered path, size, and digest match. Reuse is copy-on-write; hard links and
writable aliases into the raw snapshot are forbidden. Missing or mismatching
files go through the existing exact-file downloader.

The adaptive-confirmation root is names-only in this bootstrap. Its payloads,
all confirmation objects, target outcomes, and future information remain closed.

## Runner admission

The scientific job requires all four labels:

```text
self-hosted, Linux, X64, nvidia-smi
```

It then checks `RUNNER_NAME=workstation2`, `RUNNER_OS=Linux`,
`RUNNER_ARCH=X64`, and a working `nvidia-smi` command before checking out source.
This keeps a generically labelled or newly registered runner from silently
executing the mounted-data workflow.

## Locked Python runtime

The Python 3.12 runtime is constrained by
`requirements/locks/deform360-runner-local-science-py312.txt`. The lock was
reconstructed from successful calibration-source workflow run `31236230283`,
artifact `9015548481`, whose artifact digest is
`sha256:53371f7459242a0f8c72cf8adc7b04254c90410fb426723402f009eebb0767dd`.
It binds the environment that successfully prepared all ten frozen calibration
objects on `workstation2` with Python 3.12.3 and processing revision
`d8522a4403b766aeb387510c04e89032a56fdf35`.

The bootstrap installs exact `pip`, `setuptools`, and `wheel` versions, disables
build isolation, applies the lock to every third-party dependency, and then
validates the isolated target site. An unlisted package, a version mismatch, a
missing locked package, or an absent local BayesianPhysTwin/Deform360 package is
a terminal runtime-contract failure.

## Plan-derived storage admission

After exact local-cache reuse and before network download,
`check_deform360_runner_capacity.py` derives mutable storage demand from the
sealed plan:

1. sum bytes of unique selected files still absent from the writable data root;
2. reserve the same missing-byte total for the Hugging Face cache because a
   download may temporarily exist in both locations;
3. estimate prepared output from the complete selected-byte total using the
   frozen multiplier; and
4. retain a fixed free-space reserve once per physical filesystem.

Roles that share a filesystem are combined by device ID. The compact
`storage-capacity.json` report records the plan digest, source-byte accounting,
per-device available and required bytes, and the closed information boundary.
Insufficient storage exits before download or preparation.

## Credential boundary

`HF_TOKEN` is not a job environment variable. It is exposed only to the exact
missing-file download step, together with Xet disablement. Checkout, runner
admission, package installation, inventories, local cache inspection,
preparation, summaries, and artifact upload do not receive the token.

## Downstream evidence chain

A successful bootstrap establishes only that frozen calibration source can be
prepared reproducibly from the protected runner. Claim-bearing work remains in
the registered chain documented by:

- [prepared-source inventory](deform360_calibration_prepared_inventory.md);
- [Prob4D source calibration](deform360_prob4d_source_calibration.md);
- [Prob4D source gate](deform360_prob4d_source_gate.md);
- [atomic calibration-observability batch](deform360_calibration_observability_batch.md);
- [confirmation-opening authorization](deform360_confirmation_opening_authorization_v1.md); and
- the independent audit and immutable result records linked from the issue and
  paper repositories.

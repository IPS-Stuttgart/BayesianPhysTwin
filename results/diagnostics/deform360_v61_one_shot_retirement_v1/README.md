# Deform360 v6.1 one-shot workflow retirement

This directory preserves the terminal identities for the two temporary
Deform360 v6.1 workflows tracked by issues #642 and #645. The executable YAML
files were removed after their one authorized runs completed their registered
information boundaries. Historical GitHub Actions runs and uploaded artifacts
remain available through the Actions registry.

## Candidate producer

Workflow run `31647329129`, attempt 1, completed successfully on source revision
`2eb8d12e2120d58d0d678c3771d29faaeb765497`. Its artifact
`deform360-v61-candidate-producer-31647329129-1` has archive digest
`03065bdecf9dd5906e70d5722f8d5f1608ae4968dccda585e13e732d5b7a9849`.

The sealed receipt records 100 candidate predictions, zero technical failures,
and no source-suffix opening. It authorizes neither scoring nor confirmation.

## Public-source scorer

Workflow run `31669176135`, attempt 1, ran on source revision
`74e556d6f9b503409f3b163ef27ccb7a17c61d85`. The workflow conclusion is
`failure` because its final fail-closed assertion rejected a non-success terminal
state. The evidence-producing and custody steps completed and uploaded artifact
`deform360-v61-source-scorer-endpoint-gsplat-base-31669176135-1`, whose archive
digest is
`b1a8ea2e4d3952af4b446fa4d420ea23f310b4e43396cc8505978302fcd4e42f`.

The retained receipt is
`f284be9c6a83afe5688030cfec466f0bbe2f2a24d7ce0aa13eac272d9763742c`.
It records a terminal endpoint-processing technical failure after the public
source suffix was opened. The source gate was not evaluated, replacement and
continuation are forbidden, and no confirmation payload, target outcome, or
held-v8 artifact was opened.

## Boundary

Deleting the workflow files prevents accidental reruns. It does not delete or
reinterpret the historical runs, alter any protocol or candidate, authorize
confirmation, or create scientific evidence. The machine-readable record is
content-addressed independently of this explanatory text.

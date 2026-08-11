# Deform360 covariance residual-history adapter dry run v1

## Purpose

This gate validates the interface needed by the custom fresh-object
covariance-only study. It does **not** inspect the quarantined 24-object target
payload. The dry run uses deterministic synthetic source arrays only and tests
whether a causal residual history can feed the frozen
`independent_endpoint_v1` covariance donor without changing the registered
`last_residual` point prediction.

The lock is
`protocols/locks/deform360_covariance_only_residual_adapter_dry_run_v1.json`.
Its content identity is
`33e58fd993b89042bf7a5549384725d375ed3885776577412ed73f0767e10706`.

The parent target roster remains the 24-session custom study on branch
`deform360-covariance-only-target-v1`. None of those objects has official
processed Deform360 annotations, so this gate does not establish official
benchmark parity.

## Adapter contract

`bayesian_phystwin.deform360_covariance_residual_adapter_v1` consumes:

- one finite `float64` residual history with shape `(T, N, 3)` in metres;
- one Boolean validity mask with shape `(T, N)`;
- strictly increasing causal-history and future frame identities;
- ordered material identities shared by history, physical fallback, registered
  mean, and covariance output;
- one caller-owned physical fallback mean `(H, N, 3)` and covariance
  `(H, N, 3, 3)`;
- one caller-owned registered `last_residual` mean;
- disjoint provider and scoring camera panels;
- disjoint provider and scoring source-artifact inventories; and
- distinct provider and scoring reconstruction artifacts.

Invalid residual entries must be exactly zero. They are never imputed, carried
forward, spatially nearest-filled, or exposed to the endpoint model. Each
material point uses only its own last valid residual. A point without admitted
history retains the physical mean and covariance exactly.

The accepted point prediction is checked byte-for-byte against

```text
physical_fallback_mean + each point's own last valid causal residual.
```

The adapter passes that exact caller-owned NumPy array to
`compose_covariance_only_hybrid`; the returned mean is therefore the same Python
object. The covariance donor is the source-frozen default
`ModelAveragedEndpointConfigV1`, propagated to each registered future frame and
scaled by the frozen covariance multipliers:

| Horizon | Covariance multiplier |
| --- | ---: |
| early | 8 |
| middle | 16 |
| late | 16 |

The multipliers apply to covariance, not standard deviation.

## Minimum observed support

The dry-run lock requires:

- at least two valid causal residual observations per admitted material point;
- all material points in the accepted synthetic fixture to meet that threshold;
- at least two provider cameras and two scoring cameras; and
- exact camera, artifact, and reconstruction disjointness.

The implementation also supports a separately frozen fraction below one. In
that case unsupported material points retain their physical mean and covariance
exactly. The present dry-run protocol deliberately uses a complete-support gate.
It does not choose a target-side threshold.

## Exact fallback

Ordinary provider rejection returns the caller-owned physical fallback mean and
covariance objects without copying. The dry run exercises four registered
fallbacks:

1. provider unavailable;
2. insufficient observed residual support;
3. provider/scoring camera overlap; and
4. a shared provider/scoring reconstruction artifact.

The module additionally returns the same exact fallback after a numerical
endpoint-covariance failure. Structural errors such as wrong units, coordinate
frame, material roster, frame chronology, non-PSD physical covariance, hidden
values under an invalid mask, or a non-identical registered mean fail closed
instead of being relabelled as ordinary provider failure.

## Camera and artifact separation

Provider and scoring cameras are compared by exact camera identity. Their sets
must be disjoint. The adapter also requires disjoint content-addressed source
artifact sets and different reconstruction-artifact identities. Renaming the
same bytes or using two views from one shared reconstruction cannot satisfy the
contract.

This separation is an interface property. It does not itself prove statistical
independence, camera calibration, provider competence, covariance calibration,
or physical benefit.

## Run the dry gate

From the repository root:

```bash
PYTHONPATH=src python \
  scripts/science/run_deform360_covariance_residual_adapter_dry_run_v1.py \
  /tmp/deform360-covariance-residual-adapter-dry-run-v1.json \
  --protocol \
  protocols/locks/deform360_covariance_only_residual_adapter_dry_run_v1.json
```

A successful result records:

- the accepted adapter record identity;
- byte-identical mean preservation;
- unchanged validity support and no nearest filling;
- camera, artifact, and reconstruction disjointness;
- one content-addressed receipt for every registered exact fallback; and
- explicit false values for target payload, media, array, prediction, outcome,
  and scoring access.

The deterministic reference result has ID
`14ee5b6899737295a4431e8562110618be591055638f70c609bd4fc87e6f456d`.
This is software-contract evidence only.

## Information boundary and next gate

The dry run must not access

```text
/mnt/lexar4tb/datasets/deform360/unopened-candidate-target/
  covariance-only-v1/payload
```

or any other target copy. Passing this gate does not authorize target decode,
prediction, or scoring. The next admissible step is a separately reviewed
source-only empirical execution on already-open source objects. That execution
must bind actual residual-history, provider-camera, scoring-camera, material,
frame, and reconstruction identities to this interface and must preserve the
same exact fallback behavior. A source-only success would still require a
separate authorization before any quarantined target payload is opened.

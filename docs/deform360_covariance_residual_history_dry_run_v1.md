# Deform360 covariance residual-history dry run v1

## Purpose

This gate tests the final adapter boundary before any fresh-target media,
sensor array, prediction, or outcome is opened. It runs only on already-opened
source objects or acquisition sessions and asks whether a causal visual prefix
can be represented as a material-identity-preserving residual history suitable
for the frozen covariance-only candidate.

A passing dry run is technical admission evidence only. It does not establish
fresh-object calibration, point accuracy, official Deform360 benchmark parity,
Prob4D or Causal4D benefit, deployment safety, or state of the art.

The frozen protocol is
`protocols/locks/deform360_covariance_residual_history_dry_run_v1.json`.

## Input contract

One complete opened source unit supplies a non-pickled NPZ archive with exactly:

- `physical_prefix_m`: finite `float64` array with shape `(T, N, 3)`;
- `provider_observation_prefix_m`: array with the same shape; only rows marked
  valid must be finite;
- `provider_validity`: Boolean array with shape `(T, N)`;
- `physical_future_m`: finite, C-contiguous `float64` array with shape
  `(H, N, 3)`;
- `physical_fallback_covariance_m2`: finite, symmetric PSD `float64` array with
  shape `(H, N, 3, 3)`;
- `donor_covariance_m2`: covariance donor with the same covariance shape;
- strictly increasing `frame_indices` with shape `(T,)`;
- unique integer `material_ids` with shape `(N,)`; and
- `future_horizon_bins` with shape `(H,)`, using `0`, `1`, and `2` for early,
  middle, and late horizons.

The companion source manifest binds the archive SHA-256, source-unit identity,
complete released camera roster, the exact provider and scoring camera rosters,
distinct provider and scoring reconstruction artifact identities, and the statement
that only opened source data were used. The declared role rosters must exactly match
the deterministic whole-recorder partition; distinct artifact IDs alone are not
accepted as proof of camera separation.

The manifest is an exact-field finite-JSON contract. A relative archive path is
resolved only against the manifest directory before the target-quarantine check,
so execution does not depend on the process working directory.

## Missingness and material identities

The adapter stores a residual only where `provider_validity` is true:

```text
residual[t, i] = provider_observation[t, i] - physical_prefix[t, i]
```

Every invalid storage row is exactly zero and has no semantic value outside its
validity bit. The adapter never applies temporal carry-forward, nearest-neighbor
filling, spatial interpolation, material-identity reassignment, or silent row
removal.

The deterministic comparator is evaluated separately from storage missingness.
For every material identity, it uses that material's last valid causal residual
anywhere in the opened prefix. Therefore an identity observed before, but not at,
the final prefix frame retains its last valid residual exactly as in the frozen
`last_residual` comparator. A material never observed in the prefix retains the
physical prediction. This lookup does not fill or alter the stored history.

## Minimum observed support

The frozen technical gate requires both:

- at least 9 observed material identities at the final prefix frame; and
- at least 50% of the declared material identities observed at that frame.

Both tests are conjunctive. A failure returns the exact caller-owned physical
future mean and covariance objects by identity.

## Disjoint provider and scoring cameras

Camera stream IDs are grouped by physical recorder family: the suffix
`_cam0`, `_cam1`, and so on is removed, and every stream from one recorder stays
in one role. Recorder families are hash-ranked under the frozen namespace
`deform360-provider-scoring-camera-family-v1` and greedily balanced between the
provider and scoring roles.

Each role must contain at least 8 camera streams and at least 4 physical
recorder families. The split is deterministic, input-order invariant,
exhaustive, and disjoint. The provider and scoring reconstruction artifact
SHA-256 identities must also differ. Thus two nominally disjoint camera sets
cannot silently consume one shared reconstruction artifact.

## Covariance-only candidate

The admitted reference mean is the exact per-material last-valid causal residual
mean. The covariance donor remains `independent_endpoint_v1`, scaled by the
frozen schedule:

```text
early   8.0
middle 16.0
late   16.0
```

`compose_covariance_only_hybrid` validates finite symmetric PSD covariance and
returns the caller-owned reference mean by object identity. The adapter adds no
observation noise; the separately frozen scoring/deployment model owns the
5 mm observation-noise floor.

If donor covariance, scale broadcasting, or covariance PSD validation fails,
the dry run records `covariance-contract-rejection` and returns the exact
physical mean and covariance objects. A malformed physical fallback covariance
is a protocol error and fails closed rather than manufacturing a replacement.

## Execution

```bash
python scripts/science/run_deform360_covariance_residual_history_dry_run_v1.py \
  /path/to/opened_source_manifest.json \
  /path/to/new_output_directory \
  --protocol \
    protocols/locks/deform360_covariance_residual_history_dry_run_v1.json \
  --implementation-revision "$(git rev-parse HEAD)"
```

The script refuses all input and output paths inside
`/mnt/lexar4tb/datasets/deform360/unopened-candidate-target/covariance-only-v1/payload`.
It loads NPZ data with `allow_pickle=False`, verifies the source archive digest,
rejects duplicate or non-finite JSON, publishes atomically into a new directory,
and refuses overwrite.

Outputs are:

- `dry_run_arrays.npz`, containing the deployed mean/covariance, residual
  history, exact validity, identities, frames, and disjoint camera rosters; and
- `dry_run_result.json`, containing content-addressed policy, camera partition,
  adapter, decision, source bindings, and information-boundary records.

A fallback is a valid completed dry-run result, not a reason to weaken the gate.
The fresh target remains closed unless the source study demonstrates adequate
support and all registered contracts pass without target-dependent changes.

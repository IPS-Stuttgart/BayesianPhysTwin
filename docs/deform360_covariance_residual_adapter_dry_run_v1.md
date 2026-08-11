# Deform360 covariance residual-history adapter dry run v1

## Purpose

This gate binds the already frozen Deform360 covariance-only target method to a
concrete source-side adapter before any target media, sensor array, prediction,
or outcome is opened. It answers only a technical question:

> Can a causal residual history with explicit missingness produce the registered
> `last_residual` mean and independent-endpoint covariance while preserving exact
> fallback and provider/scoring independence?

The gate is not a Deform360 target execution and does not establish official
benchmark parity. The selected 24-object cohort has no official processed
annotations, so any later empirical result remains a custom fresh-object
calibration study.

## Frozen parent method

The dry run inherits the method in
`protocols/locks/deform360_covariance_only_target_v1_2.json` without tuning:

- point mean: exact `last_residual` persistence;
- covariance donor: `independent_endpoint_v1` using the default
  `ModelAveragedEndpointConfigV1`;
- covariance multipliers: `8`, `16`, and `16` for early, middle, and late
  horizons;
- minimum support: three valid causal observations per track;
- unsupported-track fallback: exact unchanged physical future and zero donor
  covariance; and
- provider-failure fallback: the exact reference mean and zero donor covariance.

The multipliers apply to covariance, not standard deviation.

## Adapter contract

`deform360_covariance_residual_adapter_v1.py` requires:

- residuals in metres with shape `(T, N, 3)`;
- a Boolean validity mask with shape `(T, N)`;
- a float64, C-contiguous physical future in metres with shape `(H, N, 3)`;
- registered early/middle/late labels and strictly increasing horizon steps;
- explicit object, session, material, and coordinate-frame identities;
- nonempty, pairwise-disjoint provider and scoring camera sets; and
- nonempty, pairwise-disjoint provider and scoring reconstruction-artifact sets.

Invalid residual entries are never temporally or spatially filled. They are
canonicalized to zero only after the validity mask is copied and hashed, and they
remain invalid for endpoint inference. Tracks below the support threshold are
also removed from the covariance-provider input so they cannot influence a
custom cross-track provider.

For supported tracks, the point mean is the last valid residual added to the
untouched physical future. The existing covariance-only composer returns that
exact reference NumPy object and changes only covariance. Donor covariance is
required to be finite, symmetric, and positive semidefinite. Any provider error,
shape mismatch, nonfinite value, asymmetry, or non-PSD covariance fails closed to
zero covariance without changing the reference mean.

## Locked dry-run fixture

The source-only fixture contains eight prefix frames, five tracks, and twelve
future frames. Valid observation counts are `8`, `4`, `3`, `2`, and `1`, so the
registered threshold admits exactly three tracks and retains two exact physical
fallback tracks. The four required cases are:

1. ordinary endpoint-model covariance with structured missingness;
2. masked-value invariance, replacing every invalid value without changing the
   mask;
3. an injected provider failure; and
4. a completely unsupported history for which the provider must not be called.

The run independently recomputes the raw endpoint covariance, applies the frozen
horizon multipliers, checks positive semidefiniteness, and compares every output
byte required by the fallback and mean-identity contracts.

## Workflow boundary

`.github/workflows/deform360-covariance-residual-adapter-v1.yml` runs only on
GitHub-hosted `ubuntu-latest`. It has read-only repository permissions, no
secrets, no self-hosted runner label, and no reference to the quarantined target
root. Consequently, the workflow cannot access
`/mnt/lexar4tb/datasets/deform360/unopened-candidate-target/covariance-only-v1`.

The workflow performs focused Ruff, formatting, typing, and pytest checks; runs
the locked fixture; verifies the machine-readable information boundary; and
uploads only the compact source-only result and environment identity.

## What success authorizes

Success establishes the adapter contract and exact fallback implementation. It
permits planning a separate opened-source production stage that creates residual
histories from source data with disjoint camera and reconstruction partitions.
It does not authorize opening the 24-object target, selecting cameras on target,
fitting any target-derived parameter, scoring a target suffix, excluding a
technical failure, or making a paper claim.

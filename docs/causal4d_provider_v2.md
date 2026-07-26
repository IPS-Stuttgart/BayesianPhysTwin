# Causal4D provider API v2

`bayesian_phystwin.causal4d_provider_v2` is the supported typed boundary between
Bayesian-PhysTwin and Causal4D. The v1 module remains available for frozen
experiments, but new Causal4D core code should import only v2.

## Provider-owned values

The provider validates, copies, and freezes the arrays in:

- `PhysTwinCase`: released observations, validity masks, controller trajectories,
  auxiliary structure points, graph parameters, and an optional baseline;
- `PhysTwinSpringGraph`: vertices in metres, spring endpoints, rest lengths in
  metres, masses in kilograms, and the object/controller partition;
- `PhysTwinControllerLayout`: released one- or two-hand count and deterministic
  contiguous controller groups.

Causal4D therefore no longer needs to import graph construction or controller
partitioning from Bayesian-PhysTwin implementation modules.

## Legacy artifact loading

`load_official_phystwin_case()` is the schema-specific reader for the released
legacy pickles. Passing `expected_sha256` uses the digest-bound loader before
pickle deserialization. The trusted mapping must cover every requested input:
`final_data`, `optimal_params`, and, when present, `baseline_trajectory`.
Without those digests the function is retained only as a trusted-local
compatibility path; pickle is not a safe format for untrusted input. New
artifacts remain JSON/NPZ contracts.

## Semantic compatibility

`causal4d_provider_manifest()` reports API version 2, artifact versions,
capabilities, the installed package revision when available, and
`contract_sha256`. The fingerprint is computed from a canonical JSON descriptor
of provider-owned values and operations. Causal4D validates the exact fingerprint
in addition to package/API versions, capabilities, and artifact versions.

## Replay

Use `create_official_case_replay_provider()` with a validated `PhysTwinCase` and
`PhysTwinSpringGraph`. Conversion back to the released simulator graph happens
inside Bayesian-PhysTwin; Causal4D receives only the `PhysTwinReplayProvider`
protocol.

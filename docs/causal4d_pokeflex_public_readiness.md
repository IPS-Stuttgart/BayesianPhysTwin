# Causal4D PokeFlex Public-Data Readiness

## Status

This public-data track does not replace or modify the frozen
`causal4d-preacquisition-v4` physical protocol. The v1 readiness config was
locked while access was pending and remains immutable. Access was subsequently
granted, and the server-side download completed. Raw data remain outside Git.
Only the five metadata-selected development takes have been used for source QA
and source-model evaluation; the calibration and target outcomes remain
unopened.

The public interface is pinned to:

```text
repository: https://github.com/pokeflex-dataset/reconstruction
commit:     aaa8726072834a95bbe97e1a113588968c36e185
config:     256f6c0585a1eb592583b0a0c017e116baed9126f12119e80f866cd174b58070
```

The source checksums and access boundary are locked in
`configs/causal4d_public/pokeflex_readiness_v1.json`.

## Public contract

The pinned upstream preprocessing code exposes the following raw take layout:

```text
object_id/take_id/
  robot_data.json
  meshes/mesh-fNNNNN.obj
  volucam/<camera>/camera_parameters.json
  kinect/<camera>/camera_parameters.json
  realsense/<camera>/camera_parameters.json
```

The preflight also recognizes the upstream processed layout with
`triangle_meshes/` and `images/camera_parameters.json`.

`robot_data.json` is expected to provide frame IDs, force vectors, and `T_WT`
tool transforms. Timestamps, explicit contact locations, and commanded actions
are detected independently; their absence disables only the claims that need
them.

## Critical topology boundary

The upstream preprocessing decimates every target mesh separately, and the
dataset classes load per-frame vertices and faces independently. Equal sampled
vertex counts or face hashes are useful compatibility checks but do not prove
material-point identity.

The preflight therefore separates:

- geometry-only Chamfer and point-to-surface evaluation;
- identity-dependent material track and per-vertex state evaluation.

Identity-dependent metrics remain disabled unless a checksummed companion
artifact explicitly establishes persistent vertex identities for the exact
mesh inventory. No nearest-index or same-index shortcut is permitted.

## Metadata-only split

Eligible takes are ranked per object by

```text
SHA256("causal4d-pokeflex-public-v1:<object>/<take>")
```

At least five eligible takes are required for the cross-take protocol. The
locked allocation is 60% development, 20% calibration, and 20% target, with at
least one take in each split. The split reads object/take identifiers only. It
does not read errors, meshes beyond schema checks, or model outcomes.

Nine independent calibration takes are still required before a nominal 90%
session-level split-conformal claim is allowed. A `3/1/1` exploratory split is
not a calibration result.

## Synthetic contract test

The repository can generate a small PokeFlex-shaped fixture containing no real
or restricted data:

```bash
causal4d-pokeflex-fixture \
  runs/causal4d_public_pokeflex_readiness_v1/synthetic_fixture \
  --objects 1 \
  --takes-per-object 5 \
  --frames 16
```

Run the same preflight that will later inspect the gated data:

```bash
causal4d-pokeflex-preflight \
  runs/causal4d_public_pokeflex_readiness_v1/synthetic_fixture \
  runs/causal4d_public_pokeflex_readiness_v1/synthetic_preflight.json \
  --config configs/causal4d_public/pokeflex_readiness_v1.json
```

The locked synthetic fixture produces a `3/1/1` metadata split. Geometry,
pose/wrench contact candidates, explicit contact, timing, and
command-versus-measured gates pass. Material identity and nominal 90%
calibration remain disabled. This is interface evidence, not a PokeFlex result.

Local and `gpuserver6000` runs reproduce synthetic preflight SHA-256
`a5f2a7c949d66168fbfe5e5d3227ff9b65008094b5658317274a0b439ac37dc1`
exactly. The isolated server suite with PyRecEst 2.4.1 reports 334 passed and
one skipped test. The compact verification record contains no PokeFlex data.

## Access-day procedure

1. Complete `pokeflex_download_manifest.template.json` with the exact release,
   archive sizes, and SHA-256 hashes.
2. Keep all raw and converted data outside git under
   `data/external/pokeflex` or a server-side equivalent.
3. Run the frozen preflight before fitting, rendering, or evaluating a model.
4. Archive the preflight JSON and its result hash.
5. Proceed with geometry-only factual continuation if that gate passes.
6. Add contact abduction, delay, command-versus-measured analysis, or material
   tracking only when their separate gates pass.

The preflight may reject individual malformed takes using schema metadata, but
it never hides them. All exclusions remain in the output manifest.

## First real-data preflight

An outcome-free preflight was run on the seven completed `3dPrintedBunny`
`poking` archives while the remaining download continued. All seven archives
passed ZIP integrity. Only robot records, camera calibrations, and OBJ meshes
were staged; RGB-D images and target prediction outcomes were not opened.

The public robot records encode frame IDs as zero-padded strings such as
`"00001"`. The adapter now accepts either nonnegative JSON integers or ASCII
digit strings, with a regression test for the public encoding.

The locked metadata split contains five development takes, one calibration
take, and one sealed target take. The preflight enables:

- factual geometry continuation;
- cross-take interventional evaluation;
- pose/wrench-based contact proposals.

It continues to disable:

- material-track and per-vertex metrics without verified identities;
- delay inference without timestamps;
- commanded-versus-measured separation without command logs;
- explicit-contact claims without contact annotations;
- nominal 90% session-level conformal calibration with only one calibration
  take.

The preflight result SHA-256 is
`56ff6606c3c90234f5945c23fa3999c45cc4490a0d6530b2086c79f80018b89a`.
The isolated `gpuserver6000` suite passed 433 tests with four skips using
PyRecEst 2.4.1. Compact evidence is recorded in
`milestones/pokeflex-001-public-preflight-v1`.

## Development-only source gate

A second locked phase opened only development takes T1, T3, T4, T6, and T7.
Source QA confirmed usable pose/wrench proximity cues and mesh geometry, while
rejecting persistent vertex identity and one cross-take canonical sparse graph.
The calibration take T5 and target take T2 remained unopened.

A 50-candidate shared-setting test then used take-specific 128-node surface
graphs in the pinned official PhysTwin Warp simulator. The implementation was
deterministic and satisfied the edge-strain gate, but the model failed its two
predictive admission criteria:

- pooled leave-one-take-out selection beat persistence in 0/5 takes;
- pooled selection beat median single-source selection in 2/5 takes;
- pooled mean Chamfer was 23.771 mm versus 10.093 mm for persistence;
- even the per-take source oracle lost to persistence in every take.

This is a negative result for the sparse surface-spring backend, not for the
full PhysTwin reconstruction pipeline or PokeFlex as a dataset. The complete
candidate matrix and checksummed QA artifacts are frozen in
`milestones/pokeflex-001-source-warp-gate-v1`. No target evaluation is allowed
for the rejected backend.

## Claim boundary

This track can support held-out interventional prediction across publicly
recorded pokes once a source backend passes its admission gate. It cannot supply
individual counterfactual ground truth. The first sparse official-Warp backend
did not pass, so no target prediction, calibration, contact-identification, or
Bayesian-PhysTwin result is claimed.

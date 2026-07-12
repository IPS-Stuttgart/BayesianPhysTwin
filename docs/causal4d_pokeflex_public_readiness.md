# Causal4D PokeFlex Public-Data Readiness

## Status

This is an access-independent public-data track. It does not replace or modify
the frozen `causal4d-preacquisition-v4` physical protocol. Dataset access is
pending, no PokeFlex data have been downloaded, and no PokeFlex outcome has
been inspected.

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

## Claim boundary

This track can eventually support held-out interventional prediction across
publicly recorded pokes. It cannot supply individual counterfactual ground
truth. Until the real preflight passes, it supports only software readiness and
an outcome-free protocol lock.

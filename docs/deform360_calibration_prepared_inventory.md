# Deform360 calibration prepared-source inventory

## Purpose

The official-Hub calibration-source workflow publishes a compact terminal record
and retains the larger aligned RGB, tactile, and robot products on the protected
self-hosted runner. The successful run proves that all ten frozen calibration
objects were downloaded and prepared, but its compact artifact deliberately does
not contain the videos or arrays needed by the next calibration-only stage.

`inventory_deform360_calibration_prepared_source.py` bridges that boundary. It
revalidates the exact source locks, names-only plan, download manifest, prepared
result, and successful terminal record before opening the retained calibration
files. It then hashes and records the exact media and array contracts required by
the visual/contact observability producer.

This is retained-source custody evidence, not visual competence or physical-query
evidence.

## Frozen successful source run

The main-only self-hosted workflow is bound to the authoritative least-privilege
source execution:

- exact reviewed source revision
  `0f403cbed8b5fc9ac585b5f7c237106809207b3f`;
- workflow run `31236564360`;
- compact artifact `deform360-official-calibration-source-31236564360-1`;
- artifact digest
  `sha256:866c3f05e733e0cd6548e97ea4134476a37c7e01d09614cda2e86b3cb59d97d2`;
- terminal-record ID
  `edf3692d88fed3c011ee44da2508b39e4755a0e97a83a26a0391fcfe433d7b74`;
- ten prepared physical objects, five per stratum; and
- the persistent `calibration-processed/aligned` runner root.

The earlier successful run `31236230283` remains valid execution history, but it
is not the inventory input because it predates the final step-scoped optional
credential boundary. The workflow also does not reuse the earlier failed
transport run as scientific evidence.

## Validation order

Before retained payload access, the command independently replays:

1. the source protocol, Stage-0 protocol, exact selection, visual-provider lock,
   dataset revision, and processing revision;
2. the ten-object names-only plan and its 8/10 plus 4/5 support rule;
3. the exact-file download manifest and confirmation-object denylist;
4. the complete prepared-source result; and
5. the strict successful terminal record.

Every replayed summary field must equal the terminal record. Substituting a file
with another structurally valid artifact fails before the retained source is
inventoried.

## Retained object contract

For every frozen calibration object, the inventory requires exactly one
`episode_0000` below the protected aligned root and verifies:

- the episode alignment manifest;
- the camera-calibration dictionaries and their result hashes;
- `robot/robot.npz`, including the complete array key, shape, dtype, and finite
  value contracts;
- every synchronized tactile array, including exact SHA-256, shape, dtype, and
  finite values;
- every aligned camera video, preview image, timestamp file, alignment manifest,
  and camera metadata file; and
- the exact 81-frame action-selected window recorded by the successful result.

Camera metadata must cite the actual video, preview, timestamp, and alignment
SHA-256 values observed on disk. All output paths are portable paths relative to
the prepared root; absolute runner paths are never published.

## Stable byte-custody reads

Every retained file is opened once through a no-follow file descriptor, verified
to be a regular file, streamed while hashing, and checked for unchanged device,
inode, size, modification time, and change time before the descriptor is closed.
The inventory never hashes one pathname and then reparses that pathname.

NPY, NPZ, and camera-metadata JSON are parsed from the exact verified snapshot
bytes. Replacing the source path after descriptor opening therefore cannot make
the recorded digest describe different bytes from those used to derive shapes,
dtypes, finite-value checks, array inventories, or metadata references.

Verified parse snapshots use anonymous temporary files on every supported Python
version. This deliberately avoids relying on version-specific
`SpooledTemporaryFile` seekability while keeping snapshots unnamed and absent
from the published inventory.

## One-shot retained-source admission

The reviewed main-only workflow now performs the complete metadata handoff in one
protected execution:

```text
authoritative compact source evidence
  + protected retained aligned root
        |
        v
prepared-source inventory
        |
        v
frozen all-camera visual-production plan
        |
        v
inventory-bound visual-execution admission
```

The workflow validates and publishes all three content-addressed artifacts plus a
compact receipt and canonical `SHA256SUMS`. Its reusable-workflow outputs expose
the Actions artifact identity and digest, inventory ID, plan ID, admission ID,
and admitted camera-job count. This removes the risk of manually pairing an
inventory from one retained source with a plan or admission from another source.

A one-shot push-to-main launcher calls only this reviewed reusable workflow. It
posts the terminal status and immutable identities to issue #148. Pull-request
validation remains hosted and payload-free; the retained source is opened only
after the reviewed launcher reaches `main`.

This stage does not decode a video, run MotionCrafter, run Prob4D, construct an
observability matrix, or open confirmation data.

## Information boundary

The inventory command acknowledges that the authorized calibration camera,
tactile, and robot products are opened. It fails closed if any frozen
confirmation object is present in the prepared root. It does not open geometry
annotations, compute calibration target metrics, open confirmation payloads, use
target outcomes, or permit object replacement.

The plan and admission stages consume only the newly published metadata. Their
information boundary therefore remains stricter than the inventory boundary:
retained payloads are not reopened while the executable work list is frozen.

The output claim boundary is limited to calibration-only retained-byte custody,
portable array/media contracts, and a deterministic inventory-bound visual work
list.

## Manual command

```bash
python scripts/science/inventory_deform360_calibration_prepared_source.py \
  --source-protocol \
    protocols/deform360_official_hub_calibration_source_v1.json \
  --stage0-protocol \
    protocols/deform360_official_hub_visuotactile_v1.json \
  --selection-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_selection.json \
  --visual-provider-lock \
    protocols/locks/deform360_official_hub_visuotactile_v1_visual_provider/visual-provider-lock.json \
  --plan calibration-source-plan.json \
  --download-manifest calibration-download-manifest.json \
  --result calibration-source-result.json \
  --run-record execution-manifest.json \
  --processed-root /protected/calibration-processed/aligned \
  --implementation-revision "$(git rev-parse HEAD)" \
  --output prepared-source-inventory.json
```

Publication is atomic and refuses overwrite.

## Output and downstream use

The content-addressed inventory contains one row per exact calibration object,
the exact source artifact digests, the action-selected window, all camera media
contracts, tactile array contracts, robot array contracts, and the closed
information boundary.

The frozen visual-production plan and execution admission then bind every
admitted camera job to an inventoried video and timestamp path, SHA-256, byte
count, causal frame range, seed, dependence group, and collision-free output
namespace. Their next consumer must execute exactly those calibration-only
MotionCrafter and Prob4D jobs, retaining technical failures without replacement.

The resulting visual-reference marginal precision, visual-plus-contact marginal
precision, contact-anchor artifact, and shared physical-query Jacobian for each
of the ten objects then enter the atomic calibration-observability batch. None of
the retained-source artifacts authorizes confirmation opening.

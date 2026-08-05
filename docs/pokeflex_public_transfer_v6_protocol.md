# PokeFlex public-action transfer audit v6

## Purpose

This audit evaluates the frozen all-object action-robust update on the 78 public
poking actions that were neither used for source calibration nor reserved for the
prospective v5 final-two test. It asks whether the source-calibrated scale map has
broad action-level headroom over both the released PokeFlex checkpoint and the
fixed global scale 0.125.

The 78 outcomes were exposed during earlier project work. The result is therefore
retrospective evidence, not prospective confirmation and not a reconstruction of
PokeFlex's unavailable five official validation records.

## Fixed partitions

- Public release: 116 poking takes.
- Source calibration: 36 takes, two actions for each of 18 objects.
- Prospective v5 reference: `Pillow_T4` and `PlushDice_T3`.
- Retrospective v6 audit: the exhaustive remaining 78 takes.

The protocol binds every retrospective ZIP by path, byte count, and SHA-256 in
`configs/sota/pokeflex_public78_archive_inventory_v6.json`. No failed action may be
replaced.

## Frozen method

Each target frame uses Kinect depth and robot history only through frame `f-1`.
The released checkpoint is corrected with the existing
`action_local_state_relative_0.4` field. The candidate scale is the frozen v4
object-specific multiplier times 0.125; the controls are the unchanged checkpoint
and global scale 0.125. Unsupported updates fall back exactly to the checkpoint.

The legacy predictor is byte-bound. Historical smoke outputs are not reusable
unless they contain this exact field, scale, upstream commit, and causal contract;
the server inventory found none that met that complete contract.

## Reporting boundary

Results must appear in this order:

1. the retrospective 78-action audit;
2. the already sealed prospective two-action result;
3. the descriptive combined 80 non-source actions.

For each block, report object-, action-, and frame-balanced CD-UL1, paired wins,
and an object-cluster bootstrap interval. The published 6.498 mm score is contextual
only because the public cohort is not the paper's official 18-take split.

The retrospective interpretation gate requires improvement over both physical
references, a negative 97.5% object-cluster upper bound, at least 12 of 18 object
wins, and no object regression larger than 1%. Passing can justify a genuinely
fresh guarded-update evaluation. It cannot undo the v5 prospective failure over
the global scale, authorize retuning on public outcomes, or establish SOTA.

## Operations

The 376.9 GB source archives stay on `gpuserver6000`. Each take is transferred
directly over the server LAN to `gpuserver4090`, staged, evaluated, and removed;
only the checksummed compact smoke artifact is retained. The jump server is not in
the payload path. No held-v8 artifact or process is touched.

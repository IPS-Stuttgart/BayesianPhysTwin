# DEFORM DLO long-run continuation v2

## Purpose

The frozen 280-update DLO1 reproduction reduced mean error by 76.01% relative
to action-aware persistence but failed its parity gate at 14.032 mm versus the
11.110 mm threshold. Its validation curve was still monotonically improving,
and the public DEFORM learning history reaches its best region only after
thousands of updates.

This protocol tests one specific explanation: the short run stopped before the
published model family converged. It does not revise or rerun the failed v1 gate.

## Frozen continuation

- Parent source result:
  `9722a7bf4800e18677daa15cb220a39f9a72c73ae2f7ca7d100bb4cba25e8f65`
- Parent update-280 checkpoint:
  `c42ddb273f6aefe932c42d899419e8579c798f15a6d2448d72849574197ab91d`
- Resume both model and optimizer state.
- Preserve batch size 32 and unroll horizon 50.
- Draw a new deterministic fit-only schedule with seed `20260731`.
- Continue for 6120 updates, ending at global update 6400.
- Evaluate validation checkpoints at global updates 640, 1280, 2560, 4000,
  5200, 6040, and 6400.
- Select the checkpoint using validation mean coordinate L1 only.

The DLO1 source-test trajectories are already open and remain explicitly
post-open exploratory. They are loaded only after validation checkpoint
selection. The official DLO1 evaluation directory remains protected by the
runtime audit and is forbidden.

## Decision gates

The long-run source model must again satisfy both:

1. held-out DLO1 source mean coordinate L1 at or below 11.110 mm; and
2. at least six of eight paired wins over action-aware persistence.

Failure closes this external reproduction route. Passing authorizes two
separate next steps, neither of which is a SOTA claim:

1. evaluate the already-locked late-checkpoint posterior arms with exact
   selected-single fallback; and
2. reproduce the same total budget and arm bank from scratch on fresh DLO2.

Only a passed fresh DLO2 source gate can authorize an identical-information
official evaluation.

## Command

```bash
python scripts/remote/run_deform_dlo_longrun.py \
  --protocol configs/sota/deform_dlo_longrun_v2.json \
  --source-protocol configs/sota/deform_dlo_source_v1.json \
  --source-result /path/to/source_result.json \
  --source-manifest /path/to/source_manifest.json \
  --starting-checkpoint /path/to/update_0280.pt \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/longrun-v2 \
  --device cuda:0
```

The runner requires the same PyTorch and CUDA versions as the parent result,
verifies every parent hash, and refuses a nonempty output directory.

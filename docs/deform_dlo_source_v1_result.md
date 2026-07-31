# Frozen DEFORM DLO1 source reproduction result

## Decision

The frozen source gate **failed**. The short-budget DEFORM reproduction is not
authorized to open the official evaluation split or to advance the exploratory
checkpoint-belief method.

The selected update-280 checkpoint achieved:

| Metric | Result |
| --- | ---: |
| Validation mean coordinate L1 | 12.976 mm |
| Held-out source-test mean coordinate L1 | 14.032 mm |
| Action-aware persistence mean coordinate L1 | 58.486 mm |
| Wins versus persistence | 8 / 8 |
| Frozen parity threshold | 11.110 mm |
| Published DLO1 reference used by the protocol | 10.100 mm |

The physical forward model reduced error by 76.01% relative to action-aware
persistence, but exceeded the parity threshold by 26.30%. Consequently:

```text
parity_passed = false
persistence_gate_passed = true
passed = false
advancement_authorized = false
```

This is a useful but scoped negative result. It establishes that known-action
physical dynamics carry substantial predictive signal on the held-out DLO1
source trajectories. It does not reproduce the published DEFORM accuracy under
the frozen 280-update budget, and it provides no evidence of state-of-the-art
performance.

## Evidence boundary

- The arm was fixed at source commit
  `4ef71f16b909a8db7b60f08047010250f0b765b1`.
- Training used 40 DLO1 trajectories, checkpoint selection used eight
  validation trajectories, and the gate used eight held-out source
  trajectories.
- The official DEFORM evaluation tree was protected by a runtime read audit and
  was not opened.
- Checkpoint selection used validation only; update 280 was selected from
  updates 0, 40, 80, 160, and 280.
- The exploratory checkpoint-belief implementation was independently locked at
  `1b737a7` before this result was read. Its runner requires
  `advancement_authorized = true`, so it must refuse this result and is not run.
- A separate local smoke reproduced the registered update-1 loss to within
  `3.8e-9 m`, reproduced persistence exactly, and produced the same frozen
  schedule hash. The authoritative source decision remains the server result.

## Provenance

| Artifact | SHA-256 |
| --- | --- |
| `source_result.json` | `9722a7bf4800e18677daa15cb220a39f9a72c73ae2f7ca7d100bb4cba25e8f65` |
| `source_manifest.json` | `570e8f65a4a9c5b5bbfeb923fcb8714885896ec041d862548b91ef6ff1599a8a` |
| `preflight.json` | `9b322b9aeabf3b1f70123e017b256aeee9d9d8d45418d21969be8f23d5bb3798` |
| `window_schedule.npz` | `9d8e565e357f3adeca7392b2cf7b6f8c8ce86381329aa891214ae8265523d7b3` |
| Remote runner log | `7dd1b6b0ab0cfdb52515b7f8978986f871975e959862b3d136c27dc4d3d6e528` |

Runtime: Python 3.10.12, PyTorch 2.0.1+cu118, CUDA 11.8,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. The run completed in 2721.15 seconds.

The checkpoint binaries remain in the checksummed source archive rather than in
Git. Their identities and sizes are bound inside `source_result.json`.

## Consequence

Do not increase the budget, change the parity threshold, inspect official
evaluation outcomes, or run post-open checkpoint averaging under this protocol.
Any longer-training DEFORM reproduction would be a new method-development
protocol, not a continuation of this frozen gate.

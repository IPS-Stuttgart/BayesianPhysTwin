# PhysTwin-conditioned DINO prefix competence result

Status: frozen negative source result. Do not advance this route.

## Question

The test asked whether causal DINOv2 material correspondence, searched only near
the released PhysTwin prediction and guarded by multiview RGB-D evidence, was
accurate and supported enough to justify a larger assimilation study.

The source-only protocol was locked at commit `ac1c8d5` before prediction. The
prediction used frames 114--120 of the already-open `single_lift_cloth` prefix.
Only the four manual identity positions at frame 114 were visible to prediction.
Manual values on frames 115--120 were opened only after the prediction archive
and report had been sealed.

## Frozen result

| Quantity | Result |
| --- | ---: |
| Eligible identity-frames | 24 |
| Accepted identity-frames | 11 |
| Accepted support | 45.83% |
| Released physical RMSE | 14.973 mm |
| Exact persistence RMSE | 2.612 mm |
| Exact-fallback candidate RMSE | 12.471 mm |
| Candidate gain over physical | 16.71% |
| Accepted-row physical RMSE | 17.090 mm |
| Accepted-row camera RMSE | 11.927 mm |
| Accepted-row gain over physical | 30.21% |
| Candidate last-two-frame RMSE | 13.077 mm |

The support gate failed: 45.83% is below the frozen 50% threshold. All other
declared gates passed, but the protocol requires every gate. The frozen decision
is therefore `stop-conditioned-dino-correspondence-route`.

## Interpretation

The observation is not useless. On the rows accepted without target access, it
removed 30.21% of the released physical identity error, and exact fallback
reduced the aggregate physical error by 16.71%. This confirms that a physical
search neighborhood plus appearance evidence can recover some real material
correspondence signal.

It is nevertheless the wrong route for this prefix. Exact persistence is only
2.612 mm, while the guarded camera candidate is 12.471 mm, a 377.51% regression
relative to persistence. The three-camera input never produced a three-view
accepted consensus; accepted rows used two views, and 13 of 24 rows fell back
exactly. Lowering the support threshold or widening the cross-view gate after
seeing these values would not address the larger persistence gap and is
forbidden by the lock.

The result reinforces the broader source evidence:

- short action-only windows can make persistence extremely difficult to beat;
- camera-internal agreement is insufficient protection against coherent metric
  bias;
- a useful update must be admitted relative to the unchanged baseline, not only
  relative to a poor physical point estimate;
- dynamic windows require target-free evidence of causal motion before a camera
  update is even considered;
- sparse material correspondence remains valuable as an observation proposal,
  but not as an unconditional replacement for the baseline.

No larger source panel, assimilation experiment, fresh-object run, or
state-of-the-art claim is authorized by this result.

## Provenance

- Implementation commit: `86e6dd76a34f1e91ffcb1c6b46a5b62631382949`
- Protocol-lock commit: `ac1c8d5`
- DINOv2 revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- DINO checkpoint SHA-256:
  `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`
- Prediction archive SHA-256:
  `a174f7fcaacfeaa6e0bf9deded8f8e27fa34179b0126a8dd84a461389e18bec7`
- Prediction report SHA-256:
  `284c7cabf7d2cde144a75b4a34e270ac0e286e13558f31ae8c13f8cd2db7d831`
- Prediction seal SHA-256:
  `233abe6c560b52ef3a9b040fea6111fa877da5a0ff16398f72f85131368d4840`
- Evaluation self-hash:
  `702b8957fcc41578ff0cbd0f99161241c8747428ff3843b1d26e7989c6bd1b27`

The compact evidence bundle is under
`results/sota/phystwin_conditioned_dino_prefix_competence_v1/`. It does not
contain the withheld manual target artifact.

# MolmoMotion Acceptance Gate

MolmoMotion remains outside the physical posterior and outside semantic beta
selection until it passes a direct competence benchmark. Selecting `beta=0`
is a successful safety decision: it preserves physical weights byte for byte
and prevents a weak learned forecast from degrading Causal4D.

## Locked benchmark

The protocol is stored in
`configs/causal4d/molmo_acceptance_v1.json`. Every source case tests whether
MolmoMotion:

1. beats zero motion by at least 5% in query-point ADE;
2. beats last-history constant velocity by at least 5%;
3. produces between 0.5 and 2 times the true RMS motion scale;
4. preserves material-point anchors, metric units, and camera/world transforms;
5. ranks the labeled correct action first in a prior-neutral feasible physical bank;
6. remains useful across three prompt paraphrases and leave-one-query-out scores.

At least three independent real executions are required, and at least 80% of
cases must pass all gates. Molmo evidence is scored directly before any beta is
fit. Physical action priors are normalized within each action, so a correct rank
cannot be inherited from the bank prior.

The benchmark also verifies the input contract:

- exact processed-to-raw material identities;
- metric 3D world coordinates and explicit camera-to-world conversion;
- source and forecast frame rates;
- regularly sampled history;
- sampled future frame indices;
- visibility and track-validity masks;
- fixed checkpoint, query set, prompts, and horizon.

## Temporal correction

The first pilot used adjacent PhysTwin frames and compared each Molmo timestamp
with each source frame. The raw videos are 30 fps, while the released H3/F30
checkpoint predicts 30 future frames at 15 fps. The corrected adapter therefore
uses history frames `[t0-4, t0-2, t0]` and future frames
`[t0+2, t0+4, ...]`, and stores `source_fps=30`, `forecast_fps=15`, and
`frame_stride=2` in the forecast artifact. Legacy artifacts remain readable but
retain their original stride-1 metadata and cannot pass the new temporal gate.

## Locked real result

The corrected `single_lift_sloth` run uses five captions: the original
instruction, two paraphrases, a shuffled action, and a generic caption. It is
evaluated against the frozen ambiguous-action bank from
`v0.3.0-causal4d-aip`; that milestone is not modified.

| Direct forecast | ADE | Vector RMSE | FDE |
| --- | ---: | ---: | ---: |
| Molmo instruction | 32.37 mm | 42.89 mm | 61.81 mm |
| Zero motion | 32.46 mm | 43.11 mm | 61.94 mm |
| Constant velocity | **25.09 mm** | **34.71 mm** | **46.75 mm** |

Molmo reaches only `0.0164` times the true RMS motion scale. Its ADE is
`0.997` times zero motion and `1.290` times constant velocity, so it beats
neither locked baseline. Anchor alignment (`0.000004 mm`), first-step error
(`3.87 mm`), camera/world conversion, and the 15 fps temporal contract pass.
This localizes the failure to forecast competence rather than association,
units, coordinates, or timing.

For language ranking, all three paraphrases rank the true released lift action
fifth of five and choose `history_persist` first. This remains unchanged under
every leave-one-query-out subset. The consistency is therefore consistently
wrong, not evidence of semantic competence.

**Decision:** keep `beta=0`, exclude MolmoMotion from the main claim set, and do
not tune beta or run a larger physical sweep. A new checkpoint, deformable-
object adaptation, or revised input regime must first pass this same locked
benchmark on at least three independent source executions.

The safe-fallback frequency is therefore 100% on the current source set, and
deployed physical degradation is zero by construction because no semantic
weights are applied. This is a safety result, not a positive Molmo result.

The compact result is
`runs/causal4d-molmo-acceptance-v1/acceptance_result.json`; it records hashes of
the manifest, forecast artifact, final data, and frozen rollout bank, plus a
self-digest checked before any CLI can unlock positive beta candidates.

## Reproduction

Generate a forecast at the checkpoint's rate:

```bash
causal4d-molmo-phystwin-forecast \
  CASE/final_data.pkl RAW_CASE MOLMO_CHECKPOINT molmo_15fps.npz \
  --train-end-frame 59 --forecast-fps 15 \
  --caption 'instruction=A person lifts the sloth upward with one hand.' \
  --caption 'paraphrase_one=Lift the sloth upward using one hand.' \
  --caption 'paraphrase_two=With one hand, raise the sloth vertically.'
```

Evaluate before beta selection:

```bash
causal4d-evaluate-molmo-acceptance \
  configs/causal4d/molmo_acceptance_v1.json \
  runs/causal4d-molmo-acceptance-v1/acceptance_result.json
```

The beta-fitting CLI enforces this boundary. Without a passed acceptance
artifact, requested positive candidates are reduced to `(0,)`:

```bash
causal4d-fit-semantic-trust source_manifest.json semantic_trust.json \
  --betas 0,1,3,6,12 \
  --molmo-acceptance-json \
  runs/causal4d-molmo-acceptance-v1/acceptance_result.json
```

## Hardware boundary

Semantic acceptance and hardware authorization are separate. Hardware may use
`beta=0`; a positive beta additionally requires this benchmark to pass.
Physical closed-loop execution always requires the calibration and safety gates
in `configs/causal4d/hardware_execution_gate_v1.json`.

The current graph-persistent posterior reaches only 67.78% target coverage at a
nominal 90%, and the locked source-calibrated transform worsens target coverage
to 43.03% with 30.40% worst-group coverage. The real-artifact closed-loop replay
establishes software behavior, not robot safety. Physical execution is therefore
not authorized by the current evidence.

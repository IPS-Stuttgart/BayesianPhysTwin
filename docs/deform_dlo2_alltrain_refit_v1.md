# DEFORM DLO2 all-training-data refit v1

## Purpose

The fresh DLO2 source protocol reserves 8 of 56 official training trajectories
for validation and another 8 for a transfer gate. That split is necessary for
honest method selection, but a model trained on only 40 trajectories would be
unnecessarily disadvantaged against published DEFORM, which trains on the
complete official training set.

This stage performs the standard final-refit step after confirmation. It uses
all 56 DLO2 training trajectories, but it cannot alter the method selected by
the fresh source experiment. The official evaluation directory remains
read-guarded throughout.

## Authorization

The runner refuses even preflight unless:

1. the frozen DLO2 single-checkpoint source gate passed;
2. the frozen DLO2 posterior selected a non-fallback arm;
3. that arm passed its 1% and 5-of-8 source transfer gates;
4. the posterior result and its selection seal agree exactly; and
5. every parent protocol, result, manifest, checkpoint, and selection hash
   verifies.

The selected operator, checkpoint weights, variance floor, and validation-fit
variance scale are copied unchanged. There is no validation, source, or target
reselection during the refit.

## Frozen refit

- DLO type: DLO2, 12 nodes
- Training trajectories: all 56 official `train` trajectories
- Updates: 6400 from scratch
- Batch size: 32
- Unroll horizon: 50 frames
- Checkpoints: 0, 280, 640, 1280, 2560, 4000, 5200, 6040, 6400
- Optimizer: the official DEFORM SGD parameter groups
- Official evaluation access: forbidden

For a parameter-mean posterior, the runner materializes the weighted model
state after training. For a predictive-mean posterior, it preserves the exact
member checkpoints and weights. Both forms produce one immutable final-method
artifact for the later one-shot evaluation.
The source-validation-selected single-checkpoint update is also carried into
the all-56 run as a fixed comparison baseline, allowing the untouched
evaluation to isolate the Bayesian arm's contribution under identical data and
training schedules.

## Command

```bash
python scripts/remote/run_deform_dlo2_alltrain.py \
  --protocol configs/sota/deform_dlo2_alltrain_refit_v1.json \
  --source-protocol configs/sota/deform_dlo2_fresh_v1.json \
  --source-result /path/to/dlo2/source_result.json \
  --posterior-result /path/to/dlo2/posterior_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/dlo2-alltrain-v1 \
  --device cuda:0 \
  --mode run
```

Passing this stage means only that the preselected method was refit on all
available training data. It does not itself report an evaluation result or a
state-of-the-art claim.

# Synthetic benchmark simulation-based calibration v1

## Objective

The existing `bayesian_phystwin.simulation_based_calibration` module summarizes
posterior PIT values but does not itself execute an inference model. This study
runs an end-to-end controlled check of the finite-grid physical-parameter
posterior used by the BayesianPhysTwin synthetic benchmark.

It separates two explanations for poor real-data calibration:

1. a basic posterior implementation inconsistency already present under the
   declared simulator and observation model; or
2. real-data model/discrepancy shift despite internally coherent inference.

Passing this study supports only the second interpretation as the remaining
candidate explanation. It does not validate the simulator against reality.

## Generative design

For every independently generated replicate:

1. sample `(stiffness, damping, control_scale)` uniformly from the exact frozen
   finite parameter grid;
2. select a balanced dynamic or quasi-static action;
3. simulate the complete fixed-graph trajectory;
4. add independent Gaussian noise to the registered prefix; and
5. evaluate the exact finite-grid Gaussian posterior.

Complete simulation replicates are the independent units. Frames, nodes,
coordinates, particles, and action samples are nested observations.

## Normative controls

All arms use the same truths, observations, finite prior, posterior support, and
randomized tie breakers:

- `matched_likelihood`: inference standard deviation equals the generating
  observation standard deviation;
- `underdispersed_0.5x`: inference assumes half the true standard deviation; and
- `overdispersed_2x`: inference assumes twice the true standard deviation.

The mismatched arms are deliberate negative controls. They verify that the SBC
pipeline detects posterior dispersion errors rather than merely returning a
plausible-looking histogram.

## Run

```bash
python scripts/science/run_synthetic_benchmark_sbc_v1.py \
  --replicates 512 \
  --seed 20260824 \
  --likelihood-scale-multipliers 1,0.5,2 \
  --output outputs/synthetic-benchmark-sbc-v1/result.json
```

The writer is atomic and refuses to replace an existing result unless
`--overwrite` is explicit.

## Report

For every physical parameter and arm, retain:

- randomized PIT histogram;
- mean PIT;
- Kolmogorov--Smirnov distance from uniformity;
- Cramer--von Mises discrepancy;
- central 50%, 90%, and 95% posterior mass coverage;
- lower and upper 5% tail rates; and
- the exact summary artifact identity.

The aggregate record additionally reports whether the matched arm has smaller
mean KS distance and smaller nominal-90% coverage error than both deliberate
dispersion controls. These are normative pipeline checks, not a real-data claim.

## Interpretation boundary

A well-behaved matched arm establishes controlled self-consistency for the exact
finite-grid synthetic posterior. It does not establish simulator adequacy,
physical identifiability, real observation-model validity, unseen-object
transfer, deployment calibration or safety, Prob4D provider competence,
Causal4D intervention benefit, or state of the art. No Deform360 confirmation,
DLO4/DLO5, held-v8, Prob4D target, or Causal4D physical outcome is consumed.

# Simulation-based calibration v1

Status: **Experimental controlled evidence**

Maturity: **Research-only; outside the stable wheel and public API**

## Scientific question

The operational real-data posterior is strongly undercalibrated, but that fact
alone does not distinguish an inference defect from model misspecification. This
study asks a narrower controlled question:

> Is the complete discrete-grid posterior calibrated when truth and observations
> are generated from exactly the model used by inference, and does the same
> diagnostic detect the existing correlated-corruption misspecification?

The implementation lives in
`bayesian_phystwin_experiments.simulation_based_calibration_v1`. It consumes no
external data, released PhysTwin outcome, Deform360 source or confirmation
outcome, Prob4D provider output, or Causal4D outcome.

## Frozen design

The content-addressed protocol is
[`protocols/locks/simulation_based_calibration_v1.json`](../protocols/locks/simulation_based_calibration_v1.json).
It freezes:

- 512 replicates at seeds `31000:31512`;
- the complete default `17 x 11 x 9 = 1683` parameter grid;
- a uniform discrete truth prior over stiffness, damping, and control scale;
- dynamic and quasi-static actions;
- the clean exact-model condition and the existing correlated-corruption
  condition;
- the first 60 of 90 frames as observations;
- a clean Gaussian grid posterior with the registered 6 mm observation standard
  deviation;
- scalar diagnostics for stiffness, damping, control scale, and final-frame
  last-node displacement; and
- 50%, 80%, 90%, and 95% equal-tail interval accounting.

For a discrete posterior, ordinary ranks are not uniform because multiple grid
particles may share a marginal value. The registered diagnostic therefore uses
randomized probability integral transforms:

```text
u = P(X < x_true | y) + V P(X = x_true | y),  V ~ Uniform(0, 1).
```

Under the exact joint model, each marginal randomized PIT is uniform even when
the posterior is discrete. The scalar terminal-displacement query tests a
physical prediction rather than only parameter coordinates.

## Familywise decision

There are eight clean-model tests: four quantities under two action modes. For
`n = 512`, the protocol applies a Bonferroni-adjusted 95% Dvoretzky--Kiefer--
Wolfowitz envelope to the Kolmogorov--Smirnov distance of each empirical PIT
CDF. The exact-model decision is
`exact_model_calibration_not_rejected` only if all eight clean tests remain
inside that common envelope.

The correlated condition is a fixed negative control, not a selectable stress.
`correlated_misspecification_detected` requires all eight corresponding tests to
leave the same envelope. This stress deliberately applies the clean Gaussian
posterior to occlusion, coherent drift, boundary noise, and flow inconsistency.
No robust method is selected from this result.

Equal-tail coverage is reported with 95% Wilson intervals, but it is not the
primary calibration test. Discrete-grid intervals can be conservative even when
the randomized PIT is calibrated.

## Execution

```bash
python scripts/science/run_simulation_based_calibration_v1.py \
  --protocol protocols/locks/simulation_based_calibration_v1.json \
  --output outputs/simulation-based-calibration-v1/result.json \
  --summary-output outputs/simulation-based-calibration-v1/summary.json \
  --require-registered-decision
```

The runner publishes both files atomically and refuses to overwrite an existing
path. The full result retains every replicate; the compact summary binds the
complete row table by SHA-256.

## Interpretation boundary

A clean-model pass shows that this exact discrete posterior implementation is
not detectably miscalibrated under its own declared synthetic model at the
registered resolution. A correlated-control failure shows sensitivity to this
one fixed misspecification.

Neither result proves that real BayesianPhysTwin undercalibration is caused only
by misspecification. It does not establish real-data calibration, latent-state
identification, unseen-object transfer, provider competence, Causal4D
intervention benefit, deployment safety, or state of the art. The independent
Deform360 covariance study and the physical Causal4D experiment remain separate
and necessary evidence gates.

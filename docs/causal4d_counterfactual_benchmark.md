# Causal4D Controlled Counterfactual Benchmark

This package is an independent research track under `src/causal4d`. It does
not import or alter the Bayesian PhysTwin estimators. The first milestone asks
a narrower question before any large generative model is integrated:

> Can a model infer uncertain physical properties from repeated interactions
> and predict the outcome of a genuinely unseen intervention when contact
> realization changes?

The benchmark is deliberately small enough to run on CPU and explicit enough
that every source of information can be audited.

## Locked protocol

The protocol contains three 2D spring-graph objects:

| Object | Graph | Nodes | Unknown parameters |
| --- | --- | ---: | --- |
| rope | path | 7 | stiffness, damping, contact gain |
| cloth | diagonal 3 x 3 grid | 9 | stiffness, damping, contact gain |
| soft block | diagonal 2 x 4 grid | 8 | stiffness, damping, contact gain |

Mass, rest geometry, topology, support stiffness, commanded force trajectory,
and nominal material contact node are known. The simulator also retains the
exact true values of all inferred parameters.

Each object has six action templates:

- four training actions: `left_lift`, `right_drag`, `centre_pulse`, and
  `dual_stretch`;
- one validation action: `reverse_sweep`;
- one untouched test intervention: `diagonal_hook`.

Every training action is repeated under firm and compliant contact by default.
Only the training interactions enter parameter inference and baseline fitting.
The validation interaction selects the hybrid residual scale. Neither fitting
nor model selection reads the test action trajectory.

The test intervention is executed in two independent worlds:

- `matched_contact`: nominal contact location and transmission;
- `shifted_contact`: reduced transfer, a two-step delay, and a one-edge shift
  in the realized material contact.

Both worlds include a small contact-frame bias and nonlinear strain stiffening
that are absent from the linear planning model.

## Information boundary

All methods receive the same commanded force sequence, nominal contact node,
object graph, rest state, and observed training trajectories. They do not
receive the realized contact gain multiplier, delay, shifted node, frame bias,
or nonlinear world term. Those values are known simulation ground truth and
are written to `protocol.json` for evaluation and audit.

This command-versus-realization split is intentional. Giving a predictor the
realized shifted contact would turn the test into matched-simulator replay and
remove the causal ambiguity the benchmark is meant to measure.

## Parameter ambiguity

The physical posterior is evaluated on an object-relative grid over
stiffness, damping, and contact gain. Slow deformations primarily constrain
parameter ratios, while repeated contact transmission varies between
interactions. A tempered likelihood accounts for temporal correlation in the
sparse sensor traces.

Ambiguity is reported directly through:

- effective sample size;
- normalized posterior entropy;
- maximum absolute posterior parameter correlation;
- the number of particles with greater-than-uniform posterior weight.

Parameter results include posterior mean error, a 90% credible interval,
coverage, interval width, and weighted CRPS for every object and seed.

## Baselines

### Generative only

A regularized action-to-trajectory model learns directly from observed
interactions. Its inputs are command-only descriptors: nominal contact
position, force impulse, peak force, contact count, and temporal centroid. It
does not simulate dynamics or consume candidate physical parameters.

The implementation is intentionally lightweight ridge regression. It is a
controlled stand-in for a learned trajectory generator, not a claim that ridge
regression is a competitive video model.

### Physics only

The physics baseline evaluates the repeated observations under a grid of
candidate physical parameters. It predicts the held-out action by integrating
the nominal graph simulator for every posterior particle and moment-matching
the resulting trajectory mixture.

### Hybrid

The hybrid starts from the complete physics posterior prediction. A second
regularized model learns the remaining trajectory residual from training
interactions. Its correction scale is selected from
`{0, 0.25, 0.5, 0.75, 1, 1.25}` on the validation action only. A zero selected
scale is a valid result: it means the residual failed to transfer.

## Metrics

Held-out intervention error is reported as trajectory RMSE, ADE, FDE,
direction error, early/middle/late RMSE, and RMSE relative to the magnitude of
the true intervention. A gross-failure indicator uses the configured endpoint
threshold.

Predictive uncertainty is evaluated with marginal 90% coverage, absolute
coverage error, interval width, mean normalized squared error (NEES), and
Gaussian negative log likelihood. Results are balanced by object, seed, world,
and method before aggregation.

## Run

Install the package in editable mode, then run:

```bash
python3 -m pip install -e ".[dev]"

causal4d-counterfactual-benchmark \
  --seeds 0:5 \
  --frames 56 \
  --training-repeats 2 \
  --parameter-grid-count 5 \
  --output-dir runs/causal4d-counterfactual-v1
```

The output directory contains:

| Artifact | Contents |
| --- | --- |
| `summary.json` | configuration and aggregate metrics |
| `protocol.json` | exact objects, true parameters, splits, contacts, and force trajectories |
| `interventions.csv` | one held-out metric row per seed/object/world/method |
| `parameter_recovery.csv` | one posterior row per seed/object/parameter |
| `fit_diagnostics.csv` | ambiguity and hybrid-selection diagnostics |
| `manifest.json` | byte sizes and SHA-256 checksums for every artifact |

No generation timestamp is stored, so identical inputs produce byte-identical
artifacts.

## Scope boundary

This milestone establishes an auditable counterfactual test bed. It does not
yet integrate MolmoMotion, MotionCrafter, a diffusion world model, PhysTwin's
Warp simulator, or real video. Those integrations should begin only after a
candidate method can state which information it consumes and beat these
baselines without test-action leakage.

# Causal4D Latent Contact Inference

The first controlled benchmark showed that a nominal physical posterior is
accurate under matched contact but degrades when the commanded contact is not
the contact realized by the world. A trajectory residual did not remove that
error. This milestone moves the discrepancy to its causal source:

\[
p(x_{t+1:T}, z, \theta \mid x_{0:t}, u),
\]

where `z` contains material contact assignment, transmission gain, delay,
slip spread, and control-frame rotation, while `theta` contains stiffness,
damping, and nominal contact gain.

## Contact hypotheses

For every nominal commanded contact, the model enumerates the nominal graph
node and all one-hop neighbors. It crosses these graph-relative assignments
with the locked support:

| Latent | Support |
| --- | --- |
| transmission multiplier | 0.70, 0.85, 1.00 |
| delay | 0, 1, 2 simulation steps |
| slip spread | 0.00, 0.20 |
| control-frame rotation | 0, 8 degrees |

Each contact hypothesis is simulated under the twelve highest-mass particles
from the object's physical posterior. The resulting rollout bank is a finite
joint approximation to `p(z, theta)`. Predictions are posterior-weighted
mixtures, not a MAP rollout.

Slip distributes force between the selected node and its graph neighbors. A
strictly zero default preserves the first benchmark's trajectories.

## Information boundary

Every method receives:

- graph topology and rest geometry;
- commanded force history and nominal contact node;
- the object's training interactions used for physical-parameter inference;
- in the online setting, only the configured early observation prefix.

The target's realized contact node, gain, delay, slip, frame rotation, and
future trajectory remain evaluator-only. Perturbing any target frame after the
forecast boundary leaves the inferred posterior byte-identical in the leakage
test.

## Two inference settings

### Pre-intervention

Before observing the response, the model marginalizes the source-trained
contact prior and physical posterior. It therefore reports both an expected
trajectory and uncertainty over possible contact realizations.

### Online adaptation

The model observes the first 20% of motion and reweights every `(z, theta)`
rollout using position, velocity, and acceleration evidence. The remaining 80%
is the scored forecast. Candidate likelihood scale, likelihood power, dynamic
weight, and posterior temperature are selected using source topologies only.

Predictive intervals use weighted mixture quantiles. Matched and shifted
interval scales are calibrated on source interactions and blended at test time
using the inferred probability of a graph-node shift. Target future frames are
never used for interval selection.

## Held-out topology protocol

Every object is evaluated once as the excluded target topology:

1. Choose rope, cloth, or soft block as target.
2. Fit the action-conditioned contact prior from matched and shifted
   `diagonal_hook` interactions on the other two topologies.
3. Select likelihood, posterior temperature, and interval scales on those
   source interactions.
4. Fit the target's physical posterior from its ordinary training actions.
5. Predict the target's untouched matched and shifted `diagonal_hook`
   interactions.

Thus the target object, its contact labels, observation prefixes, and futures
are excluded from contact-model fitting. This evaluates transfer to a new
topology. It does not claim simultaneous zero-shot transfer to both a new
topology and a contact mode absent from all source data.

## Controls

Four methods are evaluated in both settings:

| Method | Contact information | Physical parameters |
| --- | --- | --- |
| `nominal_physics` | nominal contact fixed | training posterior; updated from prefix online |
| `latent_contact` | inferred posterior over contact | joint posterior; updated from prefix online |
| `oracle_contact` | true realized contact | training posterior; updated from prefix online |
| `oracle_contact_theta` | true realized contact | true simulated parameters |

`oracle_contact` isolates the value of contact knowledge under parameter
ambiguity. Because a misspecified simulator can use an incorrect latent state
to compensate structural discrepancy, it is not guaranteed to bound the
hybrid model. `oracle_contact_theta` is therefore the strict recoverable
simulation ceiling used for the oracle-gap gate. It retains only the small
nonlinear plan/world mismatch.

## Recovery and calibration

The benchmark reports trajectory RMSE, ADE, FDE, direction and horizon errors,
NEES, NLL, 90% coverage, and interval width. Contact recovery includes:

- node MAP accuracy, confidence, multiclass Brier score, and 90% credible-set
  coverage;
- gain posterior mean error, CRPS, and interval coverage;
- delay MAP accuracy, mean error, CRPS, and interval coverage;
- slip and frame-rotation recovery;
- contact and physical-particle effective sample sizes.

## Pre-registered gates

The online aggregate passes only when all of the following hold:

| Gate | Threshold |
| --- | ---: |
| shifted nominal-to-strict-oracle gap recovered | at least 50% |
| matched-contact RMSE degradation | at most 10% |
| absolute error from 90% trajectory coverage | at most 5 percentage points |
| shifted node MAP accuracy | at least 80% |
| shifted node credible-set coverage | at least 80% |
| shifted node confidence calibration error | at most 15 percentage points |
| shifted gain MAE | at most 0.15 |
| shifted gain interval coverage | at least 80% |
| shifted delay MAE | at most 0.5 steps |
| shifted delay MAP accuracy | at least 80% |
| shifted delay interval coverage | at least 80% |
| held-out topologies | all three |
| per-topology oracle-gap closure | non-negative for every topology |

Thresholds live in `LatentContactConfig` and are emitted with every result.

## Locked five-seed result

The default protocol with seeds 0 through 4 passed all registered gates. Online
forecast RMSE and marginal coverage were:

| Method | Matched RMSE | Shifted RMSE | Matched coverage | Shifted coverage |
| --- | ---: | ---: | ---: | ---: |
| nominal physics | 2.463 mm | 4.132 mm | 87.6% | 77.9% |
| latent contact | **2.046 mm** | **0.805 mm** | 88.2% | 90.8% |
| true-contact control | 1.711 mm | 1.165 mm | 92.9% | 97.8% |
| strict contact-and-parameter oracle | 0.045 mm | 0.005 mm | 100% | 100% |

The latent model recovered 80.6% of the shifted nominal-to-strict-oracle gap
and improved matched-contact RMSE by 16.9%. Shifted-contact node accuracy was
86.7%, node confidence calibration error was 7.0 percentage points, gain MAE
was 0.084, and delay MAE was 0.403 steps. Gain and delay 90% intervals each
covered 93.3% of cases.

Oracle-gap closure remained positive on every excluded topology: 60.8% for
cloth, 85.3% for rope, and 83.8% for the soft block.

Pre-intervention marginalization is a risk-aware hedge, not an identification
result. Its single prediction improved shifted RMSE from 4.364 mm to 1.709 mm
but worsened matched RMSE from 2.164 mm to 4.129 mm. The strong result therefore
comes from causal information in the early response, not from averaging two
possible futures before contact is observed.

## Run and artifacts

```bash
causal4d-latent-contact-benchmark \
  --seeds 0:5 \
  --observation-fraction 0.20 \
  --contact-parameter-particles 12 \
  --require-gates \
  --output-dir runs/causal4d-latent-contact-v1
```

The output bundle contains:

| Artifact | Evidence |
| --- | --- |
| `summary.json` | configuration, aggregates, and gate results |
| `protocol.json` | information boundary, source supervision, and controls |
| `interventions.csv` | every object/seed/world/setting/method forecast metric |
| `contact_recovery.csv` | contact and joint-parameter posterior diagnostics |
| `fold_calibration.csv` | source objects and selected hyperparameters per fold |
| `success_gates.json` | machine-readable gate values and thresholds |
| `manifest.json` | SHA-256 and byte count for every payload |

Identical inputs produce byte-identical artifacts.

## Scope boundary

This remains a controlled 2D graph benchmark with simulation contact labels.
It establishes causal identifiability and a leakage-resistant evaluation
before adding video, a learned contact proposer, or a full generative world
model. The next empirical step is to replace supervised source contact labels
with proposals inferred from RGB-D tracks while retaining the same oracle and
held-out-topology controls.

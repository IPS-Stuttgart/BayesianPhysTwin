# Cross-intervention transport criterion: controlled helpfulness study

## Question

Does the held-out cross-intervention criterion add scientific discrimination beyond fitting the opened source action?

The study is target-free. It uses only the frozen 18-session source-to-target action roster and controlled local-linear simulations. It does not read any Causal4D physical outcome or protected confirmation cohort.

## Registered comparisons

Three decision rules are compared over complete physical sessions:

1. **source-only:** a positive lower interval for same-action source improvement;
2. **transport-only:** a positive lower interval for held-out-action gain;
3. **full protocol:** held-out gain plus superiority to discrepancy-only and last-residual controls, at least 14 accepted sessions, and the one-sided 95% harmful-accepted-update cap of 20%.

The frozen regimes include a transportable physical coefficient, independent and correlated source-local discrepancy, shared bias, an action-aligned nuisance with and without declaration, conservative physical transport, a simulator sign error, and a physical/local mixture.

## Main result

With 2,000 trials per regime and 2,000 session bootstraps:

| Regime | Source-only | Transport-only | Full protocol |
|---|---:|---:|---:|
| Transportable physical | 100.0% | 100.0% | 38.4% |
| Source-local discrepancy | 100.0% | 1.0% | 0.0% |
| Correlated local discrepancy | 100.0% | 0.9% | 0.0% |
| Shared action-independent bias | 100.0% | 0.2% | 0.0% |
| Action-aligned undeclared nuisance | 100.0% | 98.0% | 28.4% |
| Action-aligned declared nuisance | 100.0% | 0.0% | 0.0% |
| Conservative physical transport | 100.0% | 100.0% | 78.5% |
| Physical simulator sign error | 100.0% | 0.0% | 0.0% |
| Physical/local mixture | 100.0% | 41.2% | 0.4% |

## Interpretation

The theory is helpful as a **falsification criterion**. Same-action source fit is non-diagnostic and accepts local discrepancy almost universally; held-out intervention transport reduces this false physical attribution by roughly two orders of magnitude while preserving sensitivity to a shared physical coefficient.

It is not sufficient as a causal-identification theorem. An undeclared nuisance with the same action signature can fool the transport endpoint. The nuisance-aware identifiability certificate is therefore essential, not optional. A simulator sign error also causes a true physical mechanism to fail transport, which is a correctly retained limitation rather than evidence that the physical mechanism is absent.

The complete protocol is intentionally conservative. At the frozen 18-session sample size, a half-scale correction has much higher probability of satisfying the harmful-update certificate than the full posterior-mean correction. Any shrinkage rule must be selected on source/calibration data and frozen before target access.

## Reproducible primary result

The canonical evidence runtime is CPython 3.12.14 with NumPy 2.2.6. The committed primary run is generated with:

```bash
python scripts/science/run_cross_intervention_transport_criterion_v1.py \
  --roster protocols/cross_action_transport/causal4d_sloth_multi_action_v1_sparse_pairs.json \
  --trials 2000 \
  --bootstrap-replicates 2000 \
  --seed 20260827 \
  --noise-scale 0.3 \
  --guard-threshold 0.25 \
  --harmful-gain-margin 0.25 \
  --discrepancy-shrink 0.5 \
  --output-json result.json \
  --output-markdown report.md
```

The expected decision is `criterion-useful-but-requires-declared-nuisance-and-conservative-guard` and the canonical result ID is `b39b4bef293d0c976cb3ba1e86fce2dc22f505a8d9185f247fcfb02b695ad552`.

The dedicated GitHub Actions workflow pins the canonical runtime, reruns the 2,000-trial study twice on the exact reviewed revision, requires primary and replay to be byte-identical, and additionally requires the regenerated JSON and Markdown to be byte-identical to the checked-in retained result before uploading the evidence bundle.

## Claim boundary

Controlled local-linear mechanism evidence only. This study does not establish a unique physical cause, simulator adequacy, real-object transfer, real calibration, provider competence, Causal4D physical benefit, deployment safety, or state of the art.

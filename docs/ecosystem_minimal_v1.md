# Minimal Prob4D–BayesianPhysTwin–Causal4D contract smoke v1

## Purpose

`examples/ecosystem_minimal_v1.py` provides one executable narrative across the
three repositories' public boundaries:

1. a deterministic synthetic producer creates a Prob4D-compatible
   `ObservationBeliefV1`;
2. BayesianPhysTwin saves and reloads the content-addressed observation;
3. an accepted guard selects the candidate complete belief;
4. a rejected guard returns the caller-owned baseline by exact object identity;
   and
5. the example records the Causal4D provider capability manifest.

This is a contract and packaging smoke test. It does **not** establish that a
real Prob4D observation feeder improves BayesianPhysTwin, that the example
covariance is calibrated, or that the selected belief improves a Causal4D
counterfactual endpoint.

## Run

From an editable checkout:

```bash
python3 -m pip install -e ".[dev,data,graph]"
python examples/ecosystem_minimal_v1.py \
  --output-dir outputs/ecosystem-minimal-v1
```

The output directory contains:

- `prob4d_observation_belief_v1.npz`, whose descriptor includes its content address;
- `accepted_decision.json`;
- `fallback_decision.json`; and
- `ecosystem_summary.json`.

The accepted and fallback decisions deliberately share the same candidate and
baseline artifacts. Their only difference is the frozen guard decision. This
makes it easy to verify that fallback routing is not an approximation or a
reconstructed copy.

## Promotion beyond the smoke test

A real integration claim requires a frozen external forecast artifact, strict
source revision and calibration identity, independent evaluation units, a
proper predictive score, calibration/width reporting, subgroup regressions,
and the same exact fallback accounting. A downstream Causal4D claim additionally
requires a registered factual-prefix/intervention/counterfactual protocol.

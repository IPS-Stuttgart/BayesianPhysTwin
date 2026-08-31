# Deform360 matched temporal baseline v6

## Main 92-object result

| Method | Active-field RMSE |
|---|---:|
| `action_conditioned_tcn` | 0.61635616 |
| `v3_bayesian_action_ensemble` | 0.62837977 |
| `state_only_tcn` | 0.64247537 |
| `v3_state_kernel` | 0.64741558 |
| `v3_shuffled_action` | 0.65135100 |
| `persistence` | 0.66297449 |

- Action-conditioned TCN versus persistence: **-7.03%**, 90/0/2 object wins/ties/losses.
- Action-conditioned TCN versus matched state-only TCN: **-4.07%**, 91/0/1.
- Frozen Bayesian ensemble versus action-conditioned TCN: **+1.95%**; the TCN is significantly better.
- Frozen Bayesian ensemble versus matched state-only TCN: **-2.19%**; the Bayesian ensemble is significantly better.

## Held-out action-family subset

On the 21 objects whose target action family is absent from every same-object source episode, the action-conditioned TCN obtains **0.67790042**, the Bayesian ensemble **0.69182314**, and the state-only TCN **0.70549306**.

## Scientific conclusion

The competitive learned baseline is the best point predictor. This rules out a raw-SOTA or best-RMSE claim for the frozen Bayesian ensemble. The result still supports the action relation: future-action conditioning improves a capacity-matched TCN by 4.07% overall and 3.91% on the held-out action-family subset. The Bayesian ensemble remains better than the matched state-only TCN and should be positioned around guarded physical admissibility, exact fallback, source-only adaptation, and auditable belief structure rather than raw predictive dominance.

## Evidence identity

- GitHub Actions run: `33417883597`
- Job: `99572873900`
- Artifact: `9767812227`
- Artifact SHA-256: `68ff5fb8e2274f3e71a20ef79a7450d0baa0cf57985e88064b31e4c1edc2bca2`
- Execution revision: `a32948698fe43e4e52a443a93c9c1604012a21cf`
- Retained Ruff-formatted and lint-clean source revision: `f8cb2950992f212ec245afcdd1c760cb5691f5cb`
- Formatting and import-location lint correction changed no scientific configuration, model weight, prediction, metric, or result value.

## Claim boundary

This is a post-confirmation same-object tactile-forecasting audit. It does not establish fresh confirmation, zero-shot unseen-object transfer, dense 4-D geometry validation, calibrated joint uncertainty, strict counterfactual identification, or an overall state-of-the-art claim.

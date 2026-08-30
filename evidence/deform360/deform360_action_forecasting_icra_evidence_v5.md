# Stronger Deform360 action-conditioned forecasting evidence (v5)

## Primary new confirmation

The exact frozen v3 predictor was evaluated on **all 92 carrier-complete objects** selected by a metadata/file-identity-only readiness audit before their numeric robot or tactile payloads were opened by this study. No replacement, partial-roster reporting, or outcome-dependent filtering was allowed.

| Contrast | Active RMSE / difference | Relative change | 95% object bootstrap | Object W/T/L | Exact one-sided sign p |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.66297449 | — | — | — | — |
| Bayesian action ensemble | 0.62837977 | **−5.22%** | **[−0.04002, −0.02957]** | **91/0/1** | **1.878e−26** |
| Ensemble − shuffled action | −0.02297122 | **−3.53%** | **[−0.02872, −0.01810]** | **90/0/2** | **8.641e−25** |
| Ensemble − state-only kernel | −0.01903580 | **−2.94%** | **[−0.02240, −0.01587]** | **91/0/1** | — |

The action ensemble lowers active-field RMSE by **5.22%** relative to persistence and wins on **91 of 92 objects**. Breaking the future-action relation worsens prediction on **90 of 92 objects**. The action-conditioned ensemble also beats the otherwise comparable state-only nonlinear kernel on **91 of 92 objects**, directly isolating the value of the intervention input.

The source-only guarded predictor accepts **88.0%** of objects and retains almost the full gain: active-field RMSE 0.62938278, approximately **5.07%** better than persistence, with 90/92 wins.

All six populated target-action families have a negative mean ensemble-minus-persistence difference: compress, dynamic, lift, other, shape, and translate.

## Confirmation panel

The earlier four-object reserved cohort and the new 92-object cohort contain 96 unique confirmation objects. Their descriptive combined panel is:

- persistence active RMSE: **0.66991389**;
- Bayesian action ensemble active RMSE: **0.63534014**;
- relative improvement: **5.16%**;
- object wins: **95/96**;
- 95% object-bootstrap interval for the difference: **[−0.04006, −0.02947]**;
- exact one-sided sign-test p-value: **1.224e−27**.

## Information order and provenance

- Exact frozen v3 method revision: `25ba91c021124569c4dcf84c66eda5ec088868e0`.
- Readiness run: `33335281293`; artifact: `9738831748`; artifact SHA-256: `71f68d7ce3addaac12ebe0a6b928df319f1fe72ed41396a71616cc3a28562c4b`.
- The readiness audit selected every eligible object from released metadata and file identities while keeping robot/tactile numeric payloads and target scores unopened.
- Selection-manifest SHA-256: `11be5369c2a8d0a30d9db497c217ddd5446aeabdc867d8b50400dd617942a66d`.
- Confirmation run: `33335779766`; artifact: `9738998271`; artifact SHA-256: `e98f9e2687f568d0d0fcabec9ce0393a7e1b34ca3019acb1e14fdf894885a948`.
- Confirmation-result SHA-256: `26984c5ecc59ef8b10a8efe94c86f5ea55fd88111913c4b46ae06704410c6c0e`.
- The first launch, run `33335694494`, stopped in environment validation because system Python lacked NumPy; the numeric evaluation step was skipped. The successful retry changed only the isolated NumPy runtime bootstrap.
- The retained Python wrappers are Ruff-formatted and pass `py_compile`; formatting changed no protocol, evidence value, or scientific result.

## Important boundary

This materially strengthens an ICRA claim about **same-object, source-adapted, action-conditioned real tactile-response forecasting**. It still does not establish zero-shot unseen-object transfer, dense 4-D geometry reconstruction, strict individual counterfactual ground truth, or globally fresh validation.

Uncertainty remains the main negative result: marginal 90% coverage is 81.9%, but joint coverage is only 32.0% and normalized joint ANEES is 1594.7. Also, all-field MAE is 0.149449 versus 0.142674 for persistence, although both active-field RMSE and all-field RMSE improve.

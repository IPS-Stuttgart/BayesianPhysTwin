# Partial-observation conditioning v1: completed result

## Verdict

The experiment supports the usefulness of correlated conditioning for hidden
reconstruction and future prediction on this retrospective DEFORM panel. It
does **not** establish a distinctive point-accuracy advantage of the structured
Bayesian residual model over source-tuned empirical covariance or deterministic
ridge regression. The registered stronger model-value flag is **false**.

## Execution and immutable evidence

- Pull request: [#939](https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/939).
- Completed Actions run: [33985145819](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/33985145819).
- Successful job: `101357169693`, runner `workstation1`, label `gpuserver4090`, GPU `cuda:1`.
- Scientific execution commit: `287e3d7384f1759404184792e538fad815eb7d60`.
- Trigger: creation of `experiments/partial_observation_conditioning_v1/request.json` in a push, not a manual dispatch.
- All ten focused analytic/masking/conditioning tests passed. The source replay,
  source selection, pre-target source-seal upload, target baseline-parity checks,
  complete scoring and final artifact upload all passed.
- Full evidence artifact: `9974964346`,
  `partial-observation-conditioning-v1-33985145819`, 560656 bytes,
  SHA-256 `d077edb8f257bf709b42862f70a9a85855a359effd10caf8521ffb9dedc3b552`.
- Source-seal artifact: `9974963686`,
  `partial-observation-source-seal-33985145819`, 101039 bytes,
  SHA-256 `cdba9ec11870df5463c4085a68932a54e938d981de88c4264e18134f484395d3`.

This receipt transcribes the completed job's printed summary. Table entries and
interval endpoints below retain its four-decimal rounding; full-precision
`result.json`, all 5544 case/mask/horizon/method records, source-selected
parameters, model arrays, input identities and execution logs are in the
immutable full evidence artifact. This receipt is not a second simulator run or
an independent raw-data reproduction.

## Protocol actually evaluated

Two physical objects (DLO4 and DLO5), 56 source trajectories and 14 evaluation
trajectories each. The 28 evaluation records were already opened historically.
The frozen source-trained DEFORM hybrid checkpoints were replayed on source
trajectories only. Their existing target predictions were reused after SHA-256
verification and agreement with the original all-coordinate L1 metric.

The new operator reveals 2/8 or 4/8 internal nodes at fixed anchors, using six
spread/contiguous masks. Current hidden reconstruction and hidden predictions
10 and 30 frames later are scored. The observed identities and four boundary
nodes are excluded from scoring. No artificial measurement corruption, action
selection, robotic intervention or new physical data were used.

The structured Gaussian residual belief, its diagonal/sign-scrambled controls,
and the empirical Gaussian have the same pre-conditioning mean and coordinate
marginal variances. All hyperparameters are chosen on source trajectories and
sealed before this experiment reads evaluation outcomes. An independent
information-form implementation checks Gaussian posterior-mean/MAP equivalence.

## Point prediction results

Equal mask and trajectory averages, balanced across the two DLOs. The metric is
**hidden-node 3D point RMSE in millimetres**, not the original DEFORM
all-coordinate L1 benchmark metric. The larger absolute values therefore must
not be compared directly with the old 8-10 mm L1 table.

| Method | Current reconstruction | +10 frames | +30 frames (primary) | +30 translation-free shape |
|---|---:|---:|---:|---:|
| Frozen DEFORM hybrid, no new update | 33.3329 | 33.7980 | 34.3788 | 18.4331 |
| Common source-bias-corrected prior | 31.2437 | 31.8191 | 32.3808 | 16.7253 |
| Structured Gaussian conditioning | 19.1039 | 25.8846 | 30.2847 | 15.9826 |
| Diagonal marginal-matched conditioning | 31.2437 | 31.8191 | 32.3808 | 16.7253 |
| Sign-scrambled marginal-matched conditioning | 40.9364 | 37.8237 | 34.6699 | 20.2796 |
| Source-tuned empirical covariance | 19.0130 | 25.8300 | 30.5241 | 16.0123 |
| Source-tuned deterministic ridge | 19.0130 | 25.8300 | 30.5241 | 16.0123 |
| Source-tuned translation correction | 25.5709 | 28.8189 | 32.3808 | 16.7253 |
| Source-tuned spatial interpolation | 24.2500 | 28.4038 | 32.3808 | 16.7253 |
| Frozen-current interpolation | 24.2500 | 72.6005 | 171.2519 | 76.4377 |
| Equivalent information-form MAP | 19.1039 | 25.8846 | 30.2847 | 15.9826 |

Relative to the matched prior, the structured model improves current
reconstruction by approximately **38.86%**, +10-frame prediction by **18.65%**,
and +30-frame prediction by **6.47%**. Relative to the untouched DEFORM hybrid,
the corresponding reductions are **42.69%, 23.41%, and 11.91%**; those latter
comparisons include receiving additional observations as well as conditioning.

The source-tuned empirical covariance and ridge controls are slightly better
at reconstruction and +10 frames. At +30 frames the structured model has a
small favorable point estimate, approximately **0.78%**, without an interval
excluding zero.

## Primary paired comparisons

Differences are structured minus comparator; negative favors structured.
The 95% intervals resample complete trajectories within each of the two fixed
objects. They do not treat frames or nodes as independent, and do not support
population-level generalization across objects.

| Comparator | Difference (mm) | 95% trajectory-bootstrap interval (mm) |
|---|---:|---:|
| Diagonal conditioning / matched prior | -2.0961 | [-3.0299, -1.2469] |
| Empirical covariance | -0.2394 | [-0.5599, +0.1108] |
| Deterministic ridge | -0.2394 | [-0.5599, +0.1108] |
| Translation | -2.0961 | [-3.0299, -1.2469] |
| Interpolation | -2.0961 | [-3.0299, -1.2469] |
| Frozen-current interpolation | -140.9672 | [-150.5919, -131.2818] |

Source validation selected zero correction gain for translation/interpolation
at +30 frames, explaining why they equal the common prior there. Empirical
covariance selected zero shrinkage at every horizon, explaining its agreement
with the regularized deterministic regression solution.

Against diagonal conditioning, the structured model wins 11/14 DLO4 and 13/14
DLO5 trajectories, improving the per-object mean by 9.229% and 3.666%.
Against empirical covariance/ridge, it wins 9/14 in each object, but the mean
improvements are only 0.925% and 0.649%. Neither reaches the registered 1%
per-object margin, and the combined paired interval crosses zero.

The independently computed Bayesian posterior mean and same-model MAP agree
within **9.17e-13 metres** in their prediction corrections. There is no
Bayesian-versus-equivalent-optimization point-accuracy claim.

## Conditional uncertainty at +30 frames

| Method | Marginal 90% coverage | Full interval width (mm) | Joint NLL / coordinate | Normalized joint NEES |
|---|---:|---:|---:|---:|
| Structured | 92.28% | 62.879 | -3.2726 | 1.0293 |
| Diagonal | 91.64% | 66.247 | -2.5118 | 1.1986 |
| Sign-scrambled | 89.42% | 62.879 | 0.6636 | 8.9018 |
| Empirical covariance | 91.53% | 59.855 | -3.2773 | 1.3102 |

The structured model has near-nominal marginal coverage and near-unit average
normalized joint NEES on this panel. This is not an independent calibration
certificate. The empirical model has narrower intervals and a slightly better
average NLL, so no general uncertainty-score superiority is established.

## Claim boundary

Supported: joint residual conditioning can transfer sparse observations into
better hidden-region reconstruction and conditional prediction, beyond matched
independent uncertainty and the tested simple spatial correction rules.

Not supported: a distinctive benefit of the structured Bayesian formulation
over comparably equipped empirical or deterministic methods; an identified
physical-parameter posterior; fresh-object confirmation; real-camera occlusion
handling; broadly calibrated uncertainty; or robot-control safety.

The parent predictor receives the complete recorded boundary trajectory.
Results are conditional on that known trajectory. Coordinate withholding is
synthetic, but the underlying trajectories are real recordings. The backbone
was trained on all source records, including those used for the new residual
model's validation split. The study does not evaluate Prob4D or Causal4D
end-to-end and does not require either component.

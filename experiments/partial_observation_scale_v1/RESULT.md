# Completed visible-to-hidden uncertainty-scale continuation

## Verdict and research decision

The focused continuation was worth testing, but it does **not** establish a
distinctive Bayesian advantage. The scale-integrated Student-t model beats
all four Gaussian controls on joint NLL, including a Gaussian with exactly the
same conditional mean and covariance. However, it loses the registered primary
score to both independently fitted Student-t controls. The stronger model-value
criterion and the scale-update-versus-static-Student-t criterion are **false**.

The positive mechanism is the value of a non-Gaussian predictive shape relative
to its Gaussian moment approximation. It is not a successful demonstration
that Bayesian updating of the shared scale is superior to competitive empirical
uncertainty estimation.

Recommendation: **stop tuning this particular residual-scale family on the
already-open DLO4/DLO5 targets as a route to a Bayesian-superiority headline.**
Keep the useful conditioning and heavy-tail analyses as bounded evidence; they
do not replace the previously failed stronger point-accuracy criterion. This
recommendation concerns the tested implementation, not Bayesian inference in
general or the complete physical-twin research program.

## Execution and evidence

- PR: [#939](https://github.com/IPS-Stuttgart/BayesianPhysTwin/pull/939), retained as draft and unmerged.
- Successful scientific workflow: [34012836057](https://github.com/IPS-Stuttgart/BayesianPhysTwin/actions/runs/34012836057).
- Successful job: `101431480120`; runner `workstation1`, tag `gpuserver4090`, device `cuda:1`.
- Frozen code/protocol revision: `9bd6f4ec108f46fd5b181e0905e68b57b4dac869`.
- Request-file-only trigger revision: `e2fe457305483893ba930bcc33112f2df673e459`.
- Trigger: creation of `.github/requests/partial-observation-scale-v1.json`; no manual dispatch.
- All 23 focused tests passed: ten unchanged parent tests and thirteen new tests.
- Source checkpoint replay, source fitting, source-seal upload, target baseline
  checks, scoring, independent verification and artifact upload all completed.
- Full artifact: `9983001281`, `partial-observation-scale-v1-34012836057`, 520759 bytes.
- Artifact SHA-256: `d0b1ed1807af46411d6b5ad8741b54e4f4aa838ec09f17def3d0f98c98ae1193`.
- Source-seal artifact: `9983000664`, 30952 bytes, SHA-256
  `00f789aeea8c57f8778a6719e16d06fb04702c6fae0496ed0098d68ebdcbe5f6`.
- Original full-precision `result.json` SHA-256:
  `5621b0354cfd58197eadc3482e0ec4474fee6c1f7d3718dffbd5f47cebbb65a4`.

This receipt transcribes the completed job's printed results. It is not a
second empirical run or a byte-identical replacement of the original JSON.
Full-precision results, fitted configurations, 1176 case/mask/arm records,
scoring inputs, input identities and logs are retained in the full artifact.
The prior experiment and its result have not been changed or rerun for tuning.

## What was tested

Use all 14 historically opened evaluation trajectories for each of DLO4 and
DLO5, with the unchanged DEFORM hybrid anchor. Split each object's 56 source
trajectories into 32 covariance/mean fitting, 12 hyperparameter selection, and
12 scalar calibration records. Seal all selections before target preparation.

The six prior visibility masks reveal 2/8 or 4/8 internal nodes at 18 fixed
observation times. Forecast the withheld nodes 30 frames ahead. Every method
has **exactly identical point predictions**, with aggregate hidden-node 3D RMSE
`30.52306364573443 mm`. The uncertainty comparison cannot obtain a gain by
changing the mean. Its base source split differs from the first experiment,
so numerical point results should not be treated as a new comparison with that
experiment's 56-source-refit means.

The new Bayesian model integrates a positive shared residual scale after
conditioning on visible residuals. Gaussian and heavy-tailed static controls,
exact and independently recalibrated moment-matched Gaussians, and independently
fitted observation-dependent Gaussian/Student-t variance regressions receive
the same information. The direct Student-t family is expressive enough to
represent the Bayesian conditional law; an analytic test verifies inclusion.

## Main scores

Lower NLL, CRPS and Brier are better. NLL is a continuous log-density score in
nats per hidden coordinate; negative values are normal. CRPS is a marginal
probabilistic score, not point RMSE. Width is the full marginal 90% interval.

| Model | Joint NLL/coordinate | CRPS (mm) | 90% coverage | Width (mm) | Brier: absolute coordinate error above 20 mm |
|---|---:|---:|---:|---:|---:|
| Independently calibrated static Gaussian | -3.296629 | 9.7331 | 92.99% | 66.354 | 0.155414 |
| Bayesian scale-integrated Student-t | -3.431243 | 9.4446 | 90.98% | 59.915 | 0.141827 |
| Exact mean-and-covariance-matched Gaussian | -3.260419 | 9.4806 | 91.15% | 60.392 | 0.144159 |
| Independently recalibrated moment Gaussian | -3.291906 | 9.5759 | 92.67% | 65.301 | 0.148800 |
| Independently calibrated static Student-t | -3.453773 | 9.5294 | 91.92% | 61.833 | 0.144411 |
| Direct observation-dependent Gaussian | -3.309196 | 9.6406 | 92.88% | 65.075 | 0.150932 |
| Direct observation-dependent Student-t | -3.456067 | 9.4691 | 91.93% | 61.403 | 0.141399 |

Bayesian uncertainty has the smallest mean CRPS and narrower intervals, but
its CRPS differences from both Student-t controls have intervals crossing zero.
Marginal coverage near 90% is not a calibration certificate. Its normalized
joint NEES is `1.381814`, versus `1.115654` for the direct Student-t model.

## Primary comparisons

Differences below are Bayesian minus comparator. Intervals resample complete
trajectories within each of the two fixed objects, not individual points,
frames or masks. They do not establish arbitrary-object population effects.

| Comparator | NLL difference | Paired 95% interval | Bayesian case wins |
|---|---:|---:|---:|
| Static Gaussian | -0.134614 | [-0.256791, -0.060472] | 28/28 |
| Exact moment Gaussian | -0.170825 | [-0.349427, -0.067230] | 28/28 |
| Recalibrated moment Gaussian | -0.139337 | [-0.266260, -0.066867] | 28/28 |
| Direct observation-dependent Gaussian | -0.122047 | [-0.232549, -0.056810] | 28/28 |
| Static Student-t | +0.022529 | [+0.013896, +0.032546] | 1/28 |
| Direct observation-dependent Student-t | +0.024824 | [+0.015384, +0.036583] | 1/28 |

Against static Student-t, the Bayesian CRPS difference is
`-0.084742 mm [-0.272027, +0.036553]`. Against direct Student-t it is
`-0.024487 mm [-0.142013, +0.056138]`. Neither establishes a CRPS advantage.
The corresponding Brier differences are also inconclusive.

All three registered flags are retained:

```
scale_update_value_vs_static_t: false
integration_value_vs_exact_moment_gaussian: true
stronger_model_value: false
```

The successful integration flag cannot replace the failed stronger criterion.
No target-side method, scale, degree-of-freedom, mask, horizon, split, or
success-rule change followed the result.

## Independent checks

The verifier imports no experiment implementation. On all 1176 retained rows,
its reconstructed target residuals agree within `5.55e-17 m`; independently
computed SciPy log densities agree within `5.33e-15`; every aggregate agrees
within `1.42e-14`; and all reconstructed paired interval endpoints agree exactly.
These are independent arithmetic/density checks, not a second simulator run.

## Scope

This is a retrospective result on two previously studied physical objects.
The backbone was trained on all 56 source records, including the records now
used for uncertainty selection/calibration. It is therefore not an independent
calibration study. Masks withhold coordinates from real trajectories; they are
not camera-occlusion experiments. Recorded future boundary trajectories remain
inputs to the physical anchor. The posterior describes output-discrepancy
scale, not identified physical parameters. The test assumes a per-window scale
and does not establish temporal independence, real-provider competence,
end-to-end Prob4D/Causal4D benefit or robot-control safety.

No new physical data, active probing, large dataset copy, or reserved-cohort
access was used. Production APIs and manuscript claims are unchanged.

# Full-22 covariance-only hybrid result

The registered retrospective analysis completed in workflow run `31461910994`,
attempt 1, on analyzer revision
`9ac5344036331a40e9029a1aa5814f601c0eaf15`.

The evidence artifact is
`full22-covariance-only-hybrid-v1-31461910994-1` (artifact `9090224528`), with
archive digest
`sha256:945dd5ed5db9b4119d81cf15d6d6c6da304b9421ab5938e71ad65390bbca1676`.
The report ID is
`5fc777163fd6173c9669b497309d883e2780a5ebe23da5dbe4cdaf682ad8806a`.

All effects below are covariance treatment minus zero-covariance
`last_residual`; lower Gaussian negative log likelihood is better. The predictive
mean is the exact same array object in every case, so track and Chamfer effects
are exactly zero by construction.

| Arm | Mean NLL effect | Simultaneous 95% CI | Decision |
| --- | ---: | ---: | --- |
| Cross-fitted selected/scaled covariance | `-9.136` | `[-13.961, -4.312]` | hybrid better |
| Independent raw covariance | `-6.143` | `[-9.830, -2.457]` | hybrid better |
| Dynamic raw covariance | `-7.801` | `[-12.913, -2.690]` | hybrid better |
| Independent cross-fitted/scaled | `-10.414` | `[-16.186, -4.642]` | hybrid better |
| Dynamic cross-fitted/scaled | `-8.994` | `[-14.415, -3.573]` | hybrid better |

The primary cross-fitted arm was better in `17/22` complete object-session
units, worse in `5/22`, and tied in none. The simultaneous horizon effects were:

| Horizon | Mean NLL effect | Simultaneous 95% CI |
| --- | ---: | ---: |
| Early | `-1.539` | `[-2.874, -0.203]` |
| Middle | `-8.233` | `[-13.222, -3.243]` |
| Late | `-17.638` | `[-26.324, -8.952]` |

Marginal 90% coverage increased from `0.706` to `0.910`. Mean full interval
width increased from `0.01645 m` to `0.05094 m`, a `3.10×` width ratio. Thus the
result is a proper-score and calibration gain with a material interval-width
cost, not a point-prediction gain.

The leave-one-object-session-out selector chose
`independent_endpoint_v1` in `21/22` folds and `dynamic_endpoint_v2` in `1/22`.
For one separately registered, genuinely fresh study, the full-source fit is:

- exact `last_residual` mean;
- covariance donor `independent_endpoint_v1`; and
- early/middle/late covariance scales `[8.0, 16.0, 16.0]`.

## Source and information boundary

The evaluator verified the historical prefix-manifest identity, every prefix-case
archive, all prediction-manifest bindings, and the SHA-256 of
`final_data.pkl`, `inference.pkl`, `gt_track_3d.pkl`, and `split.json` for all
22 cases. It did not recompute the historical split with current helper code.

The full-22 outcomes were already open before this secondary hypothesis.
Leave-one-object-session-out selection prevents each scored case from tuning its
own donor or scale, but does not create fresh confirmatory evidence. Therefore:

- `analysis_status=retrospective-cross-fitted-development-only`;
- `claim_authorized=false`;
- `selection_authorized=false`; and
- `promotion_authorized=false`.

The frozen candidate may be evaluated in a separate fresh-object/session study.
This result does not authorize deployment, a current paper claim, target-cohort
retuning, or modification of the frozen Deform360 v6 candidate roster.

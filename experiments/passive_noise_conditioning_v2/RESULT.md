# Passive visibility/noise continuation: completed result

## Verdict

The follow-up supports operational value of measurement uncertainty for current/near-term hidden-geometry reconstruction. It does not establish a uniquely Bayesian point-estimation advantage, rod-mode superiority over equally noise-aware regression, or improved +25-frame prediction. PR #938's original failed primary result remains unchanged.

## Execution

- Successful scientific run: **34012741438**, workstation1 / gpuserver4090.
- Executed commit: `f9b7803d7321eca0f259ff9a5c063724bb49da50`.
- Source revision before request: `8fc0f4b1643c14cfb9c711ef44b385943d99c2f7`.
- Trigger: change only `.github/requests/passive-noise-conditioning-v2.json`.
- Eleven focused tests, Ruff check, and format check passed before scoring.
- Preparation run 34012707694 accessed no data and only formatted/tested new source.
- No simulator training, new physical data, active observation selection, dataset mutation, or modification of old result records.

## Protocol boundary

Real DEFORM DLO4/DLO5 recordings and checksum-bound cached hybrid-plus-local-residual forecasts. Eight source-test trajectories and fourteen already-open evaluation trajectories per object. Both objects' hyperparameters and uncertainty calibrations are source-frozen before evaluation loading.

Source selection uses every contiguous two/four-node mask and independent noise at 0/5/15 mm. Evaluation uses all non-contiguous subsets: 21 two-node and 65 four-node configurations, 18 cutoffs, horizons 0/5/25. Each is a one-step conditional query, not a recursive changing-mask tracking test.

**Trajectories are real; masks and Gaussian noise laws are artificial.** The reported risk analytically integrates measurement noise, rather than selecting sampled seeds. Supplied noise covariance is oracle-known except explicit miscalibration controls. This is not a raw-camera-noise benchmark.

Primary: current hidden geometry, two of eight free nodes observed, equal mixture of clean, 2 mm iid, 10 mm iid, heterogeneous 2/10 mm, and common 10 mm translation plus 2 mm independent noise. The 30 mm and covariance-misspecification conditions are separate stress results.

## Primary result

Metric: coordinate root expected MSE in mm, calculated per complete trajectory and equally averaged over the two objects. Primary conditions are equally weighted in MSE before the trajectory square root. Lower is better.

| Method | DLO4 | DLO5 | Equal-object |
|---|---:|---:|---:|
| Matched mean, no conditioning | 16.4704 | 16.4615 | 16.4660 |
| Source-tuned fixed ridge | 11.4255 | 11.1305 | 11.2780 |
| Source-tuned pooled-noise ridge | 11.3364 | 10.9143 | 11.1253 |
| Source-selected fixed comparator | 11.4255 | 10.9143 | 11.1699 |
| Full measurement-covariance conditioning | 10.9251 | 10.5091 | 10.7171 |
| Rod-mode covariance conditioning | 10.9895 | 10.4729 | 10.7312 |
| Observation-correlation-blind conditioning | 11.1567 | 10.6999 | 10.9283 |
| Independently solved noise-aware direct ridge | 10.9251 | 10.5091 | 10.7171 |

Full covariance versus source-selected fixed comparator: **4.054% improvement**, **28/28 wins**, difference **-0.452850 mm**, conditional trajectory-bootstrap 95% interval **[-0.563507, -0.378056] mm**. Both DLOs improve. The predeclared 2% operational-gain criterion passes.

Rod modes versus equally noise-aware direct ridge: **0.132% worse**, difference **+0.014156 mm**, interval **[-0.032019, +0.075004] mm**. Structural-superiority criterion fails.

Gaussian conditioning and independently implemented noise-aware ridge agree: maximum prediction difference **2.0123e-15 m**, maximum retained metric difference **1.4211e-14 mm**. This is an equivalence control, not a separate method win. Deterministic implementation of a Gaussian conditional mean does not negate the usefulness of the uncertainty representation; it does limit claims of exclusivity or novelty.

Full versus observation-correlation-blind covariance gives **1.933%** improvement, interval **[-0.250054, -0.157878] mm**. It is statistically favorable but below the generic 2% practical threshold, so its gate is not declared passed.

## Horizon and observation-budget results

Fixed five-condition primary mix; four-node cases are secondary.

| Visible free nodes | Horizon | Source-selected fixed | Full noise-aware conditioning | Gain |
|---:|---:|---:|---:|---:|
| 2 | Current | 11.1699 | 10.7171 | +4.05% |
| 2 | +5 | 12.2661 | 11.9333 | +2.71% |
| 2 | +25 | 16.9028 | 16.9568 | -0.32% |
| 4 | Current | 9.1807 | 8.1176 | +11.58% |
| 4 | +5 | 10.8395 | 10.2007 | +5.89% |
| 4 | +25 | 16.8445 | 16.9870 | -0.85% |

The benefit is current/near-term observation assimilation, not long-horizon dynamics. Four-node rod-mode conditioning is 2.26% worse than equally noise-aware empirical/direct regression at current completion.

## Measurement covariance misspecification

Shared 10 mm translation plus 2 mm independent noise, two visible nodes:

| Supplied covariance | Noise-aware error | Fixed comparator | Gain |
|---|---:|---:|---:|
| Correct R | 12.0028 | 13.2105 | +9.14% |
| R/4: standard deviation understated by two | 13.2417 | 13.2105 | -0.24% |
| 4R: standard deviation overstated by two | 12.1274 | 13.2105 | +8.20% |

Understating uncertainty removes the practical gain. R/4 minus fixed has difference +0.031155 mm [-0.171838, +0.154048], so it is inconclusive, not a demonstrated harmful effect. Strong-noise 30 mm iid gives 23.6387 -> 15.5642 mm versus fixed gains, but is not the primary result.

## Same-mean uncertainty score

For full empirical conditioning at the primary endpoint, source-calibrated conditional variance gives Gaussian expected NLL **-3.234712**, versus **-3.125731** for source-fitted static coordinate variances: **-0.108981 nats per coordinate**, with the predictive mean held exactly fixed. Lower is better. This secondary proper-score result supports conditional uncertainty usefulness, not calibrated coverage. At +25 frames, its NLL is slightly worse than static uncertainty.

## Audit and limitations

Downloaded original artifact ZIP and linked protocol/source-choice/source-seal/CSV hashes verified. Independently reconstructed **432 summary records and all 270 paired-bootstrap comparisons** from **10,752 rows**. Maximum summary discrepancy **2.995e-12**, maximum contrast/interval discrepancy **9.383e-12**. Source/evaluation filename sets are disjoint within each DLO. This is a compact-artifact audit, not an independent raw-data or simulator rerun.

Two physical objects, not 28 independent objects. Bootstrap resamples whole trajectories within the two fixed DLOs; it does not imply arbitrary-object generalization. The original source and evaluation predictors had different training budgets (39/56). Full initialization and known clamp trajectories are inherited shared inputs. Covariance describes predictive discrepancy, not simulator parameters or a revised physical state. Evaluation recordings were previously opened, so this is retrospective, not fresh confirmation.

## Recommendation

Retain uncertainty-aware partial-observation conditioning as a useful estimator component. Do not discard that practical value merely because an equivalent noise-aware regression implementation exists. Conversely, do not promote this controlled-noise result to a standalone novel Bayesian-superiority claim or a solution to long-horizon dynamics. No further parameter sweep or dataset was launched after inspecting these outcomes.

## Evidence identity

Artifact ID **9982968445**, name `passive-noise-conditioning-v2-34012741438`.

ZIP SHA256: `cbb4afa5adbe229cbb6913f683ffb682dbfe8d2a1c0f3d00d16a59374d872112`.

Result JSON SHA256: `7b0b5db202946ebb815fd68285156f82e2e79d97573f86d469dedf22d02ce3ea`.

Scientific results refer to the executed commit above. This documentation commit does not trigger another scientific run. PR #943 remains draft and unmerged.

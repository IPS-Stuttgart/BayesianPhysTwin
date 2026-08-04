# Full-22 source-tuned tempered model-average experiment v1

> **Evidence status:** retrospective, non-claim-bearing experiment. The official
> released cohort and its frozen three-case development / nineteen-case
> confirmation split already informed method development. These results are not
> fresh independent validation, deployment calibration, or a new state-of-the-art
> claim.

## Locked execution

- Protocol SHA-256: `a351cf37ba19130feca4dcfb87b1e7ab9a2e601d22edeed3a39a00c904ecbbe3`
- Selection ID: `d204e656894c007644c4b04cbe1b529da0e6aca0081a823dc4d5ef602bbb56fa`
- Self-hosted run: `30897954787` on `workstation2`
- Self-hosted artifact: `8887837678`
- Artifact SHA-256: `91af2dca7ada3c8dbdb606af9b872dcd0d7ec16703f317bd2d4f7e3824e241c0`
- Evaluated branch head: `04bfd1547421681a1f0178fdf5d7fdfacdd4347f`
- Evaluated PR merge revision: `d5e75db975acc9bc5b587d7b747dbf9d02180eba`
- Pre-target gates: Ruff, formatting, 52 focused tests, package consistency,
  protocol digest, source-only selection serialization, and artifact checksums all
  passed.

The workflow wrote and hash-bound `selection.json` before evaluating the nineteen
confirmation cases. The selection artifact records
`confirmation_outcomes_opened=false`.

## Development-selected parameters

- Evidence temperature: **128**
- Model-average guard threshold: **4.611 mm** endpoint disagreement
- Confirmation guard acceptance: **5/19 cases (26.3%)**
- Covariance inflation scales: **2.742 early**, **5.049 middle**, **8.261 late**

The zero-regret development constraint selected a very restrictive guard. Fourteen
confirmation cases therefore reproduce the last-residual fallback exactly.

## Confirmation point result

| Method | Chamfer (mm) | Track (mm) |
| --- | ---: | ---: |
| released PhysTwin | 11.122 | 22.189 |
| selected Bayesian anchor | 9.828 | 19.523 |
| temperature-1 model average | 9.713 | 19.335 |
| temperature-128 model average | 9.764 | 19.429 |
| temperature-128 + selected guard | 9.614 | 19.234 |
| last-supported residual | **9.581** | **19.188** |

The source-selected temperature does **not** improve the untempered model-average
mean on confirmation. Relative to temperature 1, temperature 128 is worse by
`0.050 mm` Chamfer and `0.094 mm` track error; the track-error interval is
`[+0.008, +0.191] mm`.

## Registered fallback comparison

The unguarded tempered model average is worse than last residual by:

- Chamfer: **+0.182 mm**, paired 95% interval `[+0.057, +0.322] mm`;
- track error: **+0.240 mm**, paired 95% interval `[+0.037, +0.504] mm`.

The restrictive guard removes most of that harm but does not beat the fallback:

- Chamfer: **+0.033 mm**, interval `[-0.015, +0.101] mm`, bootstrap probability
  of improvement `0.127`;
- track error: **+0.046 mm**, interval `[-0.015, +0.116] mm`, bootstrap
  probability of improvement `0.077`.

The guarded candidate does improve the selected Bayesian anchor in Chamfer by
`0.214 mm`, interval `[-0.448, -0.022] mm`; its `0.289 mm` track improvement has
an interval crossing zero. This does not overturn the primary fallback result.

## Predictive calibration diagnosis

| Posterior | Mean error (mm) | Pred. std (mm) | 90% coverage | NEES / 3 | Mean NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| temperature-1 raw | 16.641 | 3.667 | 0.380 | 2529.8 | 3779.1 |
| temperature-128 raw | 17.219 | 8.796 | 0.796 | 3.037 | -7.306 |
| temperature-128 group inflation | 17.219 | 20.696 | 0.972 | 0.636 | -8.584 |

Tempering substantially reduces the earlier covariance collapse, but the raw
posterior remains overconfident. The three-group worst-case inflation then
overshoots: nominal 90% coverage becomes `0.972` and NEES/3 falls to `0.636`.
That is useful as a retrospective robustness diagnosis, not evidence of calibrated
uncertainty on new object/session groups.

## Scientific decision

Do **not** promote source-tuned evidence tempering or this zero-regret guard over
the last-supported-residual fallback. Retain the implementation and compact
negative evidence for reproducibility. Further weighting or guard tuning on the
same released cohort is unlikely to be decisive; the next claim-bearing step
requires fresh physical object/session executions and should prioritize model
discrepancy rather than another retrospective mixture-weight adjustment.

## Retained files

- `readout.json`: compact locked scientific readout;
- `per_case.csv`: deterministic LF case-level evidence;
- `artifact_manifest.json`: hashes for the compact files and the complete
  selection/summary artifact retained by the workflow run.

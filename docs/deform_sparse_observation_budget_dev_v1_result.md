# DEFORM Sparse Observation Budget: One-Case Diagnostic

## Decision

**Do not promote this query-aware selector or its uncertainty model.** The
registered one-case development comparison completed, but future-query
selection did not beat the strongest simple selector. A modest coordinate-L1
gain at the largest budget came from maximum-variance/global-information
selection, accompanied by worse 3D RMSE and severe undercoverage.

This does not reverse the existing DEFORM/local-residual positive result. Its
predictor, source fit, original evidence, and zero-measurement mean are unchanged.
It is a negative diagnostic for the new assumed graph-correlated, persistent
readout update on this trajectory, not a general rejection of active sensing.

## Frozen Experiment

- Source/config commit: `6772036a2ff6052ab5cd7561301f974908d35413`.
- Input archive SHA-256:
  `431c778022bfb7b602512e5e6c2132a3f42e5959c959368e5203059bd2ce223b`.
- Case: `103.pkl`, lexicographically first in the already-open DLO2 archive,
  selected before this experiment's outcomes were read.
- Forecast frames: 50-169. Scored identities: 3, 5, 7, 9, disjoint from all
  added prefix measurements. Known initial states and clamped actions remain
  available under the original DEFORM contract.
- Five policies, budgets 0/1/2/4/8, 32 random orders, and a separate 16-draw
  simulated shared-bias condition. Every policy has the same budget and update.
- 3,060 prediction records are repeated plans/noise conditions on **one**
  physical trajectory, not 3,060 independent executions.

The method, graph rank, noise assumptions, horizon, identities, case and budgets
were not changed after scoring. No fresh cohort, held-v8, DLO4 or DLO5 was used.

## Native-Annotation Result

Hidden-future mean coordinate absolute error, mm; lower is better:

| Policy | 0 measurements | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| Random, mean of 32 orders | 6.365 | 6.769 | 7.308 | 7.578 | 7.533 |
| Spatial/temporal spread | 6.365 | 6.365 | 6.628 | 6.628 | 7.145 |
| Maximum variance | 6.365 | 6.628 | 6.384 | 6.716 | 6.146 |
| Global information | 6.365 | 6.628 | 6.762 | 6.761 | 6.146 |
| Future-query information | 6.365 | 7.512 | 7.430 | 6.739 | 6.392 |

At eight measurements, maximum variance and global information have the same
conditioned prediction to numerical precision. Coordinate L1 improves 3.43%,
but point RMSE worsens 3.35%. Query-aware selection worsens coordinate L1 0.43%
and point RMSE 4.99%. None of the nonzero native budgets improves both metrics.

| Forecast | Point RMSE (mm) | Nominal 90% coordinate coverage | Mean full width (mm) | Coordinate NEES |
|---|---:|---:|---:|---:|
| Unchanged DEFORM/local-residual mean, 0 measurements | 13.851 | 99.03% | 55.860 | 0.356 |
| Maximum variance/global information, 8 measurements | 14.314 | 25.21% | 4.317 | 35.561 |
| Future-query information, 8 measurements | 14.542 | 23.82% | 4.294 | 37.242 |

These are descriptive marginal diagnostics on one trajectory. The initial
uncertainty is already conservative here; neither it nor the updated Gaussian
is established as calibrated. Shrinking the model covariance is not evidence
of shrinking the actual future error.

## Synthetic-Bias Condition

The separately labeled stress test adds a shared 5 mm-coordinate-STD Gaussian
translation and independent 1 mm noise to the prefix coordinates. At eight
measurements, global information gives 6.223 mm coordinate L1 (2.23% lower than
the unchanged mean); query-aware selection gives 6.873 mm (7.99% worse).
Global information still worsens point RMSE 3.18% and reaches only 31.83%
coordinate coverage. This supplies no positive real-sensor-bias or calibration
claim. The random draws are not independent physical replications.

## Implication for the Contribution

The next useful hypothesis is **whether source-learned residual cross-covariance
can predict the actual value of a new measurement for hidden future points**.
That must be established before optimizing an acquisition policy against it.
The present prior used a hand-specified graph correlation and persistent latent
field; its expected information gain did not reliably track real improvement.

A credible follow-up would fit/check that temporal and cross-identity
relationship on permitted training trajectories, retain a model-discrepancy
floor, and compare against the same simple selectors on other declared
development trajectories. This is a suggested follow-up, not permission to
retune this completed case or open a protected target. No larger confirmatory
study or additional backend build is justified by this result alone.

## Verification and Artifacts

The synthetic and existing DEFORM/information-planning regression suite passed
194 tests. Ruff and focused MyPy passed. Independent recomputation from the
saved prediction arrays matched all 3,060 coordinate-L1/RMSE/coverage records
and all 50 aggregated curve rows. All 612 zero-budget records matched the
original forecast's dtype, shape and C-order bytes. Nine output files and all
bound implementation files were rehashed successfully; PNG and PDF generation
completed, and the PNG was visually checked.

Compact exact artifacts are in
`results/sota/deform_sparse_observation_budget_dev_v1/`. The full run, including
the sealed prediction arrays, plot, per-record metrics, and generated report,
is preserved in the task workspace at
`deform-sparse-observation-budget-v1/run-v1/`.

- Full `results.json` SHA-256:
  `524e01007006ff51f804e14d80ccf30cb06761d27253e42541a59fd00c0cd2ee`.
- Full `run-complete.json` SHA-256:
  `52687a1c8e98abc4d02bf7139812e706df0244fd5f3ae29df67a7d6ef508c2d3`.

The result is source/development evidence only. There is no official point-SOTA,
fresh-object transfer, calibrated-safety, or ICRA-acceptance claim.

# DEFORM Predictive Coupling: Development Result

## Decision

**Do not promote this method or expand it to fresh targets.** Source-learned
cross-trajectory residual coupling, even with a fixed irreducible covariance
floor, did not improve the unchanged predictor. Future-query measurement
selection did not establish an advantage over simple selection. A nested
source guard prevented most harmful updates, but its best frozen setting had
only a small, inconclusive average gain and did not prevent every regression.

This is a completed experiment, not merely an implementation or runtime test.
It does not establish a stronger accuracy, calibration, active-sensing, SOTA,
or physical-mechanism claim. The previously successful DEFORM/local-residual
predictor and its original evidence remain unchanged.

## Evidence Boundary

- Source-frozen implementation: `cf853e5e6b5e4caab89e12d9ba99f31576dfc345`.
- Input archive SHA-256:
  `431c778022bfb7b602512e5e6c2132a3f42e5959c959368e5203059bd2ce223b`.
- Fourteen already-open DLO2 trajectories from **one physical object**;
  `103.pkl`, the previous design example, was training-only.
- Thirteen whole-trajectory outer holdouts, with nested source-only guard
  selection. A forecast never received its own future truth. Other folds'
  futures were used as cross-validation training data, explicitly.
- Disjoint observed/hidden identities; 0/1/2/4/8 prefix 3D point measurements;
  forecast frames 50-169; hidden identities 3/5/7/9; fixed 1 mm score-noise floor.
- Native released 3D coordinates, not an automatic camera/tracker provider.
- All 13 prediction sets were sealed before outer metric aggregation.
- No DLO4/DLO5, fresh target, held-v8, Causal4D acquisition, or GPU was accessed.

These are exploratory, same-object cross-validation results. They are not a
fresh-object confirmation or the official PhysTwin 22-case Chamfer/track table.
Coordinate L1 below is **not Chamfer distance**. Coverage is marginal 3D point
ellipsoid coverage, not the coordinate-wise coverage of the earlier budget
experiment and not simultaneous trajectory coverage.

## Matched Results

Equal-trajectory averages over all 13 holdouts. Random orders are averaged
within trajectory before aggregation. The table uses eight measurements
except for the unchanged zero-measurement predictor. All 156 frozen
method/policy/budget summaries and all 2,028 trajectory summaries are archived.

| Method | Selection | L1 (mm) | Point RMSE (mm) | 90% coverage | Joint wins / 13 |
|---|---|---:|---:|---:|---:|
| Unchanged DEFORM/local residual | None | 10.692 | 25.677 | 95.8% | Reference |
| Original graph persistence | Future query | 13.362 | 32.546 | 0.8% | 0 |
| Learned coupling, no floor | Future query | 16.548 | 38.082 | 0.5% | 0 |
| Learned coupling + floor | Future query | 12.939 | 30.220 | 58.1% | 0 |
| Permuted coupling + floor | Future query | 13.193 | 30.607 | 58.8% | 1 |
| Learned coupling + floor | Spatial spread | 11.330 | 26.570 | 67.1% | 3 |
| Last-residual interpolation | Latest uniform | 13.985 | 33.129 | 78.6% | 1 |
| Source-guarded floor | Future query | 10.692 | 25.677 | 95.8% | 0, exact fallback |
| Source-guarded floor | Spatial spread | 10.636 | 25.570 | 95.2% | 6 |

No unguarded, nonzero-budget arm jointly improved aggregate L1 and RMSE over
the unchanged predictor. The floor improves the severe overconfidence of the
no-floor and graph-persistence updates, but remains undercovered and less
accurate than the unchanged predictor. It is a useful diagnostic, not a
successful calibrated update.

At eight future-query measurements, the floor arm has mean point NEES `9.222`
and Gaussian NLL `-6.509`, versus `1.963` and `-8.807` for the unchanged
predictor. Lower NLL is better. The learned zero-measurement prior already
has only `74.4%` coverage: learning a second moment from the other 13
trajectories did not reproduce the original predictor's uncertainty quality.

## Guard Interpretation

The most favorable frozen setting was guarded spatial spread at budget eight:

- L1 change: `-0.0565 mm` (`-0.528%`); descriptive paired-trajectory 95%
  bootstrap interval `[-0.1551, +0.0310] mm`.
- RMSE change: `-0.1063 mm` (`-0.414%`); interval
  `[-0.3056, +0.0988] mm`.
- Nine of thirteen folds chose a 0.25 blend: six jointly improved, two
  regressed on both metrics, and one improved RMSE but worsened L1. Four
  folds returned exact fallback.
- Early/middle/late L1: `9.941 / 10.287 / 11.679 mm`, versus
  `10.081 / 10.232 / 11.763 mm` unchanged. There is no uniform horizon gain.
- NLL is `-8.828`, but this small change is not a calibrated-UQ confirmation.

Across the 312 nonzero-budget outer-fold/policy/budget settings, only eleven
received a nonzero guard blend. The two other admissions (one future-query
budget-one fold and one information budget-two fold) worsened aggregate
L1 and RMSE. Therefore this guard has **no demonstrated universal
non-worsening guarantee**. Prediction-row fallback counts include repeated
random schedules and must not be interpreted as independent physical trials.

The best setting is reported descriptively after comparing the frozen bank;
it is not a newly selected winner eligible for confirmation. Both intervals
include zero, fold training sets overlap, and all trajectories share one
object. These results do not justify a larger target study of this arm.

## Verification and Preservation

- `244` focused tests passed across the new method, synthetic positive and
  zero-coupling controls, numerical verifier, original budget experiment,
  information/query APIs, and DEFORM local-residual regressions.
- Ruff and focused MyPy passed; no frozen prediction code or parameters changed.
- A separate observation-space Gaussian solve, rather than the implementation's
  latent-precision solve, verified all **12,103** saved prediction records.
- Maximum mean disagreement: `1.30e-14 m`; covariance disagreement:
  `2.72e-16 m^2`.
- Recomputed all 2,028 trajectory summaries, 156 aggregates, and fixed-seed
  bootstrap intervals using Cholesky-based scoring.
- Verified **2,431** byte-identical zero-budget means and **2,394** byte-identical
  guarded mean/covariance fallbacks. There were eleven nonzero guarded records.
- Input, seal, barrier, result, and plot digests reverified. The numerical audit
  is not an independent empirical replication or a second human review.
- Original sparse-budget source/result files are byte-identical to commit
  `5ae2263e2b8061db9d205421e4f6f7c7bce7bdba`.

Key SHA-256 values:

```text
results.json:
a48f41dce9f02b42ce7401947007924ff7f6d00e73ba1f92ea44ece4fe17b4cd
prediction-barrier.json:
bc9cfc84b8a27d3880e809d46e43d2ae630ed9cb70b5d953733d82cef8dfc888
predictive-coupling.png:
302258bf276791dc41a0552dbfc350c68f3c7f8bdf552ad6015d934541f3501e
```

Compact evidence lives under
`results/sota/deform_predictive_coupling_dev_v1/`. The full write-once run is
preserved separately in the shared workspace:

```text
/mnt/c/users/emper/documents/codex/2026-08-25/where-are-we-with-cut3r/deform-predictive-coupling-v1/run-v1
```

## Consequence for the Contribution

This closes the tested **action-frame-aligned, trajectory-level empirical
residual covariance + fixed floor + Gaussian conditioning** proposal. It does
not reject learned conditional dynamics, Bayesian state estimation, active
sensing generally, or DEFORM itself.

Do not tune another selector or promote the tiny spatial gain on these
outcomes. A more substantive next hypothesis would change how an observation
affects the future: infer sparse current position/velocity and propagate that
state through the actual DEFORM dynamics, rather than apply another readout
field. That route needs a separately specified synthetic recovery test and an
opened-source comparison before any fresh cohort. It is not implemented or
validated by this result, and no new empirical authorization is implied.

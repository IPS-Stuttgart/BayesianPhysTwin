# Joint-event real-data experiment

Goal: genuinely test joint events on public recorded data, not the earlier
single-query variance ablation. No new robot interaction is required.

## Status

- [x] Audit the earlier scalar-query result and identify calibration cancellation.
- [x] Confirm no concurrent owner of this lane; create a separate worktree.
- [x] Freeze the proposed event bank, baselines, splits, costs, and claim limits.
- [ ] Validate metadata-only carrier availability for the original development roster.
- [x] Implement and test matched query marginals, genuine joint events, and boundaries.
- [ ] Commit the exact implementation and inventory before empirical execution.
- [ ] Run the one source-only real-recording experiment without retries or retuning.
- [ ] Independently verify prediction seals, scores, hashes, and clustered uncertainty.
- [ ] Preserve outcome-bearing evidence privately and report the strongest valid claim.

The original target recording is excluded even though this is a retrospective
development cohort. Exact historical source descriptors and sampled hashes are
extracted from development run 33329809775, archive SHA-256
`4f529b6a65a778b125f5294087862968e5251ac2d8363fd8e21fdc46373a5a16`.
Only source metadata is retained, not historical outcomes. The new evaluation
recording is the highest original source ID; all other original source recordings
fit the model and calibration. Live carrier additions cannot enter. This is
not a fresh object cohort or confirmation of the historical 92-object result.

Every probabilistic arm receives exactly the same point forecast and the same
finite set of marginal residual quantiles for each of the five queries. Rank
coupling alone differs. This follows ensemble copula coupling, an established
statistical method, not a new Bayesian theorem:
https://doi.org/10.1214/13-STS443

The point predictor and field covariance use the repository's existing v3 recipe.
Direct query covariance, an empirical residual copula, and direct logistic joint
event probabilities test whether the field-level representation adds anything
beyond simpler methods. Always-fallback is included at its literal cost of 0.1.

All five primary endpoints combine at least two queries. Thresholds come solely
from fit recordings. Windows remain grouped by physical object for uncertainty.
Original confirmation, reserved objects, held-v8, DLO4/DLO5, camera/geometry,
quarantined provider attempts, and other branches are outside this experiment.

No publication, fresh confirmation, uniquely Bayesian advantage, physical force
units, or robot safety claim follows automatically from an empirical win.

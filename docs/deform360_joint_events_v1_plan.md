# Joint-event real-data experiment

Goal: genuinely test joint events on public recorded data, not the earlier
single-query variance ablation. No new robot interaction is required.

## Status

- [x] Audit the earlier scalar-query result and identify calibration cancellation.
- [x] Confirm no concurrent owner of this lane; create a separate worktree.
- [x] Freeze the proposed event bank, baselines, splits, costs, and claim limits.
- [x] Validate the exact original-source carrier fingerprints for all 14 objects.
- [x] Implement and test matched query marginals, genuine joint events, and boundaries.
- [x] Commit implementation f752deaf; bind inventory before empirical execution.
- [x] Run the one source-only real-recording experiment without retries or retuning.
- [x] Independently verify prediction seals, scores, hashes, and clustered uncertainty.
- [x] Preserve outcome-bearing evidence privately and report the strongest valid claim.

Completion custody: the exact empirical implementation remains
`f752deaf9e272435a45b2616fbc727019cb83c62`. Its source archive SHA-256 is
`eaa15469a8f10e7631db6b938768f19354d81da644ab1a049414a688d7176f7c`.
The single run's result SHA-256 is
`6abe6d41551e67a0c56eecaa973e5813acd08f981a4ee2f35c016443d86a3428`;
the independent verification receipt SHA-256 is
`edf8e4a1939c65ab8f8a261ac668dcec12832b7c2d2dca08b8b22faabf9ce462`.
Outcomes, predictions, and interpretation are stored only in private
`FlorianPfaff/BayesianPhysTwin-Paper`, branch
`evidence/deform360-joint-events-v1`, commit
`7d52440f9b535f15496fdb78029cf4ac64679bc7`, under
`evidence/deform360_joint_events_v1/`. This public record contains no scores or
scientific promotion decision. No default branch or existing manuscript changed.

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

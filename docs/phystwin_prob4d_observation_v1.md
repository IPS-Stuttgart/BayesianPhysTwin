# Prob4D MotionCrafter observation path

Run date: 2026-07-18

Status: positive offline-reconstruction result; predictive endpoint transfer
rejected.

## Contract

This experiment replaces disjoint MotionCrafter windows with Prob4D decoded
uniform overlap fusion and keeps association probability separate from prior
perception reliability. Metric covariance, assignment-mixture spread, and
unknown cross-view correlation are propagated conservatively. The state
innovation enters exactly once through the robust mixture likelihood.

Arm C is the frozen decoded-uniform, decoupled-reliability path. Arm A is the
matched disjoint-window path. The previously examined 19-case cohort is
exploratory. The subsequent 11-case protocol was locked before evaluation.

## Offline reconstruction

On the exploratory 19 cases, arm C versus arm A changes equal-case CD by
`-2.28%` with 13/19 wins and manual-track error by `-1.61%` with 11/19 wins.
The locked exploratory transfer and source-calibration gates passed.

On the independent 11 cloth cases, equal-case CD improves from `10.498` to
`9.071 mm` (`-13.59%`) with 8/11 wins. The paired bootstrap interval for the
relative mean change is `[-23.04%, -3.43%]`. The preregistered replication is
nevertheless recorded as **not confirmed**: two cases do not have a finite
late-third metric under the fixed temporal partition, so the all-11 late-CD
gate is null. On the nine complete cases, the late direction is favorable but
its interval crosses zero.

These runs use all-frame MotionCrafter observations. They are reconstruction
controls, not future prediction. They support Prob4D as a better observation
operator and do not establish a better open-loop digital twin.

## Predictive endpoint diagnostic

A causal diagnostic retained only the final prefix graph correction and held
it constant through the untouched future. This worsens equal-case CD from
`11.122` to `12.596 mm` (`+13.25%`) and track error from `22.189` to
`22.882 mm` (`+3.12%`), with only 6/19 two-metric wins.

A nested prefix-only gate falls back on 18/19 cases. Its sole accepted case,
`single_push_sloth`, transfers positively, but that isolated result cannot
close the published SOTA gap.

## Conclusion

Prob4D overlap fusion improves framewise MotionCrafter geometry, especially on
the independent cloth reconstruction cohort. Its graph correction is not a
quasi-static future anchor. The next predictive family must learn how residual
motion evolves under actions and must be selected by recursive multi-step
rollout. Static endpoint persistence is closed.

Machine-readable evidence is archived under
`results/sota/diagnostics/prob4d_observation_v1/`. The independent protocol
retains its explicit prohibition on future-prediction claims.

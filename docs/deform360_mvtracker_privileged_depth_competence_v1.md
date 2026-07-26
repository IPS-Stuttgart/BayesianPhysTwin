# Deform360 MVTracker Privileged-Depth Competence V1

## Question

Can a learned multiview 3D tracker recover useful material-point motion on
Deform360 when depth quality is deliberately made favorable?

This is an inexpensive rejection test for MVTracker, not a deployable
Bayesian-PhysTwin experiment. Deform360's public cameras are RGB-only. The
available per-frame `rendered_depth.h5` files were rendered from a
full-sequence splat, so they carry privileged reconstruction evidence. They
must not enter a predictive result.

## Frozen source case

The case is `092-squirrel-ep0008`. It was selected from the already-open
27-episode development panel using only sealed physical predictions:

```text
select argmax_case quantile_0.95(
    ||x_physical(frame 19) - x_frame_zero||
)
```

No camera future, target trajectory, outcome metric, or tracker output entered
selection. The selected 95th-percentile predicted response is `41.542 mm`.
Six of the 16 identities in the pre-existing frozen observation plan have at
least `5 mm` predicted motion at frame 19.

## Frozen observation

- MVTracker repository revision:
  `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`
- checkpoint SHA-256:
  `a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6`
- RGB and rendered-depth indices: `0..19`
- queries: the 16 material identities from the frozen open-27 camera plan
- cameras: all eight cameras from that plan
- scene normalization: MVTracker's frame-zero camera/depth normalization
- target access during prediction: none

MVTracker does not enforce an exact query-frame identity. The adapter therefore
subtracts each raw query-frame offset from the complete predicted trajectory.
This preserves every predicted displacement while making frame zero exactly
equal to the sealed material identity.

Observation variance is in square metres. It is computed from MVTracker
visibility and the query-frame anchoring correction. The physical-state
innovation is not used as prior perception reliability and may be processed
only later by the robust Bayesian likelihood.

## Seal and score

Prediction arrays and their report are checksummed and sealed before
`target_data.pkl` or `outcome.json` is loaded. Scoring then uses frames `1..19`
on common target-valid and MVTracker-visible identity support.

The control passes only if all three conditions hold:

1. at least 75% of eligible identity-frames are supported;
2. identity RMSE is at least 10% below the better of the sealed physical prior
   and exact persistence on the same support;
3. identity RMSE is at most 10 mm.

A pass authorizes one new experiment: prefix-only learned depth, generated
without frames after the current update, followed by the same frozen tracker
and gate. It does not authorize fresh objects or a Bayesian-PhysTwin claim.

A failure stops the MVTracker route. It would show that even favorable
full-sequence reconstruction depth cannot make this tracker beat the existing
belief baselines on an informative source response.

## Claim boundary

This protocol is an outcome-open, privileged-depth competence control. It is
not causal observation evidence, a Bayesian-PhysTwin improvement, a fresh
evaluation, confirmation, or a state-of-the-art result. It does not read or
modify held-v8.

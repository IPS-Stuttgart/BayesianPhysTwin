# Native Wrapping Belief Source Result

Status: complete; all native checks pass; the registered source-value gate
fails. There is a positive finite-prior decision signal, but no promoted method
or authorized larger evaluation. Work remains local/private, with no push or
main merge. Existing DEFORM code, predictions, and results are unchanged.

## Frozen Execution

The pre-outcome implementation/protocol commit is
`e184ac01f4de0791844da84bbdf6c5904c10973e`.
The unchanged public DLO-Lab wrapping task uses a 50-vertex extensible loop,
three posts, two native robot controllers, eight motions, and a duplicate.
Nine specified bending/stretching settings form the uniform model prior.
Three nominal repeats plus eight other settings give eleven batches.

All 11/11 batches and 99/99 native trajectories complete and pass qualification.
There are zero technical failures, unsealable results, replacements, retries,
or unrun batches. Each task claim precedes native initialization; every native
bundle is sealed before checks, and the complete source bank is sealed before
belief-value calculation. No GPU, real robot, recording, protected dataset,
held-v8, DLO4/DLO5, official DLO3 evaluation, PokeFlex continuation, or fresh
Deform360 data is used. The native scene, reward, solver, controller, and contact
remain unchanged; only existing material randomization hooks are fixed.

The local exact run is at
`/home/fpfaff/source-only/dlolab-wrapping-belief-source-v1`.
Compact custody/decision evidence is in
`results/sota/dlolab_wrapping_belief_source_v1/`.
That directory name is repository convention, not a SOTA claim.

## Decision Results

These are expected unchanged native final rewards over the finite nine-world
prior and the registered synthetic observation-noise model. Higher is better.
The world prior is also the model support, not independent held-out materials.

| Decision | Expected native reward |
|---|---:|
| Hold after common prefix | 0.488326649 |
| Nominal material's best action | 0.886306908 |
| Best fixed action over the source prior | 0.889813721 |
| Posterior ignoring shared observation bias | 0.906145287 |
| Bias-aware MAP material, then best action | 0.912913096 |
| Bias-aware posterior expected-reward action | **0.915810660** |
| Perfect-information per-world oracle | 0.916743338 |

The Bayesian decision gains 0.025996939 reward, or 2.92% of fixed-action reward.
This corresponds to 23.59% of the fixed action's remaining deficit from reward 1;
it is not a 23.59% increase in reward. Its raw gain over MAP is 0.002897564 and
over ignored-bias inference is 0.009665373. Its Monte Carlo standard error is
0.000018862, integrating only the assumed observation noise, not uncertainty
across independent physical experiments or unknown materials.

The fixed action finishes wider; the nominal material favors finishing narrower.
The Bayesian policy chooses narrow finish about 81.32% of the time, early lowering
11.23%, and wide finish 7.45%. The finite-world oracle favors two different
motions. The allowed prefix therefore carries useful decision information for
this task and model bank, unlike a purely dominant-action test.

## Registered Gate

Six of eight checks pass, including the best-fixed versus hold comparison, the
0.02 adjusted Bayesian gain over best fixed, its relative-deficit check, distinct
oracle actions, at least three worlds with useful per-world headroom, and the
adjusted comparison with ignored-bias inference. Two fail:

| Required quantity | Observed | Required | Decision |
|---|---:|---:|---|
| Oracle gain over best fixed minus 0.002 | 0.024929617 | >=0.05 | Fail |
| Bayesian gain over MAP minus 0.002 | 0.000897564 | >=0.002 | Fail |

The result does not pass merely because the raw mean improves. The Bayesian
policy captures 96.54% of the available raw finite-bank oracle headroom, but that
headroom is only 0.026929617 reward. This percentage is neither a task-improvement
percentage nor a generalization claim. No threshold is relaxed after the result,
and no new action bank or larger study is authorized as a rescue.

## Native and Software Verification

All ten per-batch checks pass. The largest common-prefix coordinate difference
is 2.66e-13 m, the duplicate difference 1.77e-10 m, and grasp attachment error
3.52e-6 m. Closed-segment length ratios range from 0.94766 to 1.88413, within the
registered broad sanity bound for this extensible loop. This does not validate
real material stiffness or assert inextensibility.

The three nominal repeats have maximum coordinate spread 2.28e-8 m and identical
reported float32 rewards. Native final reward reconstruction error is at most
2.96e-8; cumulative float32 rewards, including the native +1 offset, match exactly.
Observed replay ranges are not population error bounds or byte-identity claims.

Before execution, 66 new focused tests and the complete relevant DEFORM/DLO-Lab
regression selection passed (945 tests total). Ruff and focused MyPy on all four
new source/runner/verifier modules pass. The tests include an uninformative
prefix, identifiable-material oracle recovery, and ambiguous materials where
posterior expected reward correctly beats MAP. These are implementation checks,
not empirical successes. Pytest uses a native POSIX temporary directory and
`--capture=sys` because this host's Windows-backed default FD capture failed
before collection; no simulator retry is involved.

The separately implemented verifier checks 15 committed source files, all 99
native trajectories, 10,890 native micro-reward rows, repeatability, the sealed
prefix/future split, all decisions, and all eight gates. It uses angular
unwrapping for winding and Sherman-Morrison precision instead of the production
cross/dot winding and Cholesky whitening. Maximum arithmetic difference is
1.887379141862766e-15. It passes and reproduces the failed gate. This is an
alternate arithmetic implementation, not independent human review.

## Provenance

- Lock ID: `70e6054141a5652957590f5b173c36ccff99cc167b48a3f8b4f085ba4be20a31`.
- Lock file SHA-256: `b689b17db607d79bb9b7642a5ad76a25591f7e85902ccd08ac01d7e6dc970bbc`.
- Result ID: `5be8f1a54ac38e9dfc0745a5722a9490d8fa41299ca66080714caa8612a09ff0`.
- Result file SHA-256: `550b04bceab58d14f78f020a3870841ffae06e6ce8946d996f8418e868bacf9c`.
- Source bank NPZ SHA-256: `914bd948df92e8b829ac65ca8c075c789d122a63a9ec32807a302bef16e2271d`.
- Alternate verifier ID: `538f10c8722e5e8cf8022fa7c3c326954aeeb602dde9e089887a176a64e5fd4f`.
- Verifier receipt SHA-256: `8413aac8cf7c6638c07df2ce8de359f084189871578d937e3f9d5980011f5f77`.

## Interpretation

This is stronger source decision value than the earlier wiring screen: a noisy
prefix and a correlation-aware belief can select better native motions than a
fixed action, with unchanged physics and a strong fixed-action comparator.
However, all worlds lie on the exact assumed prior support, the sensor model is
synthetic and known, and the primary advancement gate fails. It is exploratory
mechanism evidence, not independently validated Bayesian control, calibrated
real sensing, a new theorem, official benchmark parity, or SOTA.

Keep the successful paired DEFORM update unchanged. Preserve this source signal
and both failed thresholds together; do not promote the wrapping arm or claim
that ordinary Bayesian expected-utility selection is itself a novel method.

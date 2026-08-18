# Conditional DEFORM long-run checkpoint posterior

This method is dormant unless the frozen long-run DLO1 source gate passes. It
cannot weaken, bypass, or reinterpret that gate.

## Registered candidates

For each locked late-checkpoint arm, validation compares two operators:

1. `parameter_mean`: weighted checkpoint tensors are averaged in double
   precision before one physical rollout;
2. `predictive_mean`: every checkpoint is rolled out physically and the
   posterior predictive trajectories are averaged.

The arm bank contains uniform means over updates 6040/6400,
5200/6040/6400, and 4000/5200/6040/6400, plus a validation-softmax arm over
updates 2560/4000/5200/6040/6400. The softmax temperature is fixed at 1 mm.

Only DLO1 validation trajectories select among these candidates. A candidate
must improve validation mean coordinate L1 by at least one percent; otherwise
the output is the exact selected single checkpoint. Only the selected candidate
is evaluated on the already-open DLO1 source split.

## Uncertainty

Checkpoint spread provides a diagonal coordinate variance. A 5 mm standard
deviation floor represents unresolved model/replay variation. One nonshrinking
scale is fitted on validation coordinates and frozen before source evaluation.
Coverage, interval width, and Gaussian NLL are reported as coordinate-marginal
diagnostics, not as an independent calibration claim.

## Advancement

The checkpoint posterior advances to fresh DLO2 only when all of the following
hold:

- the parent long-run source gate passed;
- a nonfallback candidate improved DLO1 validation by at least one percent;
- that candidate improved the already-open DLO1 source mean by at least one
  percent; and
- at least five of eight source trajectories improved.

Passing authorizes the same frozen arm bank on a from-scratch DLO2
reproduction. It does not authorize DLO1 official evaluation or a SOTA claim.

## Command

```bash
python scripts/remote/run_deform_dlo_longrun_posterior.py \
  --protocol configs/sota/deform_dlo_longrun_posterior_v1.json \
  --longrun-protocol configs/sota/deform_dlo_longrun_v2.json \
  --longrun-result /path/to/longrun_result.json \
  --source-manifest results/sota/deform_dlo_source_v1/source_manifest.json \
  --upstream-root /path/to/DEFORM \
  --output-root /path/to/posterior-output \
  --device cuda:0
```

The runner verifies all checkpoint, schedule, protocol, source-manifest, and
runtime identities and installs the same official-evaluation read guard as the
parent run.

The posterior policy is a separate immutable artifact from the executing
long-run protocol. This prevents later posterior development from changing the
bytes to which the already-running parent is bound.

A validation-only parity smoke at implementation commit `04c40a9` reproduced
the parent update-280 rollout to within `7.7e-8 m` in model L1 and `2.1e-9 m`
in persistence L1. The checksummed record is stored under
`results/sota/deform_dlo_longrun_posterior_v1/`.

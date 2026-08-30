# Deform360 DLO cross-episode pilot v1

This pilot turns the existing fragmented Deform360 holdings on `gpuserver6000`
into a frozen, reproducible source-to-held-out-episode development panel.

## Registered objects

The homogeneous DLO-like roster is:

- `001-rope`
- `002-rope-silk`
- `003-cable`
- `081-stripe-rope`

The physical object is the independent experimental unit. Episodes, actions,
cameras, frames, and tactile samples are nested observations.

## First execution stage

The file-triggered workflow first reads only raw directory names, file names,
file sizes, and each object's small `metadata.json`. It then freezes one source
and one target episode for every object, preferring a different action family
and requiring at least 32 exact camera/video timestamp pairs and four exact
tactile/timestamp pairs per selected episode.

Only after the plan has been content-addressed does the workflow run the pinned
official Deform360 processing revision. For the eight frozen episodes it creates:

1. synchronized and rectified multiview RGB;
2. synchronized tactile grids; and
3. ArUco-derived robot/gripper state in the released `robot.npz` contract.

Outputs are written outside the raw tree under:

```text
/mnt/lexar4tb/datasets/deform360/results/bayesian-phystwin/
  deform360-dlo-cross-episode-pilot-v1/<repository-sha>/<plan-id>/
```

The compact GitHub Actions artifact contains the frozen plan, environment,
per-object logs, verification result, execution receipt, and checksums. It does
not duplicate the processed video payloads.

## Explicit boundary

This is a retrospective preprocessing and action-carrier pilot. It does not run
object masks, Gaussian-splat reconstruction, rendered depth, dense tracking,
cleaned point clouds, PhysTwin control-point materialization, physical-parameter
inference, uncertainty calibration, or target-future scoring.

A successful run therefore proves only that all four physical objects provide a
uniform synchronized RGB–tactile–robot input contract for the next stage. It is
not a model-accuracy result, a physical-transport result, a fresh confirmation,
or paper-level evidence.

## Next gate

After this workflow succeeds, the next protocol can authorize geometry
materialization and a sealed source-only physical posterior. Predictions for the
four held-out target episodes must be written before target-future scoring. The
principal later comparison should include global physics, deterministic system
identification, Bayesian physical transport, residual persistence, wrong-object
posteriors, and action-response relation-breaking controls.

# RGBench Public-Backbone Source Diagnostics

## Scope

These are post-open diagnostics on the 27 already-open RGBench source cases.
They do not modify the frozen isotropic-dynamic v2 method, authorize
calibration, or open calibration or target coordinates.

## Endpoint-cloud persistence

The first diagnostic keeps the frozen physical prediction throughout the
permitted prefix, then repeats the final observed prefix cloud for every
future frame. This is the exact analogue of a no-dynamics endpoint-persistence
control.

| Source-balanced quantity | Physical rollout | Endpoint persistence |
| --- | ---: | ---: |
| Full one-sided real-to-prediction L1 | 45.62 mm | 136.64 mm |
| Future-only one-sided real-to-prediction L1 | 53.45 mm | 177.14 mm |
| Improved garment/action cells | - | 0 / 9 |
| Cells below published GarmentDynamics | - | 0 / 9 |

Endpoint persistence regresses by 199.50% when comparing the two balanced
means, and every garment/action cell regresses. The action-only persistence
advantage observed in some Deform360 windows does not transfer to RGBench:
these garment futures contain large commanded motion for which a dynamic
rollout is essential.

The compact artifact is
`results/sota/rgbbench_isotropic_dynamic_v2/source_endpoint_persistence_diagnostic.json`.

## Public plain-MuJoCo wrapper

The official RGBench plain-MuJoCo Flex wrapper was run on the already-open
`green_tshirt/grasp/01` source case with the upstream fixed-point Piper
configuration. Its official evaluation-window mean one-sided
real-to-simulation L1 error was 244.96 mm. For context, the remeshed PyBullet
source baseline on that case was 31.87 mm, while the published
GarmentDynamics three-sample cell mean is 22.6 mm.

The run also emitted 65 unstable-vertex warnings, ranging from 6 to 647
vertices below the ground plane. This is a competence failure, not a useful
backbone whose remaining error should be handled by online Bayesian
assimilation.

The compact artifact is
`results/sota/rgbbench_isotropic_dynamic_v2/source_plain_mujoco_smoke.json`.

## Decision

Close both alternatives:

1. Do not replace the RGBench physical rollout with endpoint persistence.
2. Do not build another discrepancy model around the released plain-MuJoCo
   wrapper.
3. Do not expand the failed v2 temporal-shrinkage bank.
4. Keep calibration and target coordinates sealed.

The RGBench SOTA gap now lies in the physical backbone. A new RGBench method
is justified only if a substantially stronger public dynamics model becomes
available, or if the published GarmentDynamics implementation or trajectories
are released. In the meantime, the demonstrated SOTA-facing Bayesian route
remains the physical/action-supported guarded online belief on Deform360,
under its independently owned prospective evaluation.

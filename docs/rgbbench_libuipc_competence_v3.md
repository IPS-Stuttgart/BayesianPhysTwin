# RGBench LibuIPC Competence v3

## Purpose

RGBench v2 showed that a graph-discrepancy continuation around the public
PyBullet rollout has only 3.04% source headroom even with a per-cell oracle.
The remaining gap to published GarmentDynamics is therefore a physical
backbone problem, not a reason to add another readout correction.

This protocol tests one materially different public backbone:
LibuIPC `v0.0.7`, using its strain-limited shell, discrete bending, robust
contact, friction, and soft vertex-position constraints. The first test is
only a deterministic runtime competence gate on the already-open
`green_tshirt/fling/01` source case.

## Frozen Mapping

- RGBench `stretch` is passed as shell Young's modulus in Pa.
- RGBench `density` is passed as volumetric density in kg/m3.
- The source metadata's Poisson ratio, bending value, and friction coefficient
  are passed directly.
- Shell thickness is fixed at 1 mm for this competence test.
- The two released shoulder indices are driven with the same preparation,
  wait, interpolation, and controller-frame offsets as RGBench fixed-point
  PyBullet.
- The frozen RGBench configuration uses a 5 s preparation and 5 s wait before
  replaying the measured fling trajectory.
- The simulator timestep is 10 ms. No real point cloud is used to choose it.

The competence run may read the source mesh, material metadata, and recorded
actuator trajectories. It must not enumerate or parse any segmented point
cloud. Calibration and target garments remain sealed.

## Gate

Two independent process replays must complete, preserve the vertex count,
produce only finite vertices, and have byte-identical final vertex arrays.
Failure closes this arm before accuracy evaluation.

Passing this gate authorizes only a separately frozen source accuracy study.
It does not authorize calibration or target access, and it does not establish
that LibuIPC is more accurate than GarmentDynamics.

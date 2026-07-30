# RGBench ARCSim Dirichlet Competence v10

## Question

Can ARCSim provide deterministic, identity-preserving full-resolution cloth
motion when the two measured RGBench actuator trajectories are enforced as
time-varying Dirichlet boundary conditions?

## Why This Is A New Method

The frozen v9 gate read no point-cloud filename, coordinate, or accuracy
outcome. It showed only that ARCSim's native penalty handles missed the known
actuator targets by 21.7585 mm. That violates the experiment's action contract
before cloth accuracy can be assessed.

V10 changes one model interface: after every ARCSim substep, the two declared
handle nodes are placed at their known targets and assigned the corresponding
finite-difference velocity. The native handle penalty remains active during the
implicit solve so neighboring cloth nodes still receive the moving-boundary
load. This is an in-solver time-varying Dirichlet condition, not a correction
applied to saved predictions.

The source case, mesh, material parameters, timestep, horizon, disabled
mechanisms, and competence thresholds are unchanged from v9.

## Information Boundary

Before this gate passes, the runner may read only the released source mesh,
material metadata, two actuator trajectories, simulator outputs, pin error,
motion magnitude, replay equality, and runtime. It must not enumerate or read
any segmented point cloud or any source, calibration, or target accuracy
outcome.

## Decision

- **Pass:** freeze a separate full-horizon target-free qualification.
- **Fail:** close the kinematic ARCSim backend without point-cloud scoring.

Passing this gate would establish control fidelity and numerical competence
only. It would not establish predictive accuracy or state of the art.

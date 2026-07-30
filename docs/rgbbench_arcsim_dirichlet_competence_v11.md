# RGBench ARCSim Dirichlet Competence v11

## Correction

V10 proved that per-substep kinematic enforcement alone was insufficient
because ARCSim lazily initialized each handle reference after gravity
relaxation. The resulting boundary was exact around a 21.7585 mm shifted
reference.

V11 initializes and applies the two declared kinematic handles immediately
before relaxation, reapplies them afterward, and continues enforcing them after
every substep. No point-cloud filename, coordinate, or accuracy outcome
informed this correction.

Everything else remains frozen from v10: source case, mesh, material
parameters, timestep, horizon, penalty forces during the implicit solve,
disabled mechanisms, and numerical competence thresholds.

## Decision

- **Pass:** freeze a separate target-free full-horizon qualification.
- **Fail:** close ARCSim as an RGBench backbone without point-cloud scoring.

This is the final control-interface correction in this solver line. Passing
would establish numerical and action-contract competence only, not predictive
accuracy.

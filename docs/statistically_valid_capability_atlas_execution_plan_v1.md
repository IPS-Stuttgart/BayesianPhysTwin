# Executable integration plan

1. Rebase the trajectory-level conformal calibration implementation onto the exact capability-atlas branch.
2. Add an API that accepts an `AffineCapabilityHalfspacesV1` region and returns the same region with all offsets reduced by a nonnegative calibration correction.
3. Add a combined objective-uncertainty helper using the existing support-function module.
4. Use one maximum score per complete object or trajectory across every routed window, action comparison, and registered task.
5. Test finite-sample rank selection, monotone region contraction, exact half-space shifting, and fail-closed behavior when the requested miscoverage is too small for the calibration sample.
6. Keep already opened DEFORM results retrospective. A future-object claim requires an object-disjoint, untouched target cohort.

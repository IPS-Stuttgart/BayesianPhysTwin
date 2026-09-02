# Statistically valid decision-capability atlas: integration boundary

This branch supplies trajectory-level conformal regret calibration. It is intended to compose with the exact affine decision-capability atlas, not replace its model-side certificate.

For one independent calibration unit, the conformity score must aggregate the complete routed decision procedure over all registered decision windows and downstream tasks before a single object- or trajectory-level score is emitted. Calibration units, rather than windows, are the exchangeability units.

The composed admission rule adds a nonnegative split-conformal correction to every model-side pairwise regret half-space. For affine task families this preserves polyhedral capability regions. Objective-uncertainty support-function corrections and the data-derived correction are additive.

The statistical statement is finite-sample and marginal over a future exchangeable physical unit. It is not conditional coverage for every object class, a guarantee under arbitrary distribution shift, a validation of the task loss, or a deployment-safety certificate. Reusing already opened DEFORM target trajectories can provide retrospective mechanism evidence only; a primary future-object claim requires an untouched object-disjoint calibration/target protocol.

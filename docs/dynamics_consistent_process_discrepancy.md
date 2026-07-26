# Dynamics-consistent latent process discrepancy

`bayesian_phystwin.process_discrepancy` provides a simulator-agnostic Bayesian
process-force model for cases where prospective evidence localizes residual
error to the physical dynamics rather than to the observation/readout layer.
It does **not** change any frozen PhysTwin experiment, released baseline, or
paper claim.

## Model

The nodal discrepancy force is represented in a constrained low-rank span,

\[
    f_t = B_{\mathrm{force}} c_t,
\]

where the starting span comes from a graph basis and the latent coefficients
follow a stable AR(1) process,

\[
    c_t = \phi c_{t-1} + \epsilon_t, \qquad |\phi| < 1.
\]

The implementation keeps a Gaussian posterior over `c_t`, propagates it with a
stationary process covariance, and conditions it on inverse-dynamics force
observations with optional reliability weights.

## Physical constraints

`build_process_discrepancy_basis()` projects the graph force span into hard
nullspaces for zero internal net force and zero internal net torque. Two masks
make the construction contact-aware:

- `support_weights` localize or suppress the discrepancy at individual nodes;
- `externally_supported` marks contact or attachment nodes that may exchange
  momentum with the environment and are therefore excluded from the internal
  momentum balance.

The resulting `force_operator` has orthonormal columns, reproduces the weighted
graph span, and satisfies every declared hard constraint to numerical
precision.

## Work regularization

When `local_power_prior_std_w` is configured and node velocities are supplied,
Bayesian conditioning adds one soft pseudo observation per active node,

\[
    v_i^\mathsf{T} f_i \sim \mathcal N(0, \sigma_P^2).
\]

This penalizes unsupported local energy injection without imposing a hard
zero-work assumption. The AR(1) prior supplies the temporal regularization.

## Reproducibility and claim boundary

`ProcessDiscrepancyFitBoundaryV1` fails closed when future or target outcomes
were used for fitting or model selection. It binds a method freeze, split,
released baseline, and readout-only comparator. `ProcessDiscrepancyModelV1`
then exposes a content-derived `model_id` that can be recorded in
`RunManifestV2` metadata.

The module is a research component, not evidence that process discrepancy is
beneficial. A confirmatory evaluation still has to compare, under one frozen
protocol:

1. the unchanged released PhysTwin baseline;
2. a readout-only discrepancy comparator;
3. the process-discrepancy candidate;
4. zero-correction parity and guard/fallback behavior.

Model selection must remain source-only or development-only. Held-out
continuation outcomes must not influence the basis rank, support mask, AR
coefficient, force scale, work prior, or acceptance rule.

## Zero-correction parity

`apply_process_discrepancy_force()` returns the original nominal NumPy array
unchanged when the model is disabled or the discrepancy is exactly zero. This
preserves value, dtype, memory layout, byte representation, and object identity
for the released simulator path.

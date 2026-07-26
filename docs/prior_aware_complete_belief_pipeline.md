# Prior-aware complete-belief update

The deployable Bayesian-PhysTwin update is stricter than a numerically valid
camera correction. A candidate must pass five separate boundaries:

1. **Structural observation support.** Correspondences require redundant
   geometric or independently sensed support. Pairwise consistency alone is
   useful for identity swaps but cannot reject a coherent common camera bias.
2. **Action-conditioned physical support.** The observed prefix must contain a
   response compatible with the applied action and the physical prior.
3. **Prior-aware robust inference.** Gauge, shared camera bias, view bias, and
   state remain separate latent variables. Correlated rows are grouped, and the
   state innovation enters the robust mixture once.
4. **Nonlinear PhysTwin closure.** A local state update must agree with a
   nonlinear replay within frozen absolute or relative tolerances.
5. **Source-fitted regret acceptance.** A baseline-relative certificate decides
   deployment without reading the future target.

`run_prior_aware_guarded_belief_update` binds these decisions to the
`ObservationBeliefV1`, `PhysicalLinearizationV1`, baseline belief, candidate
belief, and common query domain. Rejection at any stage returns the original
baseline belief object, preserving state, parameter particles, discrepancy,
nuisance moments, covariance, dtype, and provenance.

## Causal and calibration boundary

Structural and physical support are routing evidence, not perception
reliability. They cannot alter `prior_reliability`, and the future target cannot
enter either decision. They also cannot process the candidate state innovation
as a second likelihood term. Association support remains distinct from metric
observation covariance; assignment ambiguity must already be represented by
the feeder's mixture covariance.

The final decision content-binds the baseline, linearized, nonlinear, and
regret-selected query arrays, the nonlinear closure, the source certificate,
and both complete beliefs. A closure result cannot therefore be replayed
against a different candidate payload without changing the decision ID.

The posterior covariance is a local working covariance. Nonlinear closure does
not calibrate it, and exact fallback does not establish non-regression. Accuracy,
coverage, and baseline-relative regret still require a genuinely fresh,
object-clustered prospective evaluation. The opened Deform360-27 results may
motivate this construction but cannot confirm it.

## Source evidence boundary

The opened PhysTwin-22 manual-prefix capacity ceiling is
`7.891873 mm` CD and `13.429357 mm` manual-track error. It uses released manual
trajectories from the permitted prefix as the sparse identity channel and then
scores the same identity family in the future. It is useful headroom evidence,
but it is neither automatic nor a fair state-of-the-art result.

The completed automatic CoTracker3 multiview arm is separate. It reached
`10.627 mm` CD and `20.415 mm` manual-track error and failed its advancement
gate. Its 22 causal cue archives already exist; regenerating them would not
create independent evidence. A deployable claim for this pipeline must use
automatic prefix observations, disjoint hidden future identities, and a new
prospective cohort.

## Required controls

The source-safe synthetic controls cover:

- identifiable action-supported deformation, which may select the candidate;
- coherent global camera translation with a diffuse bias prior, which must not
  become a confident state update;
- missing structural or physical support, which must fall back exactly;
- a failed nonlinear closure check, which must fall back exactly; and
- a rejected source regret certificate, which must fall back exactly.

The next empirical step is to freeze a complete candidate configuration and
evaluate it on objects excluded by every prior source, calibration, reserved,
and technically dispositioned cohort. No held-v8 target, query, score, barrier,
or outcome artifact is needed to implement or test this module.

# Pairwise Bias-Aware Belief: Source V1

## Status

Implementation-locked source-development candidate. No real combined-method
outcome has been computed yet.

This route may use only the 27 already-open Deform360 development episodes.
It must not inspect held-v8, the failed selective-virtual-sensing target cohort,
or any fresh object. A source result can justify a new prospective protocol; it
cannot establish state of the art, calibration, or non-regression.

## Question

The strongest open-source camera result uses a pairwise-consensus gate followed
by a persistent RBF correction. It improves hidden identity RMSE and hidden
Chamfer substantially, but a rigid-invariant clique cannot detect coherent
camera bias. The separate bias-aware physical-response update rejects that
ambiguity, but its open-source gain is much smaller.

The v1 candidate asks whether those mechanisms compose:

```text
causal physical response
        + pairwise material-identity consistency
        + nuisance-marginalized observation selection
        + bias-aware state update
        -> guarded state correction or exact baseline
```

## Frozen Candidate

At update frames 19, 38, and 57:

1. Start from the already selected physical/persistence backbone.
2. Admit only finite observations with positive residual-independent prior
   reliability.
3. Find the frozen pairwise-consensus clique using the existing 30 mm plus 10%
   strain envelope, minimum nine inliers, and 70% support.
4. Require at least three causal motion centers, 0.5 mm physical response,
   0.5 mm observed motion, and physical-response agreement of at least 0.40.
5. Build the rank-4 physical-response basis from prefix-only simulated motion.
6. Remove physical modes that are less than 10% identifiable beyond the shared
   spatial and global-bias basis.
7. Select at most 12 clique members by nuisance-marginalized information gain.
   The total reliability mass is capped at eight effective observations.
8. Run the existing Student-t bias-aware update once and propagate only its
   reachable state component. Coherent shared and global bias is not added to
   the physical state.
9. Multiply the decoded state correction by the causal physical-agreement gain.
10. On every rejection or numerical failure, return the selected backbone
    byte-for-byte.

Candidate geometry determines association consistency. Prior reliability uses
triangulation redundancy, reprojection quality, and covariance, never the
innovation against the twin. The state innovation enters once through the
bias-aware robust likelihood.

## Synthetic Controls

The implementation must continue to pass:

- three large identity swaps are excluded by the pairwise clique;
- a 10 mm coherent observation translation is assigned to nuisance bias while
  the local physical mode is recovered;
- a common translation without matching physical response gives exact fallback;
- changing a global state innovation does not alter prior selection reliability;
- center-array permutation does not change selected material identities;
- duplicated/correlated evidence cannot exceed the fixed effective information
  mass;
- insufficient pairwise support gives byte-exact fallback.

These are implementation and mechanism controls, not empirical evidence.

## Frozen Source Arms

The source run must report the same hidden identities and future frames for:

| Arm | Purpose |
| --- | --- |
| Selected raw backbone | Exact no-update reference |
| Pairwise-consensus RBF | Strong open-27 camera baseline |
| Bias-aware v4 | Physical-support and nuisance-model control |
| Pairwise bias-aware v1 | New candidate |

Report object-balanced hidden identity RMSE, hidden symmetric Chamfer, late
errors, update coverage, exact-fallback count, selected-center count,
state/bias posterior correlation, and object-cluster bootstrap intervals.

## Advancement Gates

A new fresh-object accuracy protocol is justified only if all conditions hold:

1. both co-primary object-balanced metrics improve by at least 1% over the
   pairwise-consensus RBF arm;
2. both object-clustered 95% difference intervals have upper endpoints below
   zero;
3. at least four of five object means improve on both metrics;
4. no object regresses by more than 2% on either metric;
5. every rejected interval is byte-identical to its selected backbone;
6. all accepted updates satisfy the frozen pairwise, physical-response,
   identifiability, and information-support gates.

Failure closes this composition on the opened source cohort. Hyperparameters
must not be changed after source outcomes are computed.

## Information Boundary

- no future observation may construct a candidate;
- no target trajectory is accepted by the prediction function;
- open source targets are used only by the separate evaluator;
- the existing held-v8 and fresh-object exclusion boundaries remain unchanged;
- the unavailable held-v8 all-attempt exclusion manifest still blocks any new
  Deform360 object selection.

The implementation entry point is
`predict_pairwise_bias_aware_candidate_arrays` in
`bayesian_phystwin.deform360_pairwise_bias_aware_development`.

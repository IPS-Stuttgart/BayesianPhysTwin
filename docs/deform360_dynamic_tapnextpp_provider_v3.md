# Dynamic TAPNext++ Set-Valued Association V3

## Status

This is a post-open mechanism study on the two already-open V2 source cases.
It is not a confirmation, state-of-the-art result, or authorization to inspect
the sealed V1 cohort. V2 remains immutable at its recorded commit and is still
the default behavior of every shared implementation entry point.

## Motivation

The V2 source smoke produced valid three- or four-camera endpoint
triangulations, but their prior reliabilities were approximately
`0.001`--`0.009`, below the unchanged `0.02` belief threshold. Inspection
identified one structural cause:

1. normalized entropy over a dense local pixel patch reduced the birth
   association probability;
2. the same entropy reduced multiview perception reliability again; and
3. the candidate-pixel distribution already contributed assignment-mixture
   covariance in metric units.

Normalized entropy over hundreds of nearby pixels is not a calibrated
probability that the material projection lies outside the patch. V3 therefore
represents the association as a latent set-membership event and represents
within-patch ambiguity once through covariance.

## Frozen V3 Semantics

V3 changes no threshold, query schedule, camera panel, tracker checkpoint,
physical trajectory, persistence trajectory, or future-data boundary.

The opt-in configuration is:

```text
birth association:
  set-valued-covariance-v1

assignment uncertainty:
  covariance-only-assignment-uncertainty-v1

assimilation:
  set-valued-association-mixture-v3
```

For one projected graph point, physical geometry defines a distribution over
nearby depth-supported object pixels. The nominal association probability is
the geometric evidence that the projection lies in that admissible patch.
Candidate spread is propagated through the camera geometry into the metric
observation covariance. It is not multiplied into perception reliability a
second time.

Perception reliability remains residual-independent and uses tracker
visibility, mask support, depth consistency, reprojection consistency, and
independent-camera redundancy. Birth and update association probabilities
remain separate and enter as the nominal assignment event in the robust
inlier/outlier mixture. The physical innovation is formed once and processed
once inside that mixture and the covariance-aware RBF update.

Unknown camera correlation is still handled by independent-pose clustering
and covariance intersection. One coherent camera-bias factor remains outside
the local covariance. Rejected updates preserve the selected physical or
persistence backbone bit exactly.

## Information Boundary

The prediction may read:

- the sealed physical and persistence source backbones;
- RGB, depth, object masks, and calibration only through each causal update;
- target-free physical motion and visibility used by the V2 adaptive query
  schedule.

It may not read:

- future RGB, depth, geometry, tactile data, or manual identities;
- the V1 target cohort;
- any hidden source trajectory until all V3 source predictions are sealed.

The two source outcomes were already open before this mechanism was designed.
Their V3 results are therefore diagnostic only.

## Advancement Rule

The two-case smoke can only justify a broader source-development panel, not a
fresh confirmation. Advancement requires:

1. at least one nonzero guarded Bayesian update;
2. no technical failure and exact fallback wherever support is insufficient;
3. no regression versus the selected raw backbone on either available hidden
   source metric;
4. a joint improvement over both the selected backbone and persistence on the
   scorable case; and
5. no use of a lower reliability, view-count, correspondence, or regret gate.

If V3 merely raises numerical confidence without improving hidden identities,
the projected-graph-query interface is closed. The next provider must be
observation-first: select trackable image material points, triangulate them,
then infer a soft trajectory-to-graph association.

## Claim Boundary

Passing the diagnostic rule would show only that removing assignment-
uncertainty double counting unlocks a useful source update. A SOTA claim still
requires a broader source lock, a fresh-object preregistration, evaluator
parity, and sealed point-estimate and calibration outcomes.

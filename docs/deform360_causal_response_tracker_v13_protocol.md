# Causal-Response Tracker V13

## Question

V13 admitted sixteen action-supported frame-zero identities in six of eight
already-open source cases. This next stage asks one narrower question:

> Can causal TAPNext++ observations of those identities remain accurate and
> sufficiently supported through frame 57, while agreeing across two disjoint
> camera panels?

This is a provider-competence experiment. It does not construct a state update,
a readout correction, or a future prediction.

## Frozen Provider

The eight V13-selected cameras retain their original deterministic split:

- four proposal cameras produce the candidate metric observation;
- four validation cameras can only corroborate or reject it.

TAPNext++ uses the pinned public checkpoint and consumes frames `[0, 58)`.
Every query is the exact material identity and frame-zero pixel association
sealed by V13. Candidate geometry affects association probability, while prior
reliability uses only tracker visibility, masks, depth consistency, reprojection
consistency, view redundancy, association uncertainty, and cross-panel
agreement. No residual against PhysTwin enters prior reliability.

The strict V13 arm requires at least three supporting views inside each panel.
The fallback arm permits two views, retains the registered fourfold covariance
inflation, and keeps a separate 5 mm coherent shared-camera-bias nuisance.
The proposal-panel covariance is further enlarged by the outer product of the
proposal-validation displacement. Cross-panel fusion therefore cannot become
more confident than the proposal estimate.

An identity is accepted only when both panels accept it and their metric
positions differ by at most 15 mm. The agreement weight has a frozen 5 mm
scale. Unsupported or inconsistent rows have zero update support and preserve
the unchanged baseline exactly.

## Evidence Order

1. Validate all eight immutable V13 query artifacts.
2. Seal six tracker/provider predictions and two exact query abstentions.
3. Write and validate the prediction-completeness barrier.
4. Only then deserialize the released manual identities through frame 57.
5. Score provider competence against those prefix identities.
6. Construct no state or readout update under this protocol.

The released prefix identities are outcomes for the competence gate, never
tracker inputs. No object observation or metric after frame 57 is permitted.

## Advancement Gate

All registered criteria must pass:

- exactly six provider predictions and two exact abstentions;
- at least 50% pooled endpoint support;
- at least 50% endpoint support in five of six provider cases;
- at least five scored cases;
- at most 15 mm object-balanced endpoint and late-prefix RMSE;
- at least 10% object-balanced improvement over exact persistence;
- provider wins in at least four scored cases;
- at most 10 mm mean accepted proposal-validation disagreement.

A pass authorizes a separately frozen prefix-event and baseline-relative,
bias-aware update study. It is not evidence that Bayesian-PhysTwin improves.
A failure stops this TAPNext++ carrier without tuning cameras, queries,
support thresholds, covariance, shared-bias scale, or gates.

## Boundaries

The eight cases are post-open source development evidence. The V1 sealed target,
all held-v8 artifacts and processes, and every future prediction metric remain
untouched. Fresh-object selection remains prohibited until an independently
produced held-v8 all-attempt hash-only exclusion manifest exists.


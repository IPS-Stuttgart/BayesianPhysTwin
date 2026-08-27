# Causal4D paired-action physical transport registration v1

## Why this is the next paper-level experiment

The existing BayesianPhysTwin evidence shows that guarded belief revision can
improve prediction and that uncertainty can have decision value. The existing
Causal4D evidence shows that latent physical interventions can be abducted in
controlled settings. Neither result establishes that a physical correction
inferred from one real action remains useful under a different real action.

The stronger falsifiable claim is **held-out physical transport**:

> infer a guarded physical state/parameter correction from the factual prefix of
> one action, propagate it through the simulator under another action, and beat
> unchanged physics, deterministic residual persistence, discrepancy-only
> persistence, and deliberately broken transport controls on sealed target
> outcomes.

This is materially stronger than adding another observation backend or another
same-action benchmark. It tests whether the inferred correction behaves like a
transportable physical belief rather than a local residual.

## The important acquisition-design constraint

The frozen Causal4D sloth protocol contains four command profiles:

- `lift_low`;
- `lift_high`;
- `lower_high`; and
- `lateral_low`.

It does **not** execute all four profiles in every grasp session. Each of the 18
independent grasp sessions contains exactly one unordered action pair. The six
edges of the complete graph on four actions are each repeated once at each of
three contact regions:

- anatomical left forepaw;
- anatomical right forepaw; and
- upper torso.

Thus the design is a balanced incomplete block design over actions:

- 18 independent grasp sessions;
- 36 physical executions;
- two actions per session;
- all six unordered action pairs;
- all 12 directed off-diagonal transfers globally; and
- three independent session realizations per directed transfer, one per contact.

The original `CrossActionProtocolV1` assumes one identical global action roster
in every session. Applying it directly would incorrectly require a `4 x 4`
matrix in each session, which the frozen physical design neither contains nor
should be changed to contain. Version 2 therefore binds one immutable action
subset per independent session.

## New claim-bearing artifacts

### `SessionActionSetV2`

Binds one grasp-session identity to the exact actions observed in that session
and derives its complete ordered action-pair matrix. For the Causal4D design,
each session has four ordered pairs: two diagonal diagnostics and two
source-to-held-out-target transfers.

### `CrossActionProtocolV2`

Retains the v1 target-blind and exact-fallback semantics while binding:

- the exact session-specific action sets;
- the Causal4D acquisition-design certificate;
- development, calibration, and target rosters;
- query and query Jacobian;
- identifiability and nonlinear-closure certificates;
- proper score, grouping, bootstrap, margins, and harm limit;
- target-access and technical-failure policies; and
- one exact model stack and numerical environment.

The six registered transport arms are:

1. `physical_fallback`;
2. `last_residual`;
3. `discrepancy_only`;
4. `state_only`;
5. `state_parameter`; and
6. `guarded_physical`.

Only complete predictions sealed before target access are admissible. Rejected
physical candidates select the exact physical fallback.

### `CrossActionPlaceboProtocolV2`

Uses the same session-specific off-diagonal pairs and registers four controls
that deliberately break physical transport while preserving nuisance structure:

- wrong source action;
- wrong session/object donor;
- phase shift; and
- identity permutation.

The physical and placebo predictions for one pair must bind the same parent
transport prediction, target outcome, target opening, scorer, source revision,
and selection/fallback disposition.

### `Causal4DCrossActionDesignV1`

Recomputes and verifies Causal4D's canonical protocol digest and fails closed
unless the acquisition remains exactly balanced:

- 18 unique grasp sessions and 36 unique executions;
- two distinct actions and pair orders 0/1 per session;
- one contact region per session;
- every action pair exactly once at every contact region;
- all six pairs at every contact; and
- `grasp_session` as the independent analysis unit.

### `Causal4DCrossActionRegistrationV1`

Binds the validated design to exact Causal4D and BayesianPhysTwin revisions,
method-freeze and attestation identities, readiness and primary-analysis
artifacts, target-access policy, BayesianPhysTwin distribution, explicit Prob4D
used/unused declaration, and prediction-batch policy. It must be frozen before
physical execution 1 and cannot be constructed from target outcomes.

### `Causal4DJointTransportResultV1`

Returns a positive result only if both conditions hold:

1. the guarded physical arm beats the physical fallback, discrepancy-only arm,
   and `last_residual` by their preregistered session-bootstrap margins while
   satisfying the harmful-session bound; and
2. the same guarded physical predictions beat every registered broken-mechanism
   placebo by the preregistered contrast margin.

The joint artifact also requires one target opening, one scorer, one target
accounting artifact, identical exclusions, one BayesianPhysTwin revision, and
exact parent-prediction linkage between the transport and placebo tables.

## Statistical unit and table size

For each of 18 sessions, the transport table contains:

```text
2 actions x 2 actions x 6 arms = 24 rows
```

Hence the complete transport table has 432 rows. Only the two off-diagonal pairs
per session enter the primary transport gains; diagonal entries are retained as
registered diagnostics.

The placebo table contains:

```text
2 directed off-diagonal pairs x (1 physical + 4 placebo arms) = 10 rows/session
```

Hence the complete placebo table has 180 rows. The combined table has 612 scored
rows, but **none of those rows is an independent replicate**. All pair-level
scores are averaged within each grasp session before uncertainty estimation. The
bootstrap sample size is therefore at most 18, after preregistered exclusions.

## Paper claim enabled by a positive result

A positive result would support the following bounded central claim:

> A target-blind guarded Bayesian physical twin can infer a correction from one
> real deformable-object action that improves prediction under a held-out action,
> beyond deterministic residual persistence and discrepancy-only correction, and
> the benefit disappears when the proposed physical transport mechanism is
> deliberately broken.

That claim connects BayesianPhysTwin's uncertainty-aware selection to Causal4D's
interventional semantics in one real, preregistered experiment. It is stronger
and more distinctive than a further within-action prediction improvement.

## What remains physical and cannot be inferred from software

The registration code does not fabricate physical prerequisites. Before the
confirmatory run, Causal4D still needs the stable inventory serial attached to
the exact sloth, completed physical object/contact registration, method-freeze
artifacts, readiness approval, and the 36 physical executions. Until those exist,
this work is a target-closed executable protocol, not empirical evidence.

## Scientific boundary

Even a positive joint result would not prove a unique data-generating cause,
unseen-object generalization, arbitrary-action generalization, calibrated raw
posterior covariance, real Prob4D provider competence, closed-loop safety, or
general deformable-object state of the art. Negative and insufficient-session
outcomes remain valid terminal outcomes and may not be retuned on the opened
cohort.

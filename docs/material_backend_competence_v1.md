# External material-backend competence protocol v1

## Purpose

A valid material-backend bundle establishes structural compatibility and
content-addressed custody. It does not establish that the external simulator is
a useful physical prior. This protocol defines the minimum source-only evidence
required before one registered backend may enter a fresh target experiment.

The protocol applies uniformly to JAX-FEM, Warp FEM, SOFA, Genesis MPM,
PositionBasedDynamics XPBD/PBD, PhysX deformables, MuJoCo Flex, and future
canonical families registered by `material_backend_v1`.

## Machine-readable qualification boundary

`material_backend_qualification_v1.MaterialBackendQualificationV1` closes the
executable gap between a portable backend bundle and this broader competence
protocol. One content-addressed record binds the canonical family, exact
producer transport, runtime, frozen source groups, incumbent, protocol, source
evidence, thresholds, numerical checks, information order, and exact fallback.

The qualification record covers structural and numerical admission only:

- units, coordinate frame, persistent entity order, and query identity;
- deterministic replay and zero-action equilibrium drift;
- rigid-transform equivariance and time-step refinement;
- topology identity and registered physical-sanity checks;
- finite-difference Jacobian agreement when gradients are claimed;
- source-query parity to the incumbent;
- protocol freeze before source outcomes;
- absence of target-outcome use; and
- byte-identical fallback.

A passing qualification is a prerequisite for the competence endpoints below.
It is not itself evidence that the backend improves a physical prediction. A
failed record remains a complete result for that exact runtime and preserves
every failure reason.

## Contiguous evidence promotion

The executable promotion chain is defined by
[`material_backend_evidence_v1`](material_backend_evidence_v1.md). It separates
seven stages from transport registration (T0) through downstream-query benefit
(T6). A source or target decision cannot be attached unless every predecessor is
present, the qualification matches the exact runtime, and source/target object or
session rosters are disjoint.

The canonical backend registry's `preferred`, `supported`, and `experimental`
labels describe integration maturity only. They must not be used as substitutes
for T3 numerical qualification, T4 source competence, T5 fresh-object
validation, or T6 downstream benefit.

## Frozen comparison roster

Every competence run must compare complete physical object/session groups under
one fixed roster:

1. the incumbent PhysTwin physical prediction;
2. persistence or zero-action replay;
3. the candidate external backend;
4. the candidate with the common BayesianPhysTwin guard and exact fallback.

A transport-specific legacy identifier and its canonical family are one
candidate, not two independent methods.

## Required source-only endpoints

The source gate must report all of the following at the complete object/session
level:

- future trajectory and official Chamfer/track error relative to the incumbent;
- zero-action drift and repeated-run numerical floor;
- sensitivity to each admitted material parameter over a frozen finite grid;
- nondegenerate ensemble or parameter-posterior spread;
- Gaussian NLL and an energy-score diagnostic with a common observation floor;
- nominal coverage together with full interval width;
- runtime, peak memory, and technical-failure frequency;
- BayesianPhysTwin acceptance and exact-fallback counts; and
- harmful accepted-update frequency and worst-group regret.

Frames, points, particles, mesh nodes, views, and taxels are not independent
calibration units.

## Information order

Before source outcomes are opened, seal:

- canonical backend family and exact producer-profile transport;
- engine repository, revision, runtime, device, source files, and assets;
- object/session roster and all technical-exclusion rules;
- material parameterization, finite search grid, and prior;
- observation floor, proper score, bootstrap seed/count, and grouping;
- incumbent, persistence, guard, and exact-fallback identities; and
- the target roster, which remains closed until a source-positive decision.

A valid support-negative or technical-failure result stops that provider
version. Deleting difficult objects, cameras, nodes, particles, or executions
after source outcomes are known is not permitted.

## Advancement decision

A backend advances only when all registered conditions pass:

1. mean incumbent-relative source regret has a nonpositive paired upper bound;
2. worst-group regression remains within the frozen practical margin;
3. no harmful accepted update exceeds the registered harm threshold;
4. the numerical floor is materially below the claimed physical improvement;
5. parameter sensitivity and uncertainty spread are nondegenerate;
6. proper score does not regress after accounting for interval width;
7. technical failures remain within the predeclared budget; and
8. every rejection reproduces the incumbent physical fallback exactly.

Runtime may break an otherwise exact score tie, but machine scheduling must not
change scientific selection.

## Target boundary

Passing this source gate selects at most one backend for one separately frozen
fresh-object or fresh-session protocol. It does not establish:

- independent-object transfer;
- calibrated deployment covariance;
- parameter identification;
- Causal4D intervention benefit;
- deployment safety; or
- state of the art.

A negative source result is complete and must not be retuned on the same source
cohort. A positive source result authorizes exactly the target access and
analysis written into the frozen protocol.

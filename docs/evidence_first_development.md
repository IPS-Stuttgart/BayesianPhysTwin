# Evidence-first development policy

## Purpose

BayesianPhysTwin already contains the principal estimator, artifact, provider,
provenance, and exact-fallback machinery needed for its registered scientific
studies. The default development priority is therefore to execute decisive
evidence gates, not to add parallel architectures whose only evidence is that
they compile or pass synthetic tests.

This policy applies to new methods, backends, public modules, artifact schemas,
provider versions, workflow files, and claim-facing diagnostics. It does not
change a frozen protocol, estimator, artifact, target-access boundary, or
scientific result.

## Admission bases

A proposed change should satisfy at least one of the following:

1. **Registered evidence gate** — it executes or directly removes a concrete
   blocker from a named, already registered experiment without changing that
   experiment's frozen method, split, endpoint, or information order.
2. **Reproduced defect** — it fixes a demonstrated numerical, implementation,
   security, packaging, or cross-repository interoperability failure and adds a
   regression test for the failure.
3. **Consolidation** — it removes duplicated code, workflows, schemas, public
   surface, or maintenance paths while preserving frozen evidence and exact
   compatibility where claimed.
4. **Required maintenance** — it is necessary security, release, dependency, or
   compatibility maintenance and does not expand the scientific method family.

A design document, green unit test, interface compatibility result, controlled
synthetic mechanism result, or newly content-addressed artifact is not by itself
an admission basis for another scientific component.

## Current evidence priorities

As of 2026-08-24, the principal open empirical gates are:

- `IPS-Stuttgart/BayesianPhysTwin#461`: independent object/session validation of
  the one frozen covariance-only candidate against `last_residual` and the exact
  physical fallback;
- `IPS-Stuttgart/Prob4D#49`: support-feasible held-out real-provider competence
  followed by a separately gated BayesianPhysTwin physical-query evaluation;
- `IPS-Stuttgart/Causal4D#377`: resolve the independent-verifier governance
  blocker; and
- `IPS-Stuttgart/Causal4D#25`: complete the registered 18-session,
  36-execution physical experiment after readiness passes.

New generic provider, covariance, attribution, semantic, backend, or evidence
container work should remain out of scope unless one of those gates retains a
result that localizes a capability missing from the current implementation.

## Pull-request requirements

A pull request must identify its admission basis and the owning issue, protocol,
or reproduced defect. It must also state:

- why existing interfaces or implementations are insufficient;
- whether any source, calibration, target, or confirmation information was
  accessed;
- the exact fallback and compatibility behavior;
- which artifact, schema, command, dependency, or protocol identities change;
- exact-head validation evidence; and
- the permitted claim and explicit non-claims.

Changes that cannot identify an admission basis should normally be deferred,
kept exploratory outside the stable package, or discussed in an issue before
implementation.

## Stop rules

- Do not add another challenger after a frozen source or target gate has opened
  merely to rescue the result.
- Do not let a downstream positive result rescue failed provider support,
  provider competence, mean identity, calibration, or upstream physical-query
  value.
- Do not treat frames, points, tracks, views, or taxels as independent units
  when the registered unit is a physical object, session, or execution block.
- Do not replace a valid negative result with another architecture on the same
  opened cohort.
- Do not expand the stable API or workflow inventory when an existing grouped
  route, versioned facade, script, or reusable workflow can own the operation.

## Scientific boundary

This policy improves prioritization, reviewability, and maintenance discipline.
It does not establish estimator accuracy, calibrated uncertainty, unseen-object
transfer, provider competence, Causal4D intervention benefit, deployment safety,
or state of the art.

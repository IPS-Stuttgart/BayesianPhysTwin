# When is a simulator a valid twin?

## Registered question

For a physical context

\[
x=(o,a,h,q),
\]

where `o` is an object, `a` an action, `h` a forecast horizon, and `q` a
downstream query, determine before the physical continuation is observed
whether one exact simulator runtime is competent enough to replace a registered
fallback for that query.

This is not a global simulator-ranking study. A runtime can be numerically
valid and still be incompetent for a particular query. A low average error can
also coexist with unacceptable localized harm.

## Claim hierarchy

Evidence is intentionally hierarchical.

| Gate | Question | Failure interpretation |
| --- | --- | --- |
| A0 | Are custody, units, entity order, and interfaces exact? | No scientific comparison is possible. |
| A1 | Is replay uncertainty bounded and the runtime identity fixed? | Numerical output is not stable evidence. |
| A2 | Does the simulator realize the registered control and attachment? | The nominal action was not simulated. |
| A3 | Does it remain physically sane over the claimed horizon? | Short replay is not horizon validity. |
| B1 | Does it beat or non-regress against the fallback on source groups? | It is not source competent. |
| B2 | Can outcome-unopened diagnostics order query risk? | Abstention has no source support. |
| B3 | Does a frozen threshold have a finite-group accepted-harm bound? | Selective competence is uncertified. |
| C1 | Does the frozen policy replicate on a disjoint public cohort? | No fresh-domain claim is authorized. |

Every failed gate is terminal for the exact runtime, cohort, and claim. It is
reported as evidence rather than hidden by replacing cases or tuning a later
gate.

## Certificate

Let `S` be the candidate simulator, `B` the exact fallback, `L` the registered
query loss, and `delta >= 0` a practical harm margin. A harmful accepted query
is

\[
H=1[L(S;x) > L(B;x)+\delta].
\]

A risk score `r(x)` may use only outcome-unopened information. The registered
feature families are:

1. within-runtime replay or ensemble dispersion;
2. cross-runtime model-form disagreement;
3. control, attachment, topology, and numerical diagnostics;
4. horizon-normalized instability indicators;
5. causal-prefix residuals; and
6. source-domain distance.

Lower scores are safer and the frozen router accepts inclusively when
`r(x) <= tau`. Threshold selection groups and certification groups must be
disjoint physical units. The certificate reports the exact one-sided
Clopper-Pearson upper confidence bound on

\[
P(H=1 \mid r(x) \le \tau).
\]

It therefore bounds harm among accepted queries, not unconditional error and
not safety outside the registered domain. Rejected queries return the same
fallback object supplied by the caller, rather than a reconstructed copy.

The machine-readable contract is implemented in
`src/bayesian_phystwin/simulator_competence_v1.py`. It binds the backend family,
producer, runtime, source evidence, method, partitions, object/action domains,
query functional, horizon, loss, feature schema, risk model, threshold source,
threshold, and fallback policy.

## Primary hypotheses

- **H1, selective harm:** On independent certification groups, the frozen full
  certificate attains its registered accepted-harm upper bound with nonzero
  useful coverage.
- **H2, query conditioning:** Competence depends materially on object, action,
  horizon, or query; one global backend ranking is insufficient.
- **H3, hierarchy:** Passing numerical/control gates alone does not predict
  source or query competence.
- **H4, model-form signal:** Cross-runtime disagreement adds risk-ordering value
  beyond within-runtime spread and causal-prefix residual alone.
- **H5, exact fallback:** Selective routing reduces harmful candidate use without
  changing the registered fallback on rejected groups.

Failure of H1 is a negative result and ends any deployment-style claim. H2-H5
remain descriptive only when H1 fails.

## Baselines and ablations

The primary comparison set is frozen before certification outcomes:

1. always use the registered fallback;
2. always use the globally best source simulator;
3. within-runtime spread only;
4. causal-prefix residual only;
5. source-domain distance only;
6. cross-runtime disagreement only;
7. a flat score model with the same outcome-unopened features; and
8. the full hierarchical certificate.

All selective baselines use the same threshold-selection groups, loss, harm
margin, minimum support, and fallback. Target thresholds are never swept.

## Metrics

Primary metrics are participant- or object-balanced:

- accepted-query coverage;
- harmful accepted count and exact one-sided upper bound;
- selected-policy loss relative to exact fallback;
- accepted-query regret relative to the best available simulator in hindsight;
- exact-fallback identity rate on rejected queries; and
- complete ordinary-success, technical-failure, and unsealable accounting.

Secondary source-only metrics are risk-coverage area, pairwise risk ordering,
Brier score for the registered harm event, and performance by action, horizon,
query, and object stratum. Frame-level intervals are forbidden because frames
from one object or participant are not independent physical groups.

## Preferred public confirmation study

### 4D-DRESS

The preferred common-domain study is 4D-DRESS. It provides real 4D garment
meshes for 32 participants, 64 outfits, more than 520 motion sequences, and an
existing simulation benchmark spanning LBS, PBNS, NCS, and HOOD. The physical
participant, not the frame or outfit, is the independent statistical unit.

Why this is stronger than another backend demonstration:

- several simulators face the same real continuation and query metric;
- object, motion, horizon, and garment category all vary;
- model-form disagreement is observable before scoring the continuation;
- enough independent participants exist for a finite-group statement; and
- failures can be localized to contexts instead of averaged into one ranking.

The exact participant split, sequence roster, upstream revisions, simulator
checkpoints, and query functionals must be hash-selected and locked before any
continuation scores are opened. A practical initial design reserves 8
participants for method and threshold selection and 24 for certification. With
zero harmful accepted groups, 24 accepted participants give a 95% one-sided
upper bound of 11.7%. Fewer accepted participants weaken the bound and may make
the source gate terminal.

Dataset access and simulator-output availability must be audited before this
split is registered. No dataset outcome is authorized by this document.

### Current feasibility boundary

The access audit on 2026-08-30 found no active or reserved 4D-DRESS lane, no
dataset payload, no license or access receipt, and no active process on the two
shared compute hosts. A detached code-only preprocessing checkout exists, but
it is not evidence of dataset authorization. The exact access-closed record is
`protocols/locks/fourddress_query_competence_feasibility_v1.json`.

The feasibility record pins the public 4D-DRESS code repository at
`d1685e18b438587f00227df41ec7659e67f04df1` and HOOD at
`9bc1076195979ac6c027fdd729c6e960cad62f2a`. Source and certification execution
remain unauthorized until a user-accepted dataset-license receipt, payload and
participant-metadata manifests, HOOD checkpoint, permitted SMPL assets, source
adapter qualification, and metadata-only participant split are independently
hash-bound. The registered software can already derive the 8/24 split from a
future names-only roster without reading outcomes. It cannot create or infer
that roster while access is closed.

### Public HOOD runtime qualification

The separate `run_hood_mesh_source_qualification_v1.py` runner uses only HOOD's
published arbitrary-mesh example and post-CVPR checkpoint. It binds the exact
code, input files, Python environment, and one-attempt ledger before two
independently reconstructed 30-step replays. It checks finite bounded
coordinates, valid identical topology, nontrivial cloth and obstacle motion,
and replay RMSE no greater than `1e-7 m`.

This is an interface and numerical qualification, not a physical comparison.
It does not establish correct attachment physics, force response, source
competence, or any accepted-harm bound. It consumes no 4D-DRESS, SMPL model,
physical outcome, or certification data, and it cannot authorize that later
study by itself. Failed execution is retained without replacement.

## Manipulation-domain stress test

The handed-off RGBench MatPhys protocol remains useful as a domain-shift stress
test. Its amended split has seven previously exposed source garments and only
two untouched non-manifold target garments. Those two targets cannot support a
population harm certificate: even zero harms yield a 95% upper bound of 77.6%.
It may therefore provide a bounded prospective case study, but never the main
safety or general-validity claim.

The source gate must first show deterministic causal replay, nondegenerate
MatPhys spread, leave-one-garment-out mean non-regression, risk ordering better
than prefix residual, exact fallback, and complete attempt accounting. A failed
source gate closes target access.

## Controlled mechanism study

The public DLO-Lab slingshot environment is retained only to isolate mechanisms.
Existing source evidence already shows:

- query headroom can exist when optimal actions vary across worlds;
- a probe can be informative about model identity yet have zero task value;
- posterior means can improve aggregate loss while harming several worlds; and
- exact-fallback guards can remove harm by abstaining completely.

New controlled worlds, if needed, must be generated from a frozen hash-selected
roster. Synthetic worlds demonstrate mechanism, not physical deployment risk.

## Stop rules

1. No backend reaches B1 without a passing exact T3 qualification and source
   competence decision.
2. No risk certificate is built from groups used for method or threshold
   selection.
3. No low-harm claim is made with insufficient accepted physical groups. At 95%
   confidence, zero harms require 9, 14, or 29 accepted groups to bound harm by
   30%, 20%, or 10%, respectively.
4. No target is opened when the source score fails to beat the registered simple
   comparator or when exact fallback fails.
5. Provider crashes, incomplete outputs, unsealable attempts, and invalid
   physical trajectories count against advancement and are never silently
   replaced.
6. No method, feature, threshold, margin, cohort, query, or horizon changes after
   certification outcomes are opened.
7. One registered held-out execution is frozen whether positive or negative.

## Claim language

The strongest admissible positive claim is:

> For the registered public cohort, simulator runtimes, query family, horizons,
> and fallback, the frozen pre-outcome certificate achieved the reported
> coverage while its one-sided accepted-harm bound remained below the registered
> target.

It is not admissible to say that a backend is generally trustworthy, that the
certificate guarantees safety, or that rejected queries establish simulator
failure outside the registered context.

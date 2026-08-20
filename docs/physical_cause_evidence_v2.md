# Physical-cause evidence certificates v2

## Purpose

`bayesian_phystwin.physical_cause_evidence_v2` is a **claim-facing**, additive
layer above `physical_cause_selection_v1`. The v1 selector remains the
operational complete-belief router. V2 does not change a selected mean,
covariance, parameter, nuisance term, provenance record, fallback object, or
target-access decision.

The stronger question addressed here is whether an operationally selected cause
has enough source evidence to support a bounded scientific attribution. The
contract distinguishes three requirements:

1. simultaneous baseline-relative proper-score evidence on independent physical
   source groups;
2. paired evidence separating the selected cause from every other source-eligible
   cause; and
3. for physical-state or physical-parameter attribution, separately bound
   nonlinear-closure and held-out transport evidence.

A missing, mismatched, or unresolved certificate fails closed for attribution.
It does not retroactively invalidate or change the v1 operational belief.

## Simultaneous selective-regret evidence

For a frozen proper score `S`, define the target-population regret of candidate
cause `c` relative to the exact physical baseline `B0` by

```text
R[c] = E[S(B[c], Y) - S(B0, Y)].
```

Lower is better. `PhysicalCauseRegretCertificateV2` binds a simultaneous upper
bound `U[c]` together with the exact:

- baseline and candidate belief identities;
- candidate-construction identity;
- registered physical query;
- proper score and grouping rule;
- source-evidence partition;
- complete candidate-universe identity;
- independent source object/session identities;
- common harmful-group margin and upper probability bound;
- registered stratum upper-regret bounds; and
- familywise confidence level and information-order declarations.

A registered policy may treat a candidate as source-eligible only when, for the
predeclared thresholds,

```text
U[c] < -minimum_improvement
harm_probability_upper <= maximum_harm_probability
all registered stratum bounds <= maximum_stratum_regret
source_group_count >= minimum_source_group_count.
```

Frames, coordinates, tracks, views, pixels, tactile taxels, and horizons are
nested observations. They do not increase the number of independent source
units.

The certificate requires the candidate universe and all thresholds to be frozen
before source scores, forbids target-outcome use, and requires independent
physical source groups. These declarations are part of the content identity.
Every candidate in one attribution decision must use the same baseline, domain,
query, source partition, proper score, grouping rule, candidate universe,
confidence level, independent groups, and harmful-group margin.

## Paired cause separation

Marginal baseline-relative bounds establish whether a candidate can improve on
the baseline; they do not establish which of two nonbaseline causes is better.
`PhysicalCausePairwiseCertificateV2` therefore binds a simultaneous interval for

```text
R[left] - R[right]
```

The pairwise artifact repeats the exact baseline, domain, query, source-evidence,
proper-score, grouping-rule, source-group, and candidate-universe identities.
This prevents an interval computed for another query, score, partition, or group
semantics from being substituted merely because the candidate IDs match. It also
requires the pairwise procedure and candidate universe to have been frozen
before source scores, requires independent physical groups, and forbids target
outcomes.

For a preregistered pairwise advantage `m`:

```text
upper(left - right) < -m  -> left dominates right
lower(left - right) >  m  -> right dominates left
otherwise                  -> attribution remains unresolved.
```

`PhysicalCauseAttributionDecisionV2` marks paired attribution as resolved only
when the operationally selected source-eligible cause is separated from **every**
other source-eligible cause. Missing pairwise evidence or an interval crossing
the registered equivalence region is an ambiguity result, not permission to use
a point estimate or arbitrary tolerance.

## Exact belief and decision binding

The attribution decision binds both the exact baseline belief and the exact
selected belief. A baseline decision must name that baseline identity. A
nonbaseline decision must name the candidate belief carried by the selected
cause's regret certificate. Substituting another candidate or an approximately
equal reconstructed belief fails validation.

The minimum-improvement, harm-probability, stratum-regret, source-group-count,
and pairwise-advantage thresholds are themselves part of the content-addressed
decision. The decision requires an explicit declaration that these thresholds
were frozen before source scores and rejects any target-outcome use. A valid
operational decision ID alone therefore cannot authorize claim-facing
attribution under newly chosen thresholds.

## Physical transport scope

Local subspace identifiability is necessary but not sufficient for a physical
interpretation. Claim-facing physical state and parameter attribution also bind
nonlinear replay closure and held-out transport evidence.

| Interpretation | Expected transport scope |
| --- | --- |
| observation bias | same camera, registration, or nuisance process |
| readout/model discrepancy | predictor/query/action-specific unless separately shown otherwise |
| realized intervention | one command execution; owned by Causal4D |
| physical state | same physical prefix/reset across held-out branch actions |
| physical/object parameter | same object across actions and independent resets |
| population prior | separately tested transfer to unseen objects |

A same-object transport result must not be promoted into unseen-object material
transfer. Likewise, predictive improvement under the action used for inference
is not itself held-out physical transport evidence. The referenced nonlinear and
transport artifacts must be validated under their owning typed contracts before
their content identities are supplied here; an arbitrary digest is not evidence.

## Operational selection versus scientific attribution

The v1 router answers:

> Which complete belief is operationally selected under the registered source
> policy?

The v2 evidence layer answers the narrower follow-up:

> Is the selected cause scientifically distinguishable under the registered
> source groups, and—if it is physical—does it carry the required nonlinear and
> held-out transport evidence?

A downstream consumer should therefore retain both identities. The operational
belief is the v1 output. The v2 artifact is evidence about the interpretation of
that output and must never reconstruct, edit, or substitute the belief.

## Statistical interpretation

Suppose the simultaneous source procedure satisfies

```text
P(all registered R[c] <= U[c]) >= 1 - alpha.
```

On that event, any candidate admitted by `U[c] < -minimum_improvement` satisfies
the declared expected proper-score improvement margin under the registered
source-to-target sampling assumptions. Exact v1 fallback makes the
implementation-level change relative to the caller-owned baseline exactly zero
whenever that router rejects an update.

This is a **selective expected-regret** statement. It is not a per-execution
safety theorem, does not make the baseline universally safe, and does not prove
a unique data-generating cause.

## Relationship to current experiments

The existing PhysTwin/PokeFlex admissibility ladder remains useful evidence that
support-dependent positive and negative cases exist. Because those rungs use
different datasets and operators, they should not be treated as a matched proof
that the cause selector distinguishes state from discrepancy.

The decisive next test is a preregistered matched transport experiment in which
baseline, discrepancy, state, parameter, and—where applicable—realized
intervention interpretations are evaluated on the same physical sessions with
held-out actions or resets. Existing frozen confirmation protocols must not be
modified to incorporate this v2 layer after their information boundary has been
sealed.

## Claim boundary

A valid v2 artifact supports only the registered finite-group selective-regret,
paired-separation, and transport statement. It does not establish provider
competence, calibrated raw posterior covariance, unseen-object generalization,
deployment safety, Causal4D intervention benefit, or state of the art. A valid
negative or unresolved result is complete evidence and must not be rescued by
retuning on the same opened groups.

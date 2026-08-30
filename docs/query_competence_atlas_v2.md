# Query-Conditional Simulator Competence Atlas v2

## Why a staged atlas

A single backend score cannot say why a simulator is or is not useful for a
decision query. A study may fail because the native rollout is invalid, because
the action bank has no meaningful choice, because a state-conditioned policy
does not transfer, or because a seemingly useful policy cannot bound harm on a
fresh cohort. These are different scientific outcomes and imply different next
work.

`bayesian_phystwin.query_competence_atlas_v2` records four ordered stages for
each exact simulator query:

1. `native_qualification`: the registered simulator world executes and passes
   physical/numerical checks;
2. `action_headroom`: the action bank has enough oracle value beyond its best
   fixed member to make state-conditioned selection meaningful;
3. `source_transfer`: a frozen guarded policy improves held-out source worlds;
4. `prospective_risk`: an independently frozen cohort satisfies value and
   finite-group harm gates.

The query identity still binds simulator, task, observation policy, action
bank, metric, world distribution, and statistical unit. Stages are not pooled
across query identities and a later stage cannot pass after an earlier
non-pass.

## Current public-simulator evidence

The atlas binds the existing DLO-Lab v1 competence registry and the new
off-grid coiling source result. It does not rewrite any frozen metric.

| Exact query | Native | Headroom | Source transfer | Prospective risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Wrapping v9 | pass | pass | pass | pass | **certified** |
| Slingshot v2 | pass | fail | fail | fail | **rejected** |
| Coiling off-grid v2 | pass | fail | fail | not evaluated | **rejected** |

Wrapping improves mean native decision reward by `0.004721` on 288 fresh
worlds, with paired 95% gain interval `[0.003894, 0.005597]`, one harmed world,
and a one-sided 95% harm upper bound of `0.016365`.

Slingshot reaches a fresh 288-world evaluation but gains only `0.000220`; its
paired interval crosses zero and 14 harmed worlds yield harm upper bound
`0.074952`. It fails at the query level without invalidating wrapping.

Coiling completes and qualifies all twelve off-grid source worlds, but its
oracle is only `0.001606` above the best fixed action. The guarded cross-fit
gain is `-0.000430`, with one fully harmful admitted world. It therefore stops
before any prospective cohort is selected. This exposes a different failure:
the action bank and transfer mapping are insufficient even though the native
simulator is valid.

## Exact fallback

`select_atlas_candidate` selects a candidate complete belief only for a query
whose exact atlas entry is fully certified and whose current inference is
admissible. Failed stages, incomplete stages, unknown queries, and failed
inference return the caller's original baseline object by identity. The atlas
therefore preserves the wrapping result while making unsupported scope
transfer executable as a rejection rather than a warning in prose.

## Contribution boundary

The staged atlas turns heterogeneous positive and negative evidence into a
falsifiable validation-domain contract. It answers both whether a simulator is
useful and where the validation chain breaks. That is stronger than claiming a
backend is generally competent or reporting only successful demos.

The evidence is public-simulator evidence. It is not a real-robot safety
certificate, an official DLO-Lab benchmark, a distribution-free guarantee, or
independent human review. Coiling remains source-only; no prospective coiling
claim is made. No result may transfer to a changed task, observation policy,
action bank, metric, or world distribution without a new entry.

## Reproduction

The builder rehashes every bound evidence file and reconstructs all derived
stage decisions:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_competence_atlas_v2.py \
  --output /tmp/dlolab-query-competence-atlas-v2.json
```

The committed atlas is
`results/source/dlolab_query_competence_atlas_v2/atlas.json`.

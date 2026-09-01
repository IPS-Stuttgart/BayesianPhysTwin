# Public-real query-conditional simulator competence

## Result

A simulator should not receive one global trust label. Its competence domain is
the intersection of the physical object profile, action family, prediction
horizon, physical query, and runtime identity. This retrospective audit applies
that rule to already-published public real-world results without changing any
prediction.

The central result is a query-rank reversal on 92 Deform360 objects. The same
Bayesian action-ensemble prediction is 5.22% better than persistence for the
registered active-field RMSE query, but 4.75% worse for all-field MAE and loses
on 74 of 92 objects for that alternate query. A query-independent simulator
ranking therefore cannot make the correct routing decision for both tasks.

| Public-real evidence | Context decision | Result with exact fallback |
| --- | --- | --- |
| Deform360, active-field RMSE | Accept 71/92 objects with matched action support | 71/0/0 accepted wins/ties/losses; 3.97% lower mean loss; harm upper bound 4.13% |
| Deform360, all-field MAE | Reject because the query is outside the certificate | Exact persistence; avoids the candidate's 4.75% regression |
| Tracking Cloth, shake to twist | Reject because the action domain changes | Exact persistence on all 8 specimens; avoids a 76.05% mean regression |
| PokeFlex, calibrated object profiles | Retain the registered same-profile route | 3.06% lower mean CD on 2 objects and 3 takes |
| PokeFlex, unseen object profiles | Reject because the object profile changes | Exact baseline on 4 objects; avoids a 1.16% mean regression and a 39.44% false-safe route |

For the Deform360 registered query, the exact context rule requires the target
action family to occur among the same object's source actions and requires the
source cross-validated candidate not to regress against persistence. It accepts
71 objects, falls back on 21, and retains 76.00% of the always-candidate mean
gain. The object-level paired bootstrap mean improvement is 0.02629 RMSE units,
with a 95% interval of [0.02099, 0.03210]. No accepted object is harmed; the
exact one-sided 95% Clopper-Pearson upper bound is 0.04132.

This is not merely a favorable subset report. The registered controls miss the
same 5% harm ceiling:

| Deform360 routing policy | Accepted | Harms | 95% harm upper bound | Mean change |
| --- | ---: | ---: | ---: | ---: |
| Always use candidate | 92 | 1 | 5.05% | -5.22% |
| Source non-regression only | 92 | 1 | 5.05% | -5.22% |
| Legacy source guard plus exact fallback | 81 | 1 | 5.72% | -4.85% |
| Exact object/action/query support plus fallback | 71 | 0 | 4.13% | -3.97% |

The sole active-query candidate loss occurs outside the exact action-support
domain. The action condition is therefore doing identifiable work beyond a
generic source-performance threshold.

## Query-rank reversal

Let `C` be a candidate simulator and `B` an unchanged fallback. If two physical
queries `q1` and `q2` satisfy

```text
L_q1(C) < L_q1(B), while L_q2(C) > L_q2(B),
```

then any deterministic query-independent router must incur avoidable regret on
at least one query. It either always selects `C`, which is wrong for `q2`, or
always selects `B`, which is wrong for `q1`. A query-conditioned certificate is
therefore necessary whenever model rankings reverse across physical
functionals. Deform360 exhibits exactly this reversal for active-field RMSE and
all-field MAE using the same prediction pair.

This proposition is deliberately modest. It does not claim that the exact
support rule is optimal. It establishes why uncertainty or competence must be
indexed by the downstream physical query rather than attached globally to a
simulator.

## Paper contribution

The resulting contribution is not a new globally superior simulator. It is a
backend-neutral decision layer with three parts:

1. Represent simulator competence over an explicit context tuple: object
   profile, action, horizon, query, and runtime.
2. Admit a model only inside demonstrated source support and otherwise retain a
   byte-identical fallback.
3. Audit decision value at the physical-object level, including abstentions,
   harmful-use probability, and query-rank reversals.

The public evidence supports the claim that context-specific competence has
real decision value. It also explains several earlier negative results without
discarding them: those failures are out-of-domain uses along different context
axes.

## Claim boundary

This is retrospective mechanism evidence composed from previously published
aggregate outcomes. The foundational object-by-action-by-horizon-by-query
contract at commit `bb282c8083bbc8598cb0382a07a14982eadea97e` predates the
Deform360 result, but this exact action-family specialization was not frozen as
a fresh prospective certificate before those outcomes. Therefore this audit is
not a universal safety guarantee, an official benchmark result, a state-of-the-
art claim, or prospective confirmation of a 5% harm bound.

No new measurements, raw media, held-v8 material, DLO4/DLO5 material, protected
targets, or new backend predictions were used. A future confirmation should
freeze this exact rule on untouched public objects before any outcome access.

## Reproduction

The protocol is
`protocols/public_real_query_competence_retrospective_v1.json`, with canonical
ID `d561ff6c3143c8f562cf64ba169436a303effd5fe117d9eb0afd0f42e7946992`.
The compact result is
`evidence/public_real_query_competence_retrospective_v1.json`, with artifact ID
`6f77ec5658d4e77e5eb514e584f090410c940c201a0eaaf7f2923ffcafbb12f6`.
Every input is SHA-256 bound in the protocol.

```bash
PYTHONPATH=src python scripts/science/run_public_real_query_competence_v1.py \
  --protocol protocols/public_real_query_competence_retrospective_v1.json \
  --deform360-result <deform360-result.json> \
  --tracking-protocol <tracking-protocol.json> \
  --tracking-metrics <tracking-metrics.json> \
  --tracking-specimen-scores <tracking-specimen-scores.csv> \
  --pokeflex-same-profile \
    results/sota/pokeflex_independent_depth_regret_guard_prospective_v1/prospective_evaluation.json \
  --pokeflex-independent-object \
    results/sota/pokeflex_independent_depth_regret_guard_calibration_v1/calibration_evaluation.json \
  --output evidence/public_real_query_competence_retrospective_v1.json
```

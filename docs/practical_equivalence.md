# Prospective practical-equivalence analysis

`bayesian_phystwin.practical_equivalence` evaluates whether a matched candidate
is practically indistinguishable from, noninferior to, or better than a
registered reference method on physical losses. It is intended for comparisons
such as the BayesianPhysTwin versus last-residual near tie, where reporting the
sign of a tiny observed difference as a win or loss would overstate the evidence.

The analysis consumes:

1. a `DecisiveEvidenceV1` bundle containing matched candidate and reference
   losses; and
2. a separately frozen margin policy that states the acceptable candidate-minus-
   reference loss difference for every metric and raw/deployed stream.

The report is diagnostic. It always records `claim_authorized=false` and
`promotion_authorized=false`; a scientific claim still requires a separately
registered cohort, method lock, target-access boundary, and claim decision.

## Difference and decision semantics

All differences use

```text
candidate loss - reference loss
```

and lower loss is better. For a frozen practical margin `m >= 0` and a paired
bootstrap confidence interval `[L, U]`, the report records all applicable gates:

| Gate | Condition |
| --- | --- |
| superiority | `U < 0` |
| noninferiority | `U <= m` |
| practical equivalence | `L >= -m` and `U <= m` |
| inferior beyond the margin | `L > m` |

A confidence interval can satisfy more than one gate. The compact decision label
prioritizes practical equivalence, then superiority, then noninferiority. The
individual Boolean gates remain authoritative for interpretation.

The interval is the existing two-sided equal-group paired bootstrap interval.
Using it for noninferiority is conservative relative to a one-sided interval.

## Statistical unit

Independent `group_id` values are resampled in pairs and receive equal weight.
Multiple registered horizons or query rows inside one group are averaged before
resampling. Use a complete physical object or independent acquisition session as
the group. Frames, points, tracks, taxels, and camera views are not independent
bootstrap units.

The policy specifies a minimum independent-group count in addition to the
bootstrap implementation's basic two-group requirement. A statistically
computable interval remains `diagnostic_only` when the registered minimum is not
met.

## Freezing a margin policy

Start from [the practical-equivalence policy template][policy-template].
Before target outcomes are opened, replace every placeholder and bind:

- the exact protocol and statistical unit;
- candidate and reference method identities;
- bootstrap replicates, seed, and confidence;
- each metric, raw or deployed stream, unit, and absolute margin;
- a concrete margin basis, such as independent annotation resolution,
  instrument accuracy, or a prespecified application tolerance; and
- the information-boundary declarations.

Do not choose a margin from the observed candidate-reference difference. A policy
with retrospective margins, outcome-informed margin selection, or dependent
claimed groups is parsed and scored, but none of its decisions is authorized.

Raw and deployed losses are separate targets. This is important for guarded
methods: a raw predictor can be worse than the reference while the deployed
system is equivalent because rejected units use exact fallback.

## Command line

First create a matched decisive-evidence bundle. The probabilistic scorer can do
this directly:

```bash
bpt diagnostic run score-probabilistic-predictions \
  predictions.json \
  score-report.json \
  --evidence-json decisive-evidence.json
```

Then evaluate the frozen margin policy:

```bash
bpt diagnostic run assess-practical-equivalence \
  decisive-evidence.json \
  practical-equivalence-policy.json \
  practical-equivalence-report.json
```

Both inputs are read with strict UTF-8 JSON semantics and byte budgets. The
report is written atomically and refuses replacement unless `--overwrite` is
specified. It binds content identities for the normalized policy and normalized
source evidence, plus byte identities for both supplied files.

## Report contents

For each registered target, the report contains:

- the equal-group observed difference and paired confidence interval;
- candidate-better, exact-tie, and candidate-worse group counts;
- best and worst group differences;
- superiority, noninferiority, equivalence, and clear-inferiority gates;
- whether the group-count and prospective-information requirements passed; and
- an authorized decision or `diagnostic_only`.

The overall decision is conservative across all targets. Any authorized clear
inferiority yields `failed_inferiority`; otherwise all targets must pass
noninferiority before the report can return an equivalence, superiority, or
mixed `noninferior_or_better` result.

## Scientific boundary

A favorable report supports only the frozen practical-loss comparison on the
registered groups. It does not establish fresh-object transfer, raw covariance
calibration, physical-state identification, provider competence, deployment
safety, Causal4D intervention benefit, or state of the art.

[policy-template]: ../protocols/templates/practical_equivalence_policy_v1.json

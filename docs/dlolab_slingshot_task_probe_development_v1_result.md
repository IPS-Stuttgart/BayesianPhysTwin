# Slingshot Task-valued Probe Development Result

## Decision

**The sole bounded development run ended in a result-serialization failure after
all native prefixes and the complete development bank were sealed. It is not a
prospective scientific result.**

Both 60% and 70% probes completed their two registered prefix-only native
batches. All four batches passed the fixed-endpoint, padding, command, runtime,
and 300-frame information-boundary checks. No task future was generated.

The runner then evaluated all four candidates in memory and sealed the complete
history/reward/prior bank. While publishing `result.json`, one NumPy boolean from
the metric checks reached the strict canonical JSON writer. Publication stopped
with `TypeError: Object of type bool is not JSON serializable`. The failure was
retained as artifact
`97724cd07621e6a6f63c8aa2da78f69a5308f370d64114544b0a75dff1cd35d7`.
The exact output root will not be retried.

## Post-result Development Audit

A read-only arithmetic pass over the sealed development bank gives:

| Prefix | Expected Bayes reward | Gain over blind | Mutual information (nats) |
|---|---:|---:|---:|
| Original passive | 7.008364 | 0.000043 | 0.037278 |
| Existing 50% | 7.010706 | 0.002385 | 0.464528 |
| New 60% | 7.011851 | 0.003530 | 0.589273 |
| New 70% | **7.015055** | **0.006734** | **0.867531** |

The blind reward is 7.008321 and the finite-slice oracle is 7.019442, leaving
0.011121 reward headroom. The 70% probe captures 60.55% of that headroom and
passes all four development thresholds in the post-result arithmetic. Both the
task-valued and generic mutual-information selectors choose the same 70% probe.
This is therefore a lead for active material identification, not evidence that
task-aware selection beats generic information gain.

The audit is not a repaired prospective result: it uses only the already-sealed
bank, is explicitly post-result, and authorizes no automatic future run. It can
inform a genuinely new protocol on disjoint continuous source worlds, whose
particle probe bank, decisions, and truth futures must be staged prospectively.

## Repair Boundary

After terminalization, the metric serializer was changed to convert accumulated
NumPy scalars and checks to built-in finite floats and booleans. A regression test
now requires strict JSON serialization. This is future infrastructure only and
does not write `result.json` into the closed root or reclassify the attempt.

## Claim Boundary

The study used already-open public-simulator source artifacts and four new
prefix-only CPU runs. It used no task future, fresh target, protected data,
held-v8 artifact, DLO4/DLO5 data, official DLO3 evaluation, recording, or GPU.
It does not change any DEFORM result, establish control improvement on continuous
worlds, or support a benchmark/SOTA claim.

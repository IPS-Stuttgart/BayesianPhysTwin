# Independent-action Slingshot policy certificate v3 result

## Decision

**Retained calibration-world native-QA failure.** All 16 causal calibration
prefixes and all 1,024 one-world/one-action future processes sealed ordinarily.
During complete-world reassembly, registered world 99 exceeded the unchanged
0.5 mm duplicate-position envelope. The runner stopped before sealing either
calibrator. No evaluation prefix, decision, future, or policy-value score was
created.

There is no retry, replacement, successful-subset score, or v2 recovery.

## What failed

The two independently executed copies of action slots 5 and 7 had an eventual
maximum rigid-object position difference of `0.864284 mm`. Their causal prefix
difference was only `4.997e-14 m`, their gripper trajectories differed by at
most `1.933e-13 m`, and their rod trajectories differed by at most
`2.296e-9 m`. Sphere and cube trajectories began measurable divergence after
frame 647 and crossed 0.5 mm at frames 816 and 791, respectively.

The duplicate reward difference was `0.000703335`, below the independently
registered `0.001` reward tolerance. Exact world realization, all eight fresh
processes, common prefix, duplicate reward, fixed endpoints, and sealed prefix
replay all passed. This is best described as a rare late rigid-contact
bifurcation, not a malformed rollout or an action mismatch.

## Denominator audit

A read-only post-terminal audit rederived all 128 already generated calibration
worlds. Exactly 127 passed the complete QA and only world 99 failed. Median,
90th, 95th, and 99th percentile duplicate-position differences were
`3.64e-10`, `1.12e-8`, `1.62e-7`, and `1.51e-5 m`; the single 0.864 mm event is
a pronounced heavy-tail outlier. No v3 gate was changed and these outcomes
were not used to score the method.

## Interpretation

The fresh-process interface removed the broad shared-slot failure surface, but
it exposed a rarer issue: seeded float64 execution is not uniformly
trajectory-deterministic through long-horizon rigid contact. A policy-value
certificate should therefore distinguish outcome-relevant execution
repeatability from full-state trajectory identity and should represent the
remaining simulator-process variability in its estimand or uncertainty.

That observation motivates a new, disjoint protocol; it does not authorize
loosening this one. V3's roster is closed.

## Claim boundary

There is no Slingshot policy-value, coverage, harm-probability, oracle-headroom,
benchmark, SOTA, real-world, or safety result from v3. The positive evidence is
limited to 1,024/1,024 ordinary independent calibration action executions and
127/128 complete-world QA passes under the registered interface.

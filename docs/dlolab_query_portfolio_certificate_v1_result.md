# DLO-Lab simultaneous query-portfolio certificate v1

## Decision

The complete six-query competence atlas supports a **simultaneous,
stage-aware deployment certificate**. Three registered queries reached final
prospective risk evaluation. Charging all three against one familywise error
budget leaves both certified query policies below the registered `0.05` harm
risk budget:

| Deployed exact query | Fresh worlds | Mean gain | Familywise gain lower | Harmed | Familywise harm upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wrapping v9 | 288 | +0.004721 | +0.003830 | 1 | 0.020813 |
| Slingshot reward-aligned v4 | 288 | +0.003457 | +0.001359 | 6 | 0.047069 |

The per-query confidence level is

\[
1-\frac{0.05}{3}=0.983\overline{3}.
\]

The harm bounds are exact one-sided Clopper--Pearson bounds. The gain lower
bounds use the component studies' unchanged percentile-bootstrap procedures,
seeds, and 20,000 replicates at the Bonferroni-adjusted lower-tail quantile.
Both familywise statements pass at 95% confidence. They are separate
statements; the union bound supplies a 90% lower bound for value and harm
holding jointly.

## Complete atlas accounting

No query was omitted after seeing its result:

| Exact query | Furthest stage | Portfolio action |
| --- | --- | --- |
| Wrapping v9 | prospective risk passed | certified guard |
| Slingshot reward-aligned v4 | prospective risk passed | certified guard |
| Slingshot v2 | prospective risk failed | exact fallback |
| Coiling off-grid v2 | source transfer failed | exact fallback |
| Separation development v2 | native qualification failed | exact fallback |
| Unknotting development v1 | native qualification failed | exact fallback |

The final-risk family size is three, not two. The rejected Slingshot v2 trial
is retained in the multiplicity denominator even though its deployed policy is
the exact fallback. Queries rejected before prospective risk evaluation never
expose a candidate policy and are also exact fallback.

## Portfolio guarantee

Let \(\mathcal F\) be the three queries that reached final risk evaluation and
let \(U_q\) be the adjusted harm upper bound for a deployed query. By the union
bound, with probability at least `0.95`, every deployed nonfallback query in
\(\mathcal F\) satisfies its reported bound simultaneously. All other atlas
queries use the exact baseline and therefore add no candidate-induced harm.
Consequently, for a selector that chooses a registered query scope before the
fresh world outcome is known, the deployed portfolio has harm probability at
most

\[
\max_q U_q = 0.047069 < 0.05.
\]

This statement permits an arbitrary mixture over the registered query scopes;
it does not permit choosing a policy after observing its future reward or
adding an unregistered query without recalculating the family.

## Decision value

The adjusted gain lower bounds remain positive for both deployed queries. The
method therefore establishes positive task-conditional decision value
simultaneously rather than relying on one favorable task. Cross-task reward
units are not pooled.

As a descriptive equal-query aggregate, the guarded policies harmed 7 of 576
evaluation worlds, compared with 84 of 576 for the corresponding unguarded
Bayesian policies. That is 77 fewer harmed worlds, or a `91.67%` reduction.
This aggregate is descriptive because Wrapping and Slingshot use different
native reward functions.

## Evidence and verification

- Component calibration worlds: 272 total (144 Wrapping, 128 Slingshot)
- Component evaluation worlds: 576 total
- Query atlas ID: `82aef94511f3e0db1746262d4d49ae3ff9e52a587c5c11ce41cc817faa7a7ab9`
- Portfolio certificate ID: `4af4f4aa127fdd9c53f103502e4593f87f6a3c6ad36d49a93e0406502b58da8a`
- Certificate file SHA-256: `6cf31106cb0d4c3377f0b574e8b0cd8c8dc078343ea1127554db7bc7f15635b6`
- Wrapping gain-vector ID: `fd564dc627c68be8e8df60b4ad4da8a3983345ba2571bb70b3e0d302fc50b701`
- Slingshot gain-vector ID: `adbedc553c6f2694a1beed5b1b538bb0a4129efbf0dc0bd81295bbabcce56469`

The builder reopens only the two already-frozen public-simulator result trees,
reconstructs their world-level gain vectors, reproduces the registered means,
95% intervals, and harm counts, and then applies the fixed three-query
multiplicity correction. It does not rerun either simulator or alter either
component result.

## Claim boundary

The Wrapping and Slingshot trials were prospectively registered and remain
immutable. This portfolio analysis is a **post-hoc simultaneous synthesis** of
those prospective components; the portfolio itself was not preregistered.
The result is public-simulator evidence for query-conditional decision value,
not a physical-robot safety guarantee, an official benchmark result, a point
prediction SOTA claim, or backend-wide competence. Its assumptions are
exchangeable worlds within each registered query distribution, complete atlas
accounting, exact fallback, and outcome-independent runtime query selection.

Rebuild the compact certificate with:

```bash
PYTHONPATH=src python scripts/build_dlolab_query_portfolio_certificate_v1.py \
  --output /tmp/dlolab-query-portfolio-certificate-v1.json
```

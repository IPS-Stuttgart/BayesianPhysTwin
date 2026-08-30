# DLO-Lab Slingshot posterior-aware policy-gain certificate source v2

## Question

Can a query-local, policy-level certificate retain useful Bayesian control value
while limiting harmful accepted updates more tightly than the frozen
simultaneous-action regret guard?

This is a prospective public-simulator source study. It uses no new recording,
protected target, held-v8, DLO4, or DLO5 data. It does not reclassify the closed
288-world Slingshot certified-guard result or the terminal v1 policy-certificate
attempt. The v1 evaluation futures remain absent and permanently unused.

## Frozen method

The candidate policy is the existing posterior-predictive mean action. A local
predictor estimates that selected action's gain over incumbent from seven
inverse-distance neighbors. The 147 fixed training worlds comprise the 51
already-open parent worlds and the failed v1 study's 96 calibration worlds.
Their outcomes are used only as source-training labels.

The feature is fixed before this run. Its first block contains registered rod
points relative to the sphere center plus temporal increments of every observed
point, cancelling the registered shared xyz bias. Its second block contains
residual-independent Bayesian diagnostics: joint- and iid-bias particle
weights, five incumbent-relative action-loss vectors, and three raw-regret
rows. Distances use feature means and scales fitted only on the canonical
reference set. An exact feature match averages only exact matching rows.

The calibration score for world `i` is

```text
s_i = predicted selected-policy gain_i - realized selected-policy gain_i.
```

One hundred twenty-eight new worlds form a disjoint calibration partition. At
miscoverage `alpha=0.10`, each arm's frozen offset is rank 117 of 128. Each
calibrator can set
only its one scalar; neither can change the feature, neighborhood size,
candidate policy, harm margin, or action bank.

The matched comparator is the existing simultaneous mean-action-regret guard,
recalibrated at rank 117 on exactly the same 128 worlds and all-action futures.
It uses one maximum-over-six-alternatives score per world. Thus any advantage
of the policy-level certificate cannot be attributed to a larger calibration
set or privileged future information.

On a new evaluation prefix, the lower gain bound is

```text
L = predicted selected-policy gain - conformal offset.
```

The candidate action is executed in simulation only when `L >= -0.002`.
Otherwise action 5, the exact incumbent, is retained.

## Information order

1. Hash-lock the implementation, parent reference tree, model bank, rosters,
   sensor seeds, runtime, and controls.
2. Generate and seal all 128 calibration prefixes and candidate predictions.
3. Generate all 128 calibration all-action futures.
4. Seal the single conformal offset.
5. Generate all 288 evaluation prefixes and seal all guarded decisions.
6. Require at least 24 accepted and 24 fallback worlds before any evaluation
   future is generated.
7. Generate all 288 evaluation all-action futures and score once.

Every future worker reproduces its candidate or complete guarded-decision
artifact before it creates a write-once execution claim. Reloading calibration
recomputes the conformal scalar and realized selected-policy gains from the
sealed metric bundle and checks the complete 128-task lineage. Evaluation
decisions are likewise recomputed from the sealed candidate and calibration
artifacts. This keeps the authorization path fail-closed without treating seal
metadata alone as evidence that the registered computation occurred.

After scoring, the separately invokable frozen verifier reloads the lock,
calibration, guarded decisions, and all 288 evaluation futures; recomputes every
reported arm, interval, risk bound, and gate; and requires exact equality with
the content-addressed result record.

No replacement or retry is authorized. Every evaluation world remains in the
denominator. A failed pre-future gate is a retained negative with no evaluation
future access.

## Source gate

The primary policy-gain guard must satisfy all of the following on the complete
288-world evaluation panel:

- mean gain over incumbent at least 0.002;
- paired 95% bootstrap lower endpoint above zero;
- mean gain at least 0.001 above the matched simultaneous-regret guard, with a
  paired 95% bootstrap lower endpoint above zero;
- marginal one-sided policy-gain coverage at least 0.85;
- simultaneous-action comparator coverage at least 0.85;
- exact one-sided 95% harmful-world probability upper bound at most 0.05;
- at least 10% of unguarded posterior-policy gain retained;
- at least ten unguarded posterior harms and at least five removed harms;
- at least 5% of oracle headroom captured.

The statistical unit is one fresh continuous simulator world with one frozen
sensor draw. The conformal statement is marginal under exchangeability. The
empirical Clopper-Pearson bound is also world-level. Neither is a conditional,
robot-safety, real-world, official benchmark, or SOTA guarantee.

# Nested selective expert router on public real cloth

This retrospective follow-up addresses the main weakness of the first
Tracking Cloth selective-digital-twin result: the frozen physics-versus-
persistence gate improved mean loss but selected only 10% of registered query
cases and did not beat `last_residual` on its accepted cells.

The v2 question is narrower and better matched to the evidence:

> Can observable task context route among exact persistence, the
> source-weighted Bayesian spring-bank mean, and the nominal spring forecast
> plus its last observed prefix residual, while holding an entire cloth
> material out of both fitting and model selection?

## Nested information order

For each outer held-out material:

1. only the other three materials enter tuning;
2. an inner leave-one-material-out loop predicts both non-fallback regrets;
3. ridge alpha and the admission threshold are selected only when inner
   coverage is at least 20%, practical harm among accepted cases is at most
   10%, all three inner-held-out materials have nonpositive mean regret, and
   every rejection is exact persistence;
4. the selected router is refit on all three outer-training materials; and
5. the untouched outer material is scored once.

Material identity is never a model feature. The registered features are the
joint motion/query/horizon category, commanded speed, grasp configuration, and
cloth size. The ridge implementation is NumPy-only and deterministic.

## Matched complementarity tests

The primary router admits both `bayesian_physics` and `last_residual`. Two
drop-one-expert ablations reuse the exact same outer-fold ridge alpha and
admission threshold:

- remove `last_residual`, retaining only Bayesian physics and persistence;
- remove Bayesian physics, retaining only `last_residual` and persistence.

This isolates expert complementarity rather than allowing each ablation to
change the router after seeing the outer material.

Independently nested physics-only and residual-only policies, the original v1
query/horizon gate, persistence, and an outcome oracle remain diagnostics.

## Evidence status

All Tracking Cloth outcomes were already open before v2 was designed. The
outer and inner splits prevent direct held-out-material fitting, but they do
not restore prospective freshness. The exact rerun is therefore classified as
retrospective model-development evidence.

It cannot establish fresh physical confirmation, unseen-material or
unseen-action transfer, causal intervention value, deployment safety,
calibrated joint uncertainty, universal simulator validity, or state of the
art.

# Deform360 trusted-physics residual source v1

This milestone evaluates a Bayesian residual on top of the already-frozen,
cross-fitted closure-and-simulator-self-diagnostic trust policy. It uses only
the 27 source-development episodes whose outcomes were already open.

## Result

The trusted physical prior is the positive result. Relative to persistence it
improves future track error by 6.62%, future Chamfer distance by 5.26%, late
track error by 9.49%, and late Chamfer distance by 8.59%. Its largest
episode-level future degradation is 3.33%.

The learned residual is the negative result. At 2,000 steps it is effectively
neutral against persistence and materially worse than trusted physics. Every
outer fold selects the exact residual fallback (`utility_threshold=1.1`), so
the deployed gated arm is identical to trusted physics.

## Boundary

These are leave-one-object-out source-development metrics on a deterministic
256-node farthest-point subset and a 76-frame horizon. They are not directly
comparable to the official Deform360 tables, do not establish calibration, and
do not unlock the sealed penguin or PokeFlex targets.

The result retires the current one-step residual architecture. The next
competitive experiment should scale automatic registration and the observable
trust policy across the public Deform360 corpus, using the official evaluator
and split when available. A stronger temporal residual is justified only on
top of that benchmark-aligned system.

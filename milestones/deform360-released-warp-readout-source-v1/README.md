# Deform360 Released-Particle Warp Readout Source V1

This milestone records the locked, development-only transfer test from the
previous official-Warp sparse source evaluator to the ordered particles
released by the Deform360 authors.

## Evidence Order

The protocol was locked and pushed before matched-origin dense scores existed:

- initial lock commit: `09f2bee`;
- causal contact-schedule clarification: `1e2a6e0`;
- prediction implementation commit: `8a1d310`;
- protocol SHA-256:
  `b7d06929079fd97ad8e2b6dbe149946851321bb6dc23a92c44cc9b2a0409e0c0`.

Prediction ran from detached commit `8a1d310` on `gpuserver4090` GPU 0 using
the official PhysTwin commit
`2b6630528141b9cba5a7677c8b88b2129b4a8390`. The predictor read only the
five origin particle files, the locked sparse source observations, and future
controller motion. It held the origin contact state fixed and did not read
future tactile transitions.

The prediction artifact and archive were copied off the server and validated
independently before the scoring process opened any released future particle
frame:

- prediction result SHA-256:
  `5b5934dbcf642fd4f6af7e0494760425269007ae69df389622953714d9319fd3`;
- prediction archive SHA-256:
  `7545cb7b7609ff181b61101d0f5f7baabb43b11a7691441b34b3eed542be3c01`.

## Result

The preregistered transfer gate failed.

| Arm | Dense Chamfer | Change vs persistence | Identity error | Change vs persistence | CD wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched persistence | 4.20 mm | - | 9.32 mm | - | - |
| LOO, finite velocity, rotated offset | 8.55 mm | +103.66% | 24.09 mm | +158.39% | 1/5 |
| LOO, finite velocity, fixed offset | 8.47 mm | +101.65% | 23.05 mm | +147.30% | 1/5 |
| LOO, zero velocity, rotated offset | 9.13 mm | +117.37% | 24.73 mm | +165.28% | 0/5 |
| Pooled, finite velocity, rotated offset | 6.62 mm | +57.69% | 17.85 mm | +91.46% | 1/5 |

The primary arm required at least 5% mean Chamfer improvement, at least three
of five episode wins, and no panel identity-error degradation. It instead
regressed mean Chamfer by 103.66%, won one episode, and increased identity
error by 158.39%.

Per-episode primary Chamfer was:

| Episode | Action | Persistence | Primary Warp readout |
| --- | --- | ---: | ---: |
| 0 | move edge | 1.37 mm | 2.96 mm |
| 3 | lift center | 3.92 mm | 8.46 mm |
| 4 | curl edge | 8.86 mm | 8.41 mm |
| 5 | lift both edges | 5.35 mm | 9.78 mm |
| 8 | curl both edges | 1.49 mm | 13.13 mm |

The backend remained numerically healthy: repeat-rollout RMSE was exactly zero,
all states were finite, and the maximum primary p99 relative edge strain was
12.03%, below the locked 50% ceiling.

## Interpretation

The old 21-node source-gate gain does not transfer to the author-released
ordered particles under a fair matched-origin forecast. This is an evaluator
and state-representation transfer failure, not a Warp determinism or stability
failure.

The pooled candidate is less poor than the leave-one-source candidates, which
is diagnostic evidence that sparse-evaluator candidate selection does not
align with dense ordered-particle accuracy. It still loses decisively to
persistence and was not the preregistered primary gate.

Changing offset transport or initial velocity does not rescue the route.
Accordingly, this milestone stops the sparse-Warp dense-readout path. It does
not justify a fresh-object preregistered evaluation.

## Post-Gate Shrinkage Diagnostic

After the primary gate failed, a non-authorizing diagnostic evaluated

```text
persistence + alpha * (primary Warp readout - persistence)
```

on the fixed grid `{0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1}`. Every
episode has some oracle Chamfer headroom, but its best scale ranges from 0.025
to 0.8. A pooled alpha of 0.1 improves mean Chamfer by 6.18%; leave-one-episode
selection instead degrades mean Chamfer by 1.31%, because the alpha selected
without episode 8 over-transmits that episode's response.

Thus the physical direction occasionally carries useful information, while its
magnitude is not safely transferable. This supports a separate
baseline-relative regret guard with exact fallback. It does not alter the
failed gate or authorize a fresh run of this readout method.

## Claim Boundary

All five `001-rope` source episodes were already open. This is exploratory
development evidence, not independent confirmation, dense PhysTwin
reconstruction, calibrated Bayesian prediction, or a direct Deform360 Table 4
comparison.

No forbidden rope episode, held-v8 artifact, or sealed PokeFlex target was
accessed.

## Verification

The released-readout, official-Warp helper, and released evaluator tests pass:
31 passed. Ruff, `py_compile`, and `git diff --check` also pass.

The repository-wide suite remains blocked during collection by the pre-existing
missing export `TemporalBayesianResidualModelConfig` in
`causal4d_public.deform360_bayesian_residual` (two collection errors). This is
the same unrelated blocker recorded before this milestone; the new tests
collect and pass independently.

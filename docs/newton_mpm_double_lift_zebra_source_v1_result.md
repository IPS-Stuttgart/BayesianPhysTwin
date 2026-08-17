# Newton MPM source gate v1 result

## Decision

**FAIL / retain the incumbent physical backend.**

The frozen Newton implicit-MPM bridge executed all eight candidates on the
already-open `double_lift_zebra` source case and produced deterministic,
finite physical rollouts. The selected arm beat exact persistence, but it
failed both non-regression gates against the incumbent PhysTwin rollout.
Consequently, the byte-exact incumbent fallback was selected and source frames
40--57 were not opened by the future scorer.

This is evidence about the frozen direct-particle, hard-attachment MPM adapter.
It is not a reproduction of PhysWorld, a fresh-data result, a calibration
claim, or evidence that MPM as a model family is inadequate.

## Frozen execution

- implementation commit: `a9c492f01a81201a2c81af47ff685c77b2742bb0`;
- Newton / Warp / NumPy / SciPy: `1.5.0 / 1.16.0 / 2.2.6 / 1.18.0`;
- device: NVIDIA RTX 6000 Ada Generation;
- candidates completed: `8/8`;
- selected candidate: `25 kPa`, damping `0.002`;
- deterministic replay RMSE: `0.0` m;
- maximum zero-action drift: `0.0` m;
- median final ensemble spread: `73.974` mm; and
- target or held-out artifacts read: `false`.

The selected arm was chosen on frames 1--29. The following table reports the
fit split and the separately frozen validation split, in millimetres.

| Split | Method | Identity coordinate RMSE | Symmetric Chamfer |
|---|---|---:|---:|
| Fit | Persistence | 7.937 | 6.344 |
| Fit | PhysTwin incumbent | 3.363 | 2.404 |
| Fit | MatPhys/Warp | 3.639 | 2.816 |
| Fit | Newton MPM | 4.805 | 3.615 |
| Validation | Persistence | 24.818 | 18.712 |
| Validation | PhysTwin incumbent | 7.803 | 3.754 |
| Validation | MatPhys/Warp | 7.341 | 3.542 |
| Validation | Newton MPM | 16.502 | 12.252 |

On validation, Newton improves identity RMSE by `33.51%` and Chamfer by
`34.52%` relative to persistence. It regresses by `111.49%` and `226.34%`,
respectively, relative to the incumbent. Its balanced persistence ratio is
`0.65985`, so the persistence-improvement gate passes; both incumbent
non-regression gates fail.

## Interpretation

The run establishes three useful facts:

1. The simulator-neutral physical-rollout contract can carry a genuinely
   different MPM backend, not only spring proposals replayed in Warp.
2. The known controller action creates a substantial, stable MPM response,
   and the soft end of the frozen material grid is preferred.
3. Treating the irregular, surface-heavy PhysTwin state directly as MPM
   particles with hard kinematic attachments does not preserve the incumbent's
   predictive quality. Increasing stiffness makes validation worse.

The likely next MPM-specific question is therefore geometric and interfacial,
not a larger modulus grid: a deterministic volumetric particleization with a
separate material-query readout and a less rigid contact model. That would be a
new source protocol and must not be tuned on this opened continuation. Until
such a bridge has independent source evidence, Newton remains an optional
backend implementation rather than a promoted Bayesian-PhysTwin baseline.

## Evidence

Compact content-addressed artifacts are in
`results/sota/diagnostics/newton_mpm_double_lift_zebra_source_v1/`:

- `source-custody.json`: SHA-256
  `7db6482b6caf713f37f10ae158fa0a1ae7073b2b75647bebfaf90838de52dc98`;
- `newton-grid.json`: SHA-256
  `11174df80aadd750e114ead6c4c19406143e1f5123d37d6c141421922b6fa44e`;
- `prefix-result.json`: SHA-256
  `5a4f0391e3e085df1ff0239e7eb8e24c784f1e8f4b4ecc429d31ea42e7806668`;
  and
- `future-result.json`: SHA-256
  `2d442a603b77dc6cdd59b27385d7ec362a1fbeb148f79e1fdd9e8341ec244bef`.

The future result has status `future-not-opened-validation-gate-failed` and
records `future_outcomes_read=false`.

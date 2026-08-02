# Causal tactile regret guard: opened-source result

## Status

This is **post-open source-development evidence**, not prospective confirmation
and not a state-of-the-art claim. It uses 17 already-open Deform360 objects:
five objects from the 27-case positive source panel and 12 objects from the
low-motion stress panel. Each outer fit excludes the evaluated physical object.

The locked source-development protocol is
`configs/sota/deform360_tactile_regret_guard_source_v1.json`, canonical SHA-256
`0591dfb5c88e3b578099b86503b7e8d1d9bf664258fb86b94bf354ded516986e`.
The result artifact is
`results/sota/diagnostics/deform360_tactile_regret_guard_source_v1/result.json`,
canonical SHA-256
`e3f8d8f4d37e306763776e4c50d838a269404c7500bef3bea4c0043c0fc21c26`.

## Question

The fixed 16-identity camera update improves the dynamic Open27 panel but
regresses badly when the selected physical/persistence backbone is already below
1 mm. Camera geometry and camera-internal uncertainty cannot reliably detect a
coherent shared bias. This diagnostic asks whether Deform360's independent
taxel stream can admit only updates supported by a stable physical contact.

## Method

The camera candidate is unchanged. Four raw `16 x 32` tactile grids are aligned
to each causal update at frames 19, 38, and 57. Processing is deliberately not
the released episode-normalized path: episode-wide peak normalization depends on
future tactile samples. Instead, the guard uses raw values after the released
sensor baseline is subtracted and the invalid final column is cleared.

Thirteen update-local features describe contact energy, recent taxel-pattern
stability, sensor balance, and active-taxel support. A ridge score is fitted with
equal loss mass per source object. The score is not interpreted as a probability.
The fixed admission threshold is 0.7. A rejected update is the selected raw
backbone bit exactly.

## Result

All source-development advancement gates passed.

| Panel | Arm | Hidden identity RMSE | Hidden Chamfer | Relative identity | Relative Chamfer |
|---|---:|---:|---:|---:|---:|
| Open27, 5 objects / 27 cases | selected backbone | 8.807 mm | 7.888 mm | -- | -- |
| Open27 | unguarded camera update | 7.441 mm | 6.795 mm | -15.51% | -13.86% |
| Open27 | tactile-guarded update | 8.450 mm | 7.657 mm | **-4.06%** | **-2.94%** |
| Stress, 12 objects / 12 cases | selected backbone | 0.899 mm | 0.772 mm | -- | -- |
| Stress | unguarded camera update | 2.709 mm | 2.477 mm | +201.34% | +220.84% |
| Stress | tactile-guarded update | 0.899 mm | 0.772 mm | **0.00%** | **0.00%** |

The outer object-cross-fitted guard admits 6 of 117 updates. All six are
beneficial under the maximum of hidden-identity and hidden-Chamfer regret. It
produces three case wins, 36 exact ties, and zero case regressions across the
combined 17-object panel. Object-balanced over all 17 objects, identity improves
by 3.26% and Chamfer by 2.38% relative to the unchanged selected backbone.

## Verification

The diagnostic runner reproduces `result.json` byte for byte. The seven focused
guard and evidence-lock tests pass, and Ruff reports no findings in the changed
Python files. On native POSIX, the complete candidate tree adds seven passing
tests and has the same 16 pre-existing failures as its exact parent tree
(`1350 passed, 25 skipped, 16 failed` versus `1343 passed, 25 skipped, 16
failed`). Those inherited failures concern older frozen artifact hashes and
cadence fixtures; this work does not rewrite them. Repository-wide Ruff likewise
retains its pre-existing backlog, while the scoped lint check is clean.

## Interpretation

This is a useful source result because the independent modality restores a
non-vacuous safe region: the earlier camera-only regret certificate abstained
everywhere, while looser camera-only selectors admitted coherent-bias failures.
The tradeoff is intentionally conservative. Most of the unguarded Open27 gain is
left on the table, and only three cases change.

The result does not prove that tactile contact makes camera updates safe on new
objects. The feature set and 0.7 threshold were developed after these source
panels were open. The only valid next claim-bearing step is to freeze the full
17-object model and evaluate it unchanged on genuinely fresh physical objects
excluded from held-v8, Prob4D, MolmoMotion, and this development cohort.

## Recommendation

Proceed to a small preregistered fresh-object evaluation. Require exact fallback,
zero target-tuned calibration, separate reporting of technical failures, and no
replacement of failed cases. Do not describe the present result as SOTA until
that independent evaluation passes.

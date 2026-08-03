# CoTracker3 per-view RGB-D state-update transfer result

Date: 2026-08-03

Status: source transfer gate failed; no fresh evaluation is authorized.

## Result

The bias-aware per-camera RGB-D state update transferred in direction but not
in magnitude. Lower is better.

| Cohort | Baseline CD | Candidate CD | Change | Baseline track | Candidate track | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development 3 | 11.968 mm | 11.468 mm | -4.18% | 19.834 mm | 18.606 mm | -6.19% |
| Transfer 19 | 11.297 mm | 11.201 mm | -0.85% | 21.532 mm | 21.378 mm | -0.71% |
| All 22 | 11.389 mm | 11.238 mm | -1.33% | 21.300 mm | 21.000 mm | -1.41% |

On the registered 19-case transfer subset, the selector produced only `3/19`
joint wins and a worst case-metric regression of `15.22%`. Exact fallback was
used in `10/19` cases. The underlying state update was available in `16/19`,
and six full updates hit the fixed 20 mm cap, so missing support alone does not
explain the weak transfer.

Every registered advancement condition fails:

| Gate | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| CD improvement | at least 3% | 0.85% | No |
| Track improvement | at least 3% | 0.71% | No |
| Joint case wins | at least 12/19 | 3/19 | No |
| Worst regression | at most 10% | 15.22% | No |

## Interpretation

Keeping camera observations separate, modeling shared bias, and restricting
the update to physically responsive modes fixes the catastrophic behavior of
naive camera-only fusion. It does not make prefix Chamfer a reliable selector
for future state correction. Several accepted candidates improve one metric
while degrading the other, and one full-scale update degrades both materially.

The fixed quarter-scale arm is safer post hoc: it changes transfer CD by
`-0.35%`, track error by `-1.22%`, and has a `3.14%` worst regression. It still
has only `7/19` joint wins and does not approach the locked transfer gates.
Those opened outcomes cannot be used to replace the registered selector.

This closes the current per-view depth state-update family. The next method
must add information or a genuinely baseline-relative regret guarantee; it
must not retune the cap, scales, basis rank, or prefix selector on these cases.

## Claim boundary

This is retrospective source evidence on an opened cohort. It is neither
independent confirmation nor a state-of-the-art result. Future RGB, depth,
tracker outputs, and manual trajectories did not form predictions. Manual
trajectories were used only after prediction for scoring. No held-v8 or sealed
target artifact was accessed.

## Provenance

- method commit: `95546c946ecb3935a768ff53cda111b8b0e4ec80`
- protocol commit: `8f3cd678365964425dded93065a55245f32db672`
- raw result SHA-256:
  `b382755dd8edc8cba1370e1a324a7133643ffdf1c89a05e01db78ceb4b3f98db`
- CoTracker3 cue manifest SHA-256:
  `899fadb41531bfe27d7743d8ba055e16fab3521259d1e3b7cbf945059ca82175`
- family-selection SHA-256:
  `5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890`

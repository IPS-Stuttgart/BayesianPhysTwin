# CoTracker3 sparse-prefix headroom on opened PhysTwin-22

## Evidence boundary

This is a post-open development study on the 22 released PhysTwin cases. It
localizes model and observation headroom; it is not a confirmation and does not
support a state-of-the-art claim. Future object observations never enter a
prediction. The manual arm is an online-supervised ceiling, and the frame-zero
query arm is an association oracle rather than a label-free method.

The evaluator is commit `b3aa832c1b826836d286801614fc80f7ce053c80`.
The primary protocol, crossed-control amendment, raw reports, and compact
machine-readable result are under
`results/sota/phystwin_cotracker3_open22_headroom_v1/`.

## Result

All values are equal-case future means in millimetres. The same causal temporal
selector is shown first so that post-open candidate choice cannot explain the
factorial pattern.

| Dense prefix source | Sparse prefix identity | CD | Track |
| --- | --- | ---: | ---: |
| Released pseudo-tracks | Manual trajectory | **7.892** | **13.429** |
| Released pseudo-tracks | None | 7.898 | 19.989 |
| CoTracker3 three-view | Manual trajectory | 10.588 | 13.429 |
| CoTracker3 three-view | None | 10.627 | 20.415 |

The selected physical baseline is `11.389` mm CD and `21.300` mm track error.
The manual ceiling improves it by 30.70% and 36.95%, with 21/22 joint wins, and
crosses the published `8/15` mm operating point. This is capacity evidence only:
manual prefix labels are not a deployable observation.

The strongest post-open automatic candidate is sparse graph support at
`10.651/19.581` mm, improving CD by 6.48% and track error by 8.07%, with 9/22
joint wins. Even a per-case metric oracle over the frozen candidate family is
only `10.122/18.714` mm. A fresh evaluation of this arm is therefore not
justified.

## Factorial diagnosis

The effects are nearly separated:

- Released dense pseudo-tracks supply geometry. Replacing them with strict
  three-view CoTracker3 costs about 2.7 mm CD.
- Manual sparse identities supply identity accuracy. Removing them costs about
  6.6-7.0 mm track error while changing CD by less than 0.04 mm under the same
  selector.

A final oracle fixed the nine evaluation queries from frame-zero geometry and
used CoTracker3 for every later anchor observation. It reached only
`8.916/17.575` mm; its per-case metric oracle remained at `7.779/16.198` mm.
Unknown query association is therefore not the principal gap.

On query-point frames surviving the strict three-view gate, CoTracker3
displacement RMSE is 4.35 mm, compared with 10.86 mm for the released dense
tracks. The problem is support: mean and median query support are only 32.7%
and 32.9%, and two cases have no supported query frame.

## Decision

Keep the released dense geometry channel. Do not tune the current CoTracker3
selector further or run it on fresh targets. The next candidate is a separate
sparse identity channel with:

1. conservative two-view fallback with metric covariance inflation;
2. redundant-view consistency when at least three views exist;
3. an explicit shared camera/time bias nuisance;
4. physical-action support for admission;
5. covariance intersection when cross-channel correlation is unknown;
6. exact baseline fallback under a source-calibrated regret guard.

Only an object-disjoint source gate should choose that method. A fresh-object
protocol becomes appropriate only after the source gate improves both CD and
track error and demonstrates calibrated uncertainty.

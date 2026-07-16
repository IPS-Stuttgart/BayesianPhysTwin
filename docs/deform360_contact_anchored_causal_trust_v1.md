# Deform360 contact-anchored causal trust v1

## Status and claim boundary

This is a post-hoc source discovery on `002-rope-silk`. All six source
continuations were available while the automatic material association,
physical candidate subset, contact-regime fallback, and final fixed trust rule
were developed. At the source freeze, calibration episodes 3, 4, and 8 and
target episode 1 were sealed. The calibration prefix was subsequently opened
under the frozen gate; the gate failed before future scoring, and target
episode 1 remains sealed. The result is neither a retroactive pass of the
earlier independent source gate nor a Deform360 state-of-the-art result.

## Method

The automatic observation-to-twin boundary uses only the first six staged
frames. Persistent Gaussian identities are filtered by multiview mask support
and opacity, associated to a contact-anchored filament coordinate, and lifted
to 21 material nodes. The production `Deform360MaterialAssociation` artifact
stores the association, metric observation variance in square metres,
normalized cue reliability, contact anchors, source hashes, and the causal
information boundary.

The predictor separates the official Warp rollout into an intervention
response and autonomous drift:

```text
x_hat = x0 + a * (x_driven - x_zero) + b * (x_zero - x0)
```

The source discovery supports the fixed rule:

- physical candidate 157: stretch 3000, bend 1000, controller 3000, ground friction 0.3;
- `a = 1 / controller_count`;
- `b = 0`;
- exact persistence for support-tangential drag actions;
- the physical response arm for prehensile lift, fold, and curve actions.

This has a useful mechanistic interpretation: the identified intervention
response transfers, while the simulator's autonomous settling does not.

## Exploratory source result

The fixed rule was selected after inspecting source outcomes. It improves both
metrics on all four prehensile source actions and is exactly identical to
persistence on both drag actions.

| Episode | Action | Policy | CD change | Track change |
|---:|---|---|---:|---:|
| 0 | lift side | physical response | -43.1% | -9.9% |
| 2 | drag side | exact fallback | 0.0% | 0.0% |
| 5 | lift sides | physical response | -1.8% | -0.7% |
| 6 | lift middle and side | physical response | -19.9% | -10.9% |
| 7 | drag sides | exact fallback | 0.0% | 0.0% |
| 9 | fold | physical response | -18.0% | -12.8% |

Across six execution-balanced episodes, mean Chamfer changes from 46.16 mm to
42.12 mm, an 8.75% improvement. Mean material-track error changes from 24.80 mm
to 23.11 mm, a 6.80% improvement. These are exploratory source metrics based
on reconstructed hulls and automatically associated Gaussian material tracks,
not the official Deform360 benchmark metric implementation.

The outer leave-one-prehensile-action-out selector was slightly less robust:
7.98% Chamfer and 6.42% track improvement, with a 0.82% track degradation on
episode 5. This is why the final discovered rule is the simpler fixed causal
decomposition, not the more flexible per-fold selector.

## Engineering evidence

- Source reconstructions contain all 81 frames for episodes 7 and 9.
- The six-frame association retains 672 and 730 Gaussian contributors,
  respectively, without graph-component bridging.
- Production and exploratory association artifacts agree exactly for episodes
  7 and 9 in IDs, material coordinates, slice weights, node tracks, and contact
  anchors.
- Missing multiview support is treated as zero prior reliability rather than an
  invalid episode.
- The focused association suite passes 5 tests; the full repository suite
  passes 502 tests with 1 skipped.

## Calibration gate result

The frozen calibration was attempted on the same `110:191` frame slice and
six-frame prefix used by the source result. It failed at association QA before
any future prediction metric was computed:

- episode 3 failed the required multiview mask-consensus check;
- episode 4 exceeded the frozen 30 mm Gaussian-identity match bound;
- episode 8 was not evaluated after the conjunction had already failed;
- no Warp calibration score, future CD, or future track error was computed;
- target episode 1 (`lift middle`) was not opened.

The registered decision is therefore `on_calibration_fail`: freeze the
negative result and keep the target sealed. The failure localizes the next
method work to two prediction-time assumptions: independently selected view
masks do not always form one calibrated 3D object, and equal-size Splatfacto
exports do not guarantee stable Gaussian ordering. Neither threshold may be
relaxed on this calibration cohort.

Even a positive target result is a same-object confirmation, not SOTA evidence.
The publication-grade test is a separately locked multi-object filament cohort
with official metrics, followed by sheet and volumetric association models.

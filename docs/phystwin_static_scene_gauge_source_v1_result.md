# PhysTwin Static-Scene Gauge Source Transfer v1: Result

## Decision

The frozen opened-source transfer gate **failed**. The static-scene gauge must
not advance to future-simulation assimilation or a fresh-object evaluation.

The sole development case, `single_lift_cloth`, had improved prefix manual
mean error by 7.47%. With the same frozen method on the other 21 PhysTwin
source cases, all three primary equal-case metrics worsened and every locked
gate check failed.

| Prefix metric | Raw | Gauge | Relative change | Case wins |
| --- | ---: | ---: | ---: | ---: |
| Manual mean error | 6.097 mm | 6.509 mm | **+6.75% error** | 8/21 |
| Manual RMSE | 9.145 mm | 9.720 mm | **+6.29% error** | 7/21 |
| Late manual mean error | 9.064 mm | 9.387 mm | **+3.56% error** | 9/21 |

The worst case, `single_push_rope_4`, regressed by 72.51% in mean error and
140.82% in RMSE. The median case regressed by 3.62% in mean error.

## Gate Accounting

| Locked criterion | Required | Observed | Pass |
| --- | ---: | ---: | :---: |
| Equal-case mean-error gain | at least 3% | -6.75% | no |
| Equal-case RMSE gain | at least 2% | -6.29% | no |
| Equal-case late-error gain | at least 2% | -3.56% | no |
| Mean-error wins | at least 14/21 | 8/21 | no |
| Worst mean-error regression | at most 10% | 72.51% | no |

Lower error is better. A negative gain denotes a regression.

## What Failed

This was not a support-starved or rejected-update result:

- all 63 camera-specific gauges passed the target-free background gate;
- mean held-out static-background error improved by 41.51%;
- mean object-track correction support was 71.90%;
- background cross-validation gain and manual-identity gain had correlation
  `r=-0.191`;
- support fraction and manual-identity gain had correlation `r=-0.206`.

The target-free gate successfully recognized a predictable tracker drift
field on static background pixels. That field did not transfer reliably to
moving deformable-object identities. Plausible contributors include
content- and motion-dependent tracker error and the sensitivity of
source-depth lifting to shifting an object query across depth boundaries.
The experiment does not identify which contributor dominates.

The scientific conclusion is therefore narrow:

> Spatially held-out background consistency is not sufficient prior
> reliability for correcting deformable-object tracks, even when background
> drift itself is predicted accurately.

This strengthens the existing requirement that a discrepancy update needs
object-relevant, physically supported evidence and an exact fallback. It does
not justify tuning another background-to-object transfer rule on these opened
cases.

## Admissibility Amendment

The initial worker sealed a gauge for `rope_double_hand`, then stopped before
scoring because two of nine manual identities were non-finite at frame zero.
One other case had written a score, but no value or aggregate was inspected.
The scorer was amended before resumption to use exactly the identities finite
at frame zero. The gauge, cohort, thresholds, and gate were unchanged. The
complete 21-case run used only the amended scorer and a fresh result root.

## Provenance

- implementation commit: `a3042b1`
- protocol digest:
  `70b55cf76ec25ed9212792ab2f0ae032c8e502710c00453d8dc26e903aade300`
- aggregate:
  `results/sota/phystwin_static_scene_gauge_source_v1/aggregate.json`
- aggregate SHA-256:
  `352abec306bc798ae081df106c2cc8a01b2c3b72b6e1ba0282bff681f9b6ed06`
- CoTracker3 cue-manifest SHA-256:
  `899fadb41531bfe27d7743d8ba055e16fab3521259d1e3b7cbf945059ca82175`
- remote complete artifact root:
  `/home/florianpfaff/bpt-static-scene-gauge-source-v1/results-70b55cf7`
- interrupted pre-amendment root, retained for provenance:
  `/home/florianpfaff/bpt-static-scene-gauge-source-v1/results-d7c15d14`

The result is opened-source observation-feeder evidence only. It reads no
future RGB, future manual track, future simulator outcome, held-v8 artifact,
or PokeFlex target.

## Next Step

Do not assimilate this gauge. The remaining credible automatic route is a
guarded, object-relevant belief update that requires agreement with physical
action support or an independent modality and preserves the unchanged
physical baseline on abstention. Any such method needs a new source lock and
must not reuse this cohort for confirmatory claims.

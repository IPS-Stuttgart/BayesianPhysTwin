# Deform360 Bias-Aware Prospective V2 Result

## Decision

The target-free support gate passed, but the fresh calibration accuracy gate
failed. Target access is therefore forbidden. No reserved target object, media,
or future was opened.

This is a calibration result against the exact selected raw
physical/persistence backbone. It is not official Deform360 Table-4 parity and
does not support a state-of-the-art claim.

## Frozen evidence

| Artifact | Result SHA-256 |
| --- | --- |
| Calibration cohort seal | `5d5317c9f7c1a35a28242490156cb1971f1a25f82dfbfb7200b175afe424b6c9` |
| Support gate | `58194bc33b9018e4b2ccf3e5e71bec39f911d88a98e0f8a69f66381c2413510a` |
| Accuracy gate | `44ac5447d05ba2d9c5fcc200b0bb63781cf1b6faf83d27ef1051be19365d5a2c` |
| Execution manifest | `8b579be8d4a52dd937c4e5da690880c837aec7c5f29c144cb45a64c7ce73aee8` |
| Post-open diagnostic | `41202e48caaf89bf5148d726fbec8b54e5ca430242db4e6539f7b79d3e74d0d0` |

The protocol config SHA-256 is
`67e1157fa04283f1376855a7ac60f85a4de02434612592ffe8b4ef1e4607ebe4`.
All eight automatic twins were reconstructed and scored only after the support
gate passed. The four target-free quality failures were retained without
replacement.

## Calibration result

Errors and differences below are object-level means over the three hidden
post-update intervals. Differences are candidate minus selected raw baseline;
negative is better.

| Object | Update intervals | Identity baseline/candidate (mm) | Difference (mm) | Chamfer baseline/candidate (mm) | Difference (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `076-rubber-bands` | 1 | 3.397 / 3.392 | -0.005 | 2.314 / 2.347 | +0.034 |
| `078-fishing-line` | 1 | 1.695 / 2.122 | +0.426 | 1.079 / 1.847 | +0.768 |
| `088-snake` | 0 | 0.585 / 0.585 | 0.000 | 0.748 / 0.748 | 0.000 |
| `161-tube` | 0 | 0.265 / 0.265 | 0.000 | 0.149 / 0.149 | 0.000 |
| `011-green-cloth` | 0 | 0.144 / 0.144 | 0.000 | 0.161 / 0.161 | 0.000 |
| `175-plastic-bag-cloth` | 1 | 1.005 / 1.441 | +0.437 | 0.993 / 1.325 | +0.332 |
| `163-bear` | 1 | 2.818 / 2.990 | +0.172 | 1.829 / 2.351 | +0.522 |
| `168-cat-big` | 0 | 0.010 / 0.010 | 0.000 | 0.002 / 0.002 | 0.000 |

Across all eight objects, the candidate regressed by 0.129 mm identity RMSE
and 0.207 mm Chamfer, corresponding to relative regressions of 10.38% and
22.75%. Four objects received an update, and all four were harmful on at least
one co-primary metric. All four rejected objects were bit-exact baseline
fallbacks.

## Gate audit

| Gate | Result |
| --- | --- |
| Support gate | Pass |
| New eligible calibration groups at least 5 | Fail: 4 |
| Combined eligible groups at least 9 | Fail: 8 |
| Finite-sample coverage at least 0.90 | Fail: 8/9 = 0.889 |
| Regret upper bound below -0.005 mm | Fail: +2.303 mm |
| Both object-balanced regrets negative | Fail |
| Zero harmful accepted objects | Fail: 4 |
| Every rejection bit-exact | Pass |

The prescribed failure action is active: publish the calibration failure and
keep every target sealed.

## What failed

The estimator itself and the exact-fallback boundary behaved as designed. The
failed component is transfer of the source-only acceptance certificate. All ten
candidate intervals in the open source development set had improved, but the
four candidates admitted on fresh calibration objects were all harmful on at
least one metric.

The post-open diagnostic identifies a concrete support mismatch:

- every accepted calibration update had one aggregated active view;
- every accepted update had zero independent anchors;
- directional physical agreement could be high even when update magnitude was
  weakly supported by the physical response;
- for fishing line, the physical response was only 3.34% of observed motion,
  while the state update was 9.83 times the physical-response magnitude.

The one-view/no-anchor result means the nominal shared/per-camera bias model was
not independently constrained at the point where the update decision was made.
This is consistent with the camera-only common-mode ambiguity, rather than
evidence that covariance tuning alone can repair the method.

## Next method boundary

V2 must not be rescued or retuned. A successor may use these now-open
calibration outcomes for development, but it needs a new independent calibration
cohort before any confirmatory target access.

The next credible candidate is narrower:

1. retain per-camera observations before triangulation or aggregation;
2. admit only state modes supported by action-conditioned physical response and
   independently redundant observations or another modality;
3. model shared camera bias explicitly;
4. bound correction magnitude relative to physical response;
5. use a conditional source-support regret upper bound, with bit-exact fallback
   outside support.

Until that method passes a fresh calibration gate, the selected raw
physical/persistence backbone remains the honest result.

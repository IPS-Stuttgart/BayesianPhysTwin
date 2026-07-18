# PGRD-to-Bayesian-PhysTwin development result

Status: completed development diagnostic on 2026-07-18. The source gate failed,
so the exploratory 19-case future cohort and every independent target remain
closed for this family.

## Question

PGRD is a newly released hybrid simulator that learns per-particle residual
velocity on top of a spring-mass backbone. Its official sloth checkpoint is a
high-value test of the hypothesis suggested by the Prob4D and Bayesian-PhysTwin
diagnostics: long-horizon gains require learned residual dynamics, while the
Bayesian endpoint anchor should remain the state estimate and safe fallback.

The test does not claim that a PGRD checkpoint trained for its own simulator is
already a Bayesian-PhysTwin method. It asks whether the released nonlinear
spatial-temporal residual features transfer without retraining.

## Reproducible boundary

The adapter pins upstream PGRD commit
`e294d96723054f77a1cfdd3c2c052de7b7cd9ce3` and official sloth checkpoint
SHA-256
`79cc402835b73d6f7dc38a59ea37531f52ea3d2909d434ed9a2a8673509e073c`.
PGRD remains an external lazy dependency; no upstream source or weights are
vendored.

The final adapter corrections are:

- deterministic farthest-point sampling of 512 observed object points;
- a right-handed gravity-frame map from PhysTwin `-z` to PGRD `-y`;
- 10 Hz PGRD inference over 30 Hz PhysTwin frames with interpolation;
- five allowed prefix steps to warm PGRD's temporal transformer;
- source-only selection over metric scale, yaw, and trust;
- exact preservation of the dense endpoint anchor, interpolating only dynamic
  changes from the sampled points;
- a 10 mm per-node cap and exact endpoint-persistence fallback;
- no future observations, MotionCrafter frames, or target outcomes as inputs.

The dense-anchor condition is important. An earlier engineering run
interpolated the entire endpoint field from 512 points. A near-zero learned
readout then changed metrics materially, revealing that the comparison had
degraded the baseline before evaluating PGRD. Those runs are invalidated; only
the corrected machine records under `results/sota/diagnostics/pgrd_adapter_v1`
and `pgrd_calibrated_v1` support the conclusions below.

## Frozen development gate

The reference is exact temporally filled endpoint persistence. A dynamic
candidate must improve the balanced validation score by at least 1% and may
not worsen either CD or track error by more than 2%. Future metrics are opened
only after this validation gate passes. The three sloth cases are already-open
development evidence and cannot independently confirm transfer.

## Results

Percent changes are relative to endpoint persistence; negative is better.

### Prefix-warmed anchored PGRD change

| Case | Validation CD | Validation track | Balanced | Gate |
| --- | ---: | ---: | ---: | --- |
| `single_lift_sloth` | -0.629% | -0.995% | -0.812% | fail |
| `double_lift_sloth` | +1.370% | +1.145% | +1.257% | fail |
| `double_stretch_sloth` | +1.790% | +2.687% | +2.238% | fail |

`single_lift` improves both metrics but does not reach the predeclared 1%
balanced threshold. Its future remains unopened. The other two actions reject
zero-shot transfer directly.

### Prefix-calibrated 3x3 velocity readout

This arm keeps PGRD frozen and fits only a spectrally bounded 3x3 map from its
residual velocity to observed residual increments. Thousands of point-prefix
pairs estimate nine parameters.

| Case | Validation CD | Validation track | Balanced | Future CD | Future track |
| --- | ---: | ---: | ---: | ---: | ---: |
| `single_lift_sloth` | -1.878% | -0.563% | -1.221% | -1.243% | +4.440% |
| `double_lift_sloth` | +2.396% | +2.333% | +2.364% | sealed | sealed |
| `double_stretch_sloth` | +2.127% | +3.140% | +2.633% | sealed | sealed |

Only `single_lift` passes validation. Its opened future improves CD but worsens
manual tracking enough to produce a 1.598% balanced regression. This repeats
the earlier per-case residual-velocity finding: short validation can select a
dynamic correction that does not transfer to the longer future.

## Interpretation

The released PGRD checkpoint is not a drop-in route past the PhysTwin state of
the art. Its absolute output is tied to PGRD's own optimized spring backbone,
and even causal frame/cadence correction plus a tiny prefix calibration does
not transfer across the three action regimes. The small two-metric
`single_lift` validation gain does show that its nonlinear features are not
pure noise, but it is insufficient to authorize the 19-case run.

The next credible learned-dynamics experiment must train the residual model on
PhysTwin rollouts rather than tune this zero-shot adapter further. The minimum
design is leave-one-interaction-out training across many source episodes, with
the actual PhysTwin next state as `x_sim`, the dense Bayesian anchor preserved,
and a long-horizon cross-action gate. PGRD's public architecture/checkpoints can
initialize that model, and PokeFlex or PGRD's public episodes can add breadth,
but final selection must occur on PhysTwin-source rollouts under the same
manual-track and CD metrics.

No larger released-case or independent run is justified by this result.

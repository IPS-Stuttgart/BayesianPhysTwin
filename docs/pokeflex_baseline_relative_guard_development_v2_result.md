# PokeFlex Baseline-Relative Guard v2 Development Result

## Scope

This is post-open method development using the 27 source takes and the opened
15-object public-paired v1 cohort. It cannot confirm transfer or support a SOTA
claim. Its purpose is to freeze one causal admission rule for a genuinely fresh
PokeFlex evaluation.

The candidate remains the source-selected weak Bayesian state update from v1.
For every supported target frame, the guard uses only quantities available
before that frame:

- correction-to-prior-motion RMS ratio;
- physical-prior motion RMS;
- correction/prior-motion cosine;
- correction/previous-correction cosine;
- association count; and
- raw update RMS.

A physical-object-grouped ridge model predicts candidate-minus-baseline regret.
An update is admitted only when its calibrated upper regret bound is negative
and its feature vector lies inside the source support. Every other frame uses
the released physical checkpoint exactly.

## Why A Guard Is Needed

An exact post-open scale control reran the registered mesh sampler at global
update scales 0.25, 0.50, 0.75, and 1.00. Every nonzero scale retained one
losing object. Reducing the update globally therefore cannot satisfy the
no-object-regression criterion; admission must depend on causal local evidence.

| Global scale | Mean improvement | Wins / ties / losses | Worst object |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.297% | 13 / 1 / 1 | -0.102% |
| 0.50 | 0.552% | 13 / 1 / 1 | -0.144% |
| 0.75 | 0.782% | 13 / 1 / 1 | -0.324% |
| 1.00 | 0.989% | 13 / 1 / 1 | -0.459% |

The scale-one control reproduces every sealed v1 object mean exactly.

## Development Evidence

The frozen settings are 80% object-level coverage, 80% within-object coverage,
ridge penalty 10, and a one-standard-deviation source-support margin.

| Evaluation | Mean improvement | Wins / ties / losses | Supported objects |
| --- | ---: | ---: | ---: |
| Leave-one-object-out source | 0.690% | 9 / 0 / 0 | 9 / 9 |
| Leave-one-object-out public v1 | 0.406% | 11 / 4 / 0 | 11 / 15 |
| Full-fit source replay | 0.685% | 9 / 0 / 0 | 9 / 9 |
| Full-fit public v1 replay | 0.388% | 12 / 3 / 0 | 12 / 15 |

Across 2,814 leave-one-object-out frame decisions, 816 updates are admitted:
759 improve and 57 regress, for a 6.99% false-safe rate. The held-object upper
bound covers 89.72% of in-support rows. Fully unsupported objects are included
as exact fallback ties, including the zero-support volleyball case omitted by
the first exploratory aggregation.

These numbers pass the development gates, but they are not prospective. The
same opened outcomes both motivated and evaluated this guard family.

## Frozen Next Test

A new protocol may serialize the fitted certificate and apply it without
refitting to newly selected PokeFlex takes absent from all development history.
Its primary gates should remain:

1. positive object-balanced CD improvement;
2. object-bootstrap 97.5% upper bound below zero for candidate minus baseline;
3. no physical-object regression;
4. at least 12 object wins; and
5. at least 12 objects with one or more admitted updates.

Prediction seals and a complete barrier must precede target-mesh access. A
rejected or out-of-support frame must remain byte-identical to the baseline.
The published 6.498 mm Kinect number remains cross-split context only.

## Evidence

- `source_rows.json`: `49e21b7aa6bda47b62b6d4475a8afaa4ebf73d1a737109a44630c5bc956f6ddb`
- `public_paired_raw_rows.json`: `45297345e9e9b366b23031655251b77ace8e96ce3c9570bfe48575f9c8186494`
- `alpha_scale_control.json`: `1ecf48c0af7c08e10edd9628619187d475c178ff55679b247ca4799fd821ac18`
- `development_evaluation.json`: `49007cc03f2ed10e59e3aa2588f2a5130b70d047e919d75200c2769143bf3c71`

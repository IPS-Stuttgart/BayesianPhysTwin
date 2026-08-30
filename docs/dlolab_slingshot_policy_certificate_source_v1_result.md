# DLO-Lab Slingshot policy-gain certificate source v1 result

## Decision

**Retained negative at the pre-future gate.** The policy-level certificate
admitted 12 of 288 fresh evaluation worlds, below the prospectively locked
minimum of 24. The run stopped before generating any evaluation future. No
retry, replacement, threshold change, or post-outcome score is authorized.

## What completed

- Frozen source revision: `0b62ce130efd342bf23c9751ebdca599c6b28b62`.
- Calibration: 96/96 all-action futures passed native QA.
- Evaluation: 36/36 prefix batches covering 288/288 fresh worlds passed native
  QA.
- Policy-gain calibration: rank 88/96, offset `0.035297203063964847`.
- Matched simultaneous-regret calibration: rank 88/96, offset
  `0.6162563288661695`.
- Prefix-only admission: 12 accepted, 276 exact fallbacks.
- Evaluation futures generated or read: 0.
- Technical failures and replacements: 0 and 0.

The calibration, candidate, and guarded-decision artifacts were reloaded under
the frozen source revision. The complete decision bundle reproduced
byte-for-byte, the failing barrier reproduced, and no `evaluation-future-*`
directory existed. The compact evidence is
`results/source/dlolab_slingshot_policy_certificate_source_v1/summary.json`.

## Interpretation

The retrospective 51-world capacity diagnostic did not transfer at the locked
admission rate. Its 6/51 admissions suggested useful headroom, whereas the
fresh panel admitted only 12/288. The large global conformal correction
(`0.0353` native reward) dominates the local five-neighbor gain estimates for
most fresh prefixes. Therefore this exact combination is closed:

```text
51-world five-nearest-neighbor gain predictor
+ one global rank-88 selected-policy residual offset
+ fixed -0.002 harm margin
```

This is not evidence that policy-level certification is ineffective in
general. It is evidence that an unnormalized global residual score cannot
retain enough decisions for this predictor under fresh-world transfer.

## Claim boundary

Because the pre-future gate failed, there is no prospective coverage, reward,
harm-risk, matched-comparator, benchmark, SOTA, or physical-safety result. The
96 labeled calibration worlds and 288 prefix-only evaluation worlds are now
development evidence. Their unopened evaluation futures must not be used to
retroactively rescue this method or select a successor. Any successor requires
a new calibration/evaluation roster and a new immutable protocol.

The useful next hypothesis is a normalized or explicitly heteroscedastic
policy-gain certificate whose scale is fixed from source-only residual and
support information. It must earn narrower local bounds without changing the
exact fallback or the marginal coverage accounting.

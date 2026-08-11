# Deform360 covariance-only cross-repository decision amendment v1

This target-closed amendment binds BayesianPhysTwin’s software custody to the
merged paper-side decision amendment before any confirmation payload,
prediction, or outcome is opened.

## Bound authorities

- BayesianPhysTwin custody protocol:
  `0f13d7a1f1610588ca9e7119f94814c99940fb31050419de16fa9cae06f683cc`,
  merged as `d337c5209c639430abe801a9688cc2788faa2aaf`.
- Original cross-repository binding:
  `531123205959a3d3d0549d9256b6ec222dca636198bc1e93f1b468d1a77c8f33`.
- Paper analysis protocol:
  `fa16c105e6d535d1e229ccf086fd69d05b2be74592b5c4e3f6c5289b8915fee3`.
- Paper decision amendment:
  `c78868d0397988d4ca4f438ba93ef0b02c6d07d031251dda9d8058eef4403bcc`,
  merged as `4e448ce7628b3826658fdffd8590cb680c500a88`.
- This software amendment:
  `efacabe4ceb6e1d3c4cd523e0959bdf16f8ff4253f9800e8de11574734623802`.

BayesianPhysTwin remains responsible for prediction custody, source receipts,
runtime identity, exact fallback, and the target-opening barrier. The paper
protocol plus its decision amendment govern statistical analysis and claim
wording. Any conflict fails closed and forbids target opening.

## Corrected software semantics

The parent software protocol’s `missing_unit_imputation` wording is superseded.
A missing or unscorable primary target outcome cannot be converted to a
zero-effect fallback tie, deleted, replaced, or imputed. It makes the
confirmatory analysis incomplete and claim-ineligible.

A zero effect is valid only when:

1. the primary target outcome is present and scorable; and
2. the covariance candidate and zero-covariance `last_residual` comparator emit
   the same registered fallback distribution byte-for-byte.

An incomplete analysis is neither a negative result nor evidence of
practical equivalence. Negative or statistically inconclusive evidence is
complete only for the complete ordered twelve-object target cohort.

## Decision gates

A covariance-mechanism result requires both the simultaneous max-t interval and
the Holm-adjusted exact all-`2^12` sign-flip gate for candidate minus
zero-covariance `last_residual`, plus:

- exact point identity for all twelve target units;
- exactly zero track and Chamfer differences;
- nonpositive sheet and volumetric mean NLL effects;
- exact registered fallback for every unsupported observed target unit; and
- complete ordered primary-score and audit records.

A claim of improvement over the physical fallback additionally requires the
corresponding simultaneous interval and Holm-adjusted exact sign-flip gate for
candidate minus physical fallback.

## Scientific boundary

This amendment changes no mean, covariance donor, scale, observation model,
cohort, endpoint, fallback, or target barrier. It opens no data and authorizes no
target execution, claim promotion, deployment, physical-state interpretation,
Causal4D conclusion, benchmark-parity statement, or state-of-the-art claim.

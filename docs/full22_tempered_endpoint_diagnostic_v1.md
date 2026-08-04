# Full-22 Tempered Endpoint Diagnostic v1

## Status

This is a frozen retrospective mechanism diagnostic. The released PhysTwin-22
cohort and its three/19 development split are already open and have informed
method development. A positive result can justify a new grouped-calibration
protocol, but cannot establish independent transfer, calibrated deployment
uncertainty, or a state-of-the-art claim.

## Motivation

The historical endpoint model average slightly improved the selected Bayesian
anchor but did not beat the last-supported-residual baseline. More importantly,
its cumulative prefix likelihood collapsed the 15-component family to a median
effective component count of approximately one. Between-model covariance was
therefore negligible and nominal 90% coverage fell to 38.0%.

The new candidate tests one concrete explanation: temporally correlated prefix
observations were allowed to contribute unbounded model-selection evidence.
For material track (i), component (k), accepted prefix count (n_i), and
frozen cap (N_{\mathrm{eff}}), it uses

```text
power_i = min(1, N_eff / max(n_i, 1))
weight_ik proportional to prior_k * exp(power_i * log_evidence_ik).
```

The complete robust filters, component family, physical trajectory, graph lift,
10 mm correction cap, and official future metrics remain unchanged. Tempering
changes only the evidence used to select among complete endpoint filters. Raw
covariance remains model based; calibration still requires independent groups.

## Prefix-Only Selection

Candidate effective-count caps are `1, 2, 4, 8, 16`, plus `1e6` as the
historical untempered control. For each case:

1. fit every candidate only through `fit_end`;
2. score constant endpoint residuals on the permitted interval
   `[fit_end, train_end)`;
3. choose minimum vector RMSE, breaking ties toward the smaller cap;
4. compare it with the last residual available at `fit_end`;
5. admit only with at least 0.1 mm and 0.5% validation improvement;
6. refit the selected cap through `train_end` and predict the untouched future.

A rejected case returns the exact last-residual trajectory. No future frame or
future metric participates in cap selection or admission.

## Retrospective Advancement Gate

On the 19 historical confirmation cases, the guarded candidate must:

- improve both equal-case Chamfer and track error by at least 0.5% relative to
  last residual;
- win both metrics in at least 12/19 cases; and
- keep every case-metric regression at or below 2%.

Failure closes this candidate without fresh evaluation. Passing would authorize
only design of a new object/session-grouped calibration and prospective target
protocol. It would not convert this opened-cohort result into confirmation.

The immutable machine-readable contract is
`protocols/full22_tempered_endpoint_diagnostic_v1.json`, with canonical SHA-256
`cc73650107eb44fbece39dd68e29c0b53ee880c66e6ca293a61ebcf2f86e27e8`.

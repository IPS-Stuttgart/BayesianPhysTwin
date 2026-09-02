# DEFORM DLO4/DLO5 support-aware outer certificate

This experiment tests a two-level decision certificate on the already opened public DEFORM DLO4/DLO5 decision panel.

The existing inner certificate is exact over the supplied finite physical support, quotient masses, and loss matrix. The earlier gate-risk audit showed the missing transport layer: 45 of 82 held nonfallback decisions had realized regret above the represented-support bound, even though all 82 satisfied the registered finite-support contract.

The outer gate asks whether that support misspecification is predictable **before the held outcome is read**. It uses only diagnostics emitted by the inner certificate: quotient concentration, kernel concentration, expected action gap, expected fallback advantage, hypothesis-action agreement, residual disagreement, unsupported specificity, the registered worst-case regrets, decision time, DLO identity, and the selected action identity.

## Source construction

The source panel is reconstructed from the immutable source artifact of workflow `33473378340`.

For each of the 112 official DLO4/DLO5 training trajectories:

1. remove all 19 windows of that complete trajectory;
2. refit the frozen parent model to the other 55 trajectories of the same DLO;
3. evaluate all 19 omitted windows;
4. record whether realized regret exceeds the inner certificate's represented-support bound.

This yields 2,128 leave-one-trajectory-out source decisions. The outer classifier is cross-validated with complete trajectories as groups. Its operating threshold is selected only from source out-of-fold predictions using an equal-trajectory block bootstrap.

## Retrospective prototype result

The frozen prototype obtains:

- grouped source AUC: **0.9640**;
- descriptive held AUC: **0.9333**;
- inner certificate: 82/532 nonfallback decisions, 45/82 support-bound violations, 44/82 regret-tolerance violations;
- outer + inner: 27/532 nonfallback decisions, 2/27 support-bound violations, 2/27 regret-tolerance violations;
- normalized-regret gain versus exact fallback: **12.92%** for the inner certificate and **5.25%** for outer + inner;
- strict 90% trajectory-maximum conformal inflation accepts **0/82** held inner-certified decisions, showing that a single global worst-case inflation collapses to fallback.

The outer gate therefore reduces the observed support-bound violation fraction from **54.88% to 7.41%** while retaining **32.93%** of the inner certificate's nonfallback decisions.

## Claim boundary

This is **retrospective method-development evidence** on an already opened public held panel. The held AUC and held outcomes are descriptive only and do not fit the model or operating threshold. The source operating point is controlled by a trajectory-block bootstrap, not yet by a distribution-free selective-risk theorem. The result does not establish fresh confirmation, unseen-object generalization, deployment safety, or a population-level regret guarantee.

The intended next step is a group-valid selective-risk certificate for the ratio of support violations to selected actions, followed by one untouched public-data confirmation.
# Causal4D Pre-Acquisition Amendment v4

## Status

V4 supersedes v3 before any physical execution. It changes no source-panel or
confirmatory execution, target ID, calibration split, signature threshold, or
mechanism-promotion threshold. It adds controlled gate evidence, a prospective
state/reset prediction, and stricter contact-registration provenance.

Canonical amendment SHA-256:

```text
0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f
```

## Gate controls

The frozen v3 gate requires at least 10% geometric-mean held-out correction
shrinkage in at least 8 of 12 source sessions, plus its track, late-track, and
CD gates. V4 runs that exact rule through 512 synthetic 12-session panels per
arm with the same three-fold 8-fit/4-held-out boundary and six-frame readout
refit.

The positive control is a known 10% actuation-gain loss. The matched placebo
uses the exact same nine-value scalar response bank after a 90-degree coordinate
rotation. This preserves every candidate response norm and model flexibility
while destroying the causal direction.

| Arm | Full-gate passes | Rate | 95% Wilson interval |
| --- | ---: | ---: | ---: |
| Placebo in null world | 0/512 | 0.00% | 0.00-0.745% |
| True gain mechanism | 478/512 | 93.36% | 90.86-95.21% |
| Placebo in positive world | 0/512 | 0.00% | 0.00-0.745% |

The controlled result supports retaining the v3 10%/8-of-12 threshold without
post-control adjustment. It does not estimate a real-world false-positive rate.
The checksumed evidence and a local threshold-sensitivity table are in
`runs/causal4d_preacquisition_v4/mechanism_gate_controls.json` (result SHA-256
`f48e1fbc650daf7b81a317085e3bf038087dda3d459299a4e34c61777e4a5761`).
An isolated `gpuserver6000` run with PyRecEst 2.4.1 passes 328 tests and
reproduces that result hash exactly; the verification record is stored beside
the control evidence.

## State propagation

The released trajectory audit is summarized locally as

```text
delta_x_T approximately Phi_a(T, t_p) delta_x_t_p
```

`Phi_a` is only an empirical secant or local linearization at the injected
magnitude. It is not a globally valid transition matrix. Contact-mode switching
can make the response nonsmooth, so the sign-changing `double_stretch` pattern
may reflect switching as well as smooth rotation or cancellation.

The frozen interaction-specific interpretation is:

- `single_lift`: state injection captures 82.67% of readout CD gain and 87.38%
  of track gain, mostly through mode 0; state error is plausible there.
- `double_lift`: contraction plus 73.56% final leakage outside the injected
  rank-4 basis.
- `double_stretch`: near/middle/far aligned retention of
  `+5.55%/-36.38%/+47.98%`, consistent with redistribution and cancellation.

The three attachment-coverage correlations remain descriptive only.

## Prospective mode-0 check

Before the slip/reset pilot, v4 predicts that a reset/registration explanation
for `single_lift` should have the right scale. The released mode-0 reference is
13.736 mm per-node vector RMS. The pilot will calculate the 95th percentile of
fresh-reset mode-0 RMS in the locked world frame, add the preregistered 95%
registration uncertainty, and separately report rigid translation, best-fit
SE(3), and post-SE(3) low-rank components.

The reset-scale explanation is weakened if the released 13.736 mm reference is
more than twice that pilot statistic. Scale compatibility does not confirm the
cause.

An action-dependent propagated state correction is now a named source-panel
candidate. Its parameters must use the same 8/4 cross-fitting and v3 eligibility
gates. It cannot be fitted on released cases or confirmatory targets.

## Contact registration

Schema 3 retains the selected weighted node patch and at least one rejected
candidate for every contact region. Each candidate records its node weights,
world centroid, rationale, independent artifact, and confirmation that target
outcomes were not used. The selected candidate must exactly match the approved
attachment.

Generate the incomplete operator template with:

```bash
causal4d-contact-registration template \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/contact_registration_v3.template.json \
  --camera-id camera_0 \
  --camera-id camera_1 \
  --camera-id camera_2 \
  --object-node-count 6895
```

Legacy schema-2 artifacts remain readable, but v4 physical approval requires
schema 3. No physical contact has been registered or approved.

## Commands

```bash
causal4d-audit-mechanism-gate-controls \
  runs/causal4d_preacquisition_v4/mechanism_gate_controls.json

causal4d-preacquisition-protocol-v4 validate \
  configs/causal4d/sloth_multi_action_v1.json \
  configs/causal4d/sloth_preacquisition_v2.json \
  configs/causal4d/sloth_preacquisition_v3.json \
  runs/causal4d_preacquisition_v4/mechanism_gate_controls.json \
  configs/causal4d/sloth_preacquisition_v4.json
```

The physical sequence remains contact registration, independent approval,
slip/reset and synchronization pilot, then the locked 12-run source panel.

# Deform360 action-response source v1

## Purpose

This is a frozen, outcome-blind smoke of the new
`ActionResponseAdmissionV1` certificate. It uses the already-open
`059-shoe-ep0000` source carrier because its previous hidden audit showed that
the selected persistence baseline was exceptionally strong. The hidden result
is not an input to this run.

The smoke asks only whether causal RGB prefixes contain a spatial object
response that is:

- supported by the sealed PhysTwin action rollout;
- accompanied by a nontrivial controller trajectory;
- consistent across three disjoint camera panels;
- identifiable after shared translation is removed; and
- strong enough under reprojection-derived metric covariance.

It does not score a candidate or inspect any future material identity.

## Frozen inputs

- Physical source carrier:
  `deform360-dynamic-tapnextpp-provider-v1`, case `059-shoe-ep0000`.
- RGB and calibration:
  the already-processed source episode bound by that carrier.
- Tracker:
  AllTracker source and checkpoint already hash-locked by
  `RawCameraObservationConfig`.
- Causal updates:
  frames `19`, `38`, and `57`.
- Queries:
  16 deterministic frame-zero physical identities.
- Camera evidence:
  eight selected cameras partitioned into three disjoint azimuth-spread
  groups.
- Admission:
  the default `ActionResponseAdmissionConfig` committed with this protocol.

The prediction-only controller carrier is permitted because it contains the
known action and only frame-zero object geometry. No `final_data.pkl`, future
object point cloud, target trajectory, hidden identity, tactile future, or
outcome metric may be opened.

## Decision

This smoke has no advancement power by itself.

- Rejection is a useful negative control: the previously harmful update would
  have been blocked before candidate construction.
- Admission only establishes provider competence on one source prefix. It does
  not authorize target scoring or prove that a candidate beats persistence.

A larger source study is justified only after this run validates artifact
construction and the complete-belief exact-fallback path. A fresh-object
evaluation still requires source-object transfer and a frozen
baseline-relative regret certificate.

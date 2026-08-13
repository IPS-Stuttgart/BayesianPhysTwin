# MatPhys plus automatic CoTracker selected-overlay source result

## Decision

The frozen composition fails and is stopped. Applying the automatic CoTracker3
source-depth correction after the object-disjoint MatPhys/Bayesian selected
overlay improves Chamfer distance but materially worsens identity tracking.

This is a post-open source diagnostic on the previously examined PhysTwin-22
cohort. It is not independent evidence or a state-of-the-art comparison.

## Registered result

Lower is better. Values are equal-case means.

| Arm | Chamfer | Track | Change vs selected overlay | Joint wins | Worst regression |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected MatPhys/Bayesian overlay | 10.242 mm | 19.059 mm | - | - | - |
| Fixed automatic scale | 8.639 mm | 19.375 mm | -15.65% / +1.66% | 11/22 | +69.44% |
| Frozen causal temporal arm | 8.575 mm | 22.808 mm | -16.28% / +19.67% | 7/22 | +136.72% |

Only the causal temporal arm controlled advancement. It passed the 5% Chamfer
improvement gate and failed every other gate: track improvement, 16/22 joint
wins, at most 10% worst regression, and the joint `8/15 mm` operating point.

## Model-class limit

After freezing the failure, a diagnostic oracle was computed over the same
automatic candidate bank. Even independently choosing the best candidate for
each case and each metric reaches only:

| Diagnostic oracle | Chamfer | Track |
| --- | ---: | ---: |
| Two registered update arms plus exact baseline | 8.172 mm | 18.036 mm |
| Entire historical automatic candidate bank plus exact baseline | 7.909 mm | 16.611 mm |

The second row is an optimistic, post-open upper bound and cannot be reported as
a method. More importantly, it remains 1.611 mm above the 15 mm track context
value. A better selector cannot make this candidate bank cross both values.

## Interpretation

The selected physical overlay and source-depth correction are not simply
additive. The automatic field primarily improves point-cloud geometry while
degrading the material identities used by the track metric. This agrees with
the earlier disjoint-identity result: the unresolved problem is automatic
material correspondence, not additional spring-field strength or selector
tuning.

Do not rerun this composition with another cap, temporal coefficient, or
cross-family selector on PhysTwin-22. The next justified observation experiment
must add genuinely different identity information and retain exact fallback.
TAPNext++ is the current candidate because one sealed high-motion prefix test
showed strong identity accuracy, although its support gate failed and therefore
requires a new support-preserving design before any state update.

## Provenance

- frozen implementation and protocol commit: `f9fafec3ad577bfb9600d78fe926bfe8ae8f6266`
- protocol SHA-256: `8b044af71530cda3c374c03f31a1c7eb4d8891d8ba3686b76f372b88c0ce6678`
- CoTracker cue manifest SHA-256: `899fadb41531bfe27d7743d8ba055e16fab3521259d1e3b7cbf945059ca82175`
- MatPhys family selection SHA-256: `5eedb6cb5a747b856c0af696c5029038a8022f00828f43295f201578a4494890`
- raw report SHA-256: `67d4d222172610d571f0691ea0c60abd4f6fd34e68552a4750a38838afc12927`
- compact decision SHA-256: `87ebe1aa94785e7e7f50bd1f0975ea65417786e6fa4cb79330d8d6d77ef03201`
- durable evidence root: `/mnt/corsair/florianpfaff/matphys-cotracker-selected-overlay-source-v1-f9fafec3`
- no held-v8 artifact or process was accessed

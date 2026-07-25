# Deform360 Dynamic-Window Source Smoke

## Status

This is an exhausted-source, post-open diagnostic. It does not revise the
frozen `deform360-selective-virtual-sensing-v1` result, support a prospective
accuracy claim, or authorize evaluation on fresh or reserved objects.

The frozen v1 window selector averaged all five rows of the released
Deform360 action tensor as if they were Cartesian points. In the released
schema, row 0 is end-effector translation, rows 1--3 are a rotation matrix,
and row 4 contains aperture metadata. The v1 negative remains frozen. The
source-only v2 selector instead maximizes closure- and tactile-contact-weighted
end-effector translation after the first update. It reads no object geometry,
tracks, or target metric.

## Sealed smoke

The strongest target-free source candidate was selected before its corrected
outcome was built:

- case: `035-wipe-cloth-ep0005`
- raw window: `[98, 179)`
- contact-supported future gripper path: `52.070 mm`
- prediction cameras: 8 selected from 32 valid frame-zero views
- reconstructed material points: 598
- code revision: `ec077bf2f14c1f1a4dcd44707950b537727b34a4`
- prediction sealed before any future RGB, dense reconstruction, particle
  tracks, or target metric were read

The official Deform360 reconstruction shows only `0.929 mm` target-motion
RMSE. Contact-supported actuator travel therefore did not establish a
nontrivial deformable-object response.

| Arm | Hidden identity RMSE | Hidden Chamfer | Change vs persistence |
| --- | ---: | ---: | ---: |
| Exact persistence | 0.529 mm | 0.304 mm | reference |
| Pairwise-consensus RBF update | 9.017 mm | 7.618 mm | +1605.21% / +2407.85% |
| Support-gated raw RBF | 18.152 mm | 16.228 mm | +3332.75% / +5242.08% |
| Independent CPD | 18.321 mm | 15.962 mm | +3364.74% / +5154.46% |

The camera measurement error was `31.078 mm` on average and `49.693 mm` at
the 90th percentile. Nevertheless, the pairwise gate accepted the first two
updates because 13/13 and 10/12 centers were mutually compatible. It fell
back exactly only at the third update. This is another direct example of the
common-mode ambiguity: internally coherent cameras can agree on a biased
motion field while the object is nearly static.

## Decision

The smoke fails both prerequisites for cohort expansion:

1. the selected window does not contain a sufficiently large object response;
2. the sealed camera update is much worse than exact persistence.

No other case in the 24-case source cohort will be opened under this method,
and no fresh-object protocol is justified for it.

## Method implication

Known action, gripper closure, and tactile contact are necessary causal
context, but they do not prove that the object followed the actuator. A
camera-only consistency gate cannot identify coherent shared bias. The next
Bayesian-PhysTwin candidate must therefore use:

1. an action-conditioned physical rollout as the unchanged baseline;
2. an explicit shared camera/time/spatial bias variable;
3. structurally redundant evidence or an independent modality;
4. a source-calibrated upper bound on regret relative to the baseline;
5. bit-exact fallback whenever improvement cannot be certified.

Dynamic-window selection for any future protocol must additionally require
target-free, gripper-excluded evidence of object response. The source smoke
shows that contact-supported actuator path alone is insufficient.

## Provenance

- source-window selection result SHA-256:
  `42e345602dd8a7bc325d39d94270b66f4740645b04f9c097bbafdd774d033b84`
- sealed prediction result SHA-256:
  `ba71aadfd0843275735b439540104afb34a9d862fd72739ac03f0d53615f002b`
- source evaluation result SHA-256:
  `02feea88d99b0a7ae63158ad7934a3c3cb6c5949f29d2b816ce17aa3feb37a47`
- compact evidence:
  `results/sota/deform360_dynamic_window_source_smoke_v1/summary.json`

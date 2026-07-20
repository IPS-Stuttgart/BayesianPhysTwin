# PokeFlex Force-Supported Contact Result

## Status

This post-open development family did not pass the existing 5% escalation
threshold. Development takes `T2`, `T7`, and `T8`, all calibration objects, and
all target objects remain unopened for this method.

The family was evaluated only after the frozen action-guard result had been
opened. It asks whether the measured three-dimensional force vector supplies
enough independent physical information to reject camera-derived state
innovations in unsupported directions. It is a mechanism diagnostic, not a
prospective confirmation.

## Method

At causal source frame `t`, the accepted Bayesian registration update gives a
camera-derived state innovation `d_t`. The measured tool motion and force
define a local contact neighborhood and one of four restricted innovations:

- projection onto the force direction;
- projection onto the measured tool-motion direction;
- projection onto their joint plane;
- a force-parallel local mean.

The action-velocity mismatch is retained inside the same local support. Four
object-relative support radii (`0.25`, `0.4`, `0.55`, and `0.7`) and four
nonzero gains (`0.125`, `0.25`, `0.5`, and `1.0`) form 64 nonzero candidates.
The released checkpoint is returned exactly when camera or action support is
absent. No target-frame geometry enters a candidate prediction.

## Development Result

The 20 already opened `T1`, `T4`, `T5`, and `T6` takes contain 1,418 causal
target frames across five development objects.

| Object | Released checkpoint CD_UL1 (mm) | Best force-supported CD_UL1 (mm) | Change |
| --- | ---: | ---: | ---: |
| 3dPrintedHeart | 3.695 | 3.604 | -2.44% |
| FoamDice | 6.034 | 5.792 | -4.01% |
| MemoryFoam | 2.350 | 2.318 | -1.36% |
| PlushOctopus | 5.585 | 5.464 | -2.16% |
| ToiletPaperRoll | 5.581 | 5.371 | -3.76% |
| **Object-balanced** | **4.649** | **4.510** | **-2.99%** |

The best fixed candidate projects the innovation onto the force/action plane,
uses a support radius of `0.7` times the object radius, and applies gain
`0.25`. It improves all five object means, but its 2.99% object-balanced gain
does not pass the 5% threshold. It is also 0.006 mm worse than the earlier
post-open action-local candidate, whose gain was 3.12%.

The per-frame oracle over all 64 nonzero candidates reaches 8.80%
object-balanced improvement. That remains diagnostic selection headroom, not
evidence for a deployable selector. The previous camera-only regret audit
already showed that the available two depth views cannot identify that oracle
under coherent bias.

## Interpretation

The force direction provides a useful inductive restriction: unlike the
frozen force-magnitude gate, the best projected arm improves every object.
However, it does not explain enough of the checkpoint error to justify opening
reserved data. PokeFlex force data can support a local state-update prior, but
directional projection alone is not the missing SOTA-level mechanism.

This closes the current PokeFlex development search. Further progress should
not come from tuning more camera-derived fields on these same takes. A new
attempt needs a separately locked hypothesis with stronger independent
information, such as calibrated contact geometry, tactile/depth anchors, or a
physical model that predicts force jointly with deformation.

## Claim Boundary

- This is a post-open development result, not a prospective or SOTA claim.
- The published PokeFlex Kinect reference is not directly comparable to this
  five-object development split.
- No reserved, calibration, or target outcome was inspected.
- The complete compact evidence and source hashes are stored in
  `results/sota/pokeflex_force_supported_contact_v1/summary.json`.

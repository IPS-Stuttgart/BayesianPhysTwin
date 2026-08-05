# Deform360 tactile-prompted carrier development result

## Result

The source-only development run on the already-open `026-sock-cloth` case
constructed one ambiguity-preserving, bias-aware metric carrier. The method
used causal tactile geometry to select object-facing SAM2 masks, then used a
second camera only to validate geometry and inflate covariance. It did not
gain precision by treating correlated camera evidence as independent.

| Tactile assignment | Status | Reference / support camera | Mutual blocks | p90 distance | Supported nodes |
| --- | --- | --- | ---: | ---: | ---: |
| Direct | Exact fallback | none | 0 | n/a | 0 |
| Swapped | Development carrier admitted | `013_cam1` / `019_cam1` | 7 | 13.56 mm | 45 / 128 |

The admitted branch represented 128 physical-backend nodes using 117 unique
fixed `8x8` image-block information clusters. Repeating a cluster therefore
did not manufacture independent evidence. The estimated cross-view shared
bias was `[0.28, 3.50, -3.65]` mm. The covariance contribution from the
second view was positive semidefinite, with a minimum added eigenvalue of
`2.5e-5 m^2`; it could only maintain or reduce confidence relative to the
reference view.

The two tactile-to-finger assignments retained their original prior mass.
The failed branch remained an exact baseline fallback rather than being
silently removed.

## Meaning

This result repairs the failure mode exposed by the preceding strict
three-view carrier smoke. Tactile geometry can identify an object-facing
mask in two views even when an automatic segmentation selects robot hardware
and a third camera has no eligible object mask. The resulting carrier also
preserves the uncertainty that the source evidence cannot resolve:

- tactile ambiguity remains an explicit mixture;
- dense nodes inherit fixed information-cluster identities;
- local metric covariance contains a 5 mm floor;
- two-view disagreement becomes shared-bias covariance;
- unsupported nodes receive additional variance and lower reliability;
- no PhysTwin innovation or future residual enters prior reliability.

This is not yet evidence that the carrier improves Bayesian-PhysTwin. The
case was used to develop the mask convention and cross-view rule, and no
state update or prediction score was evaluated. The next admissible test is
an identically frozen run on an independent calibration object, followed by
an exact fallback unless all source-only admission gates pass.

The recorded boundary is therefore:

- exploratory source-only carrier feasibility on an already-open case;
- no calibration score access;
- no state, contact, or discrepancy update;
- no future camera or tactile input;
- confirmation and target payloads unopened;
- no held-v8 access;
- no SOTA claim.

## Reproducibility

The original run and independent replay produced byte-identical artifacts:

- implementation revision: `f3aa067bc5dc75bfd7bef4718c24db43d055ac28`;
- full result artifact ID: `d6e51e9ae69de981f41cce66771b59b441a3ed2ebf93f8405146786e9d184ccc`;
- full result SHA-256: `73d5301d759b4bb9c21f9e282c6bb073827774fa79f4713c7ac5bd280ad0bb8d`;
- carrier NPZ SHA-256: `697cdf3fae0b3792c7d62e1abc7e0844d92c28d35fd96b9e8441368090877a17`.

The full source-only artifacts are preserved at
`/home/florianpfaff/source-only/deform360-tactile-prompted-carrier-development-v1-f3aa067b`.
Focused verification passed 6 tests in both the local worktree and the exact
remote checkout.

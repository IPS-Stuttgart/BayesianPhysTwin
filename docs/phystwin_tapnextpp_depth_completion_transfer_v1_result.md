# TAPNext++ depth-completion source-transfer result

## Decision

The frozen eight-case opened-source transfer gate **passed**. Seven of eight
cases passed every per-case provider gate. Aggregate support was 90.09%,
case-balanced identity RMSE was 4.658 mm, and case-balanced gain over exact
persistence was 78.38%.

This authorizes a separately frozen Bayesian-PhysTwin assimilation study on
already-open source cases. It is not yet evidence that a future physical
rollout improves.

## Results

| Case | Support | Candidate RMSE (mm) | Persistence (mm) | Gain | Endpoint (mm) | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `double_lift_cloth_1` | 100.0% | 4.707 | 13.016 | 63.83% | 7.529 | yes |
| `double_stretch_sloth` | 100.0% | 3.183 | 54.313 | 94.14% | 3.577 | yes |
| `double_lift_zebra` | 100.0% | 2.411 | 27.187 | 91.13% | 3.417 | yes |
| `rope_double_hand` | 100.0% | 7.285 | 31.378 | 76.78% | 8.278 | yes |
| `single_lift_rope` | 100.0% | 3.368 | 32.537 | 89.65% | 4.123 | yes |
| `single_push_rope_4` | 34.2% | 3.717 | 5.908 | 37.09% | unavailable | no |
| `single_lift_dinosor` | 100.0% | 4.218 | 37.912 | 88.88% | 4.954 | yes |
| `weird_package` | 88.2% | 8.377 | 57.901 | 85.53% | 14.015 | yes |

The failed push case was retained. Its source-qualified camera could not add
support beyond the strict 30/80-row carrier, so exact fallback preserved the
low-support carrier and the support/endpoint gates failed. No case was replaced
or rerun with altered settings.

## Observation-path behavior

The strict multiview carrier already covered every row in three cases. In four
other cases, source-qualified single-camera RGB-D lifting filled at least some
abstentions while preserving every strict row exactly. The remaining push case
abstained. This is the intended selective behavior: completion is useful when
carrier-calibrated camera evidence exists and otherwise leaves the original
carrier unchanged.

## Provenance

- Prediction/source-lock commit:
  `a5b039cc7a28c19debae70f019d6117d496c8f0c`
- Evaluation-custody commit:
  `628d9209ea0bbd38212278762776a5c75db7c116`
- Aggregate summary file SHA-256:
  `ed555519be8f5bcc6e8a3734b9ce536f3692f4a1ee7a9bd97392e9861b7ae1d9`
- Aggregate canonical result SHA-256:
  `08e24ba766b2904312bfc8898fb9bfd92ffeae37e3e9de544a864ceff7fe8dc6`
- TAPNext++ revision:
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`
- TAPNext++ checkpoint SHA-256:
  `6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f`

All strict reports and seals, completion reports and seals, per-case results,
and the aggregate summary are archived under
`results/sota/phystwin_tapnextpp_depth_completion_transfer_v1/source_result/`.

## Claim boundary

The tracker saw only each causal RGB-D prefix, prefix masks, and the manually
identified query positions at the source frame. Manual trajectories over the
rest of the prefix were opened only after both prediction stages were sealed.
No future manual track, future RGB-D frame, future Chamfer distance,
Bayesian-PhysTwin rollout metric, held-v8 artifact, or independent target was
opened.

The result establishes transferable source-prefix observation competence. The
next experiment must determine whether a guarded Bayesian state/discrepancy
update improves future physical prediction rather than merely tracking the
observed identities accurately inside the prefix.

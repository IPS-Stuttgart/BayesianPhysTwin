# Deform360 official-Hub Stage-1 acquisition result

## Scope

This milestone acquired and aligned the ten locked calibration episodes for
`deform360-official-hub-visuotactile-v1`. It did not open any confirmation
object tree or payload, inspect a confirmation outcome, fit a contact mapping,
select a provider, or evaluate a Bayesian-PhysTwin prediction.

This is therefore acquisition and preprocessing evidence, not a method result.
The later official annotation stages (`masks`, `robot`, `gripper-masks`,
`reconstruct`, `depth`, `tracking`, `pcd`, and `control-points`) remain pending.

## Locked inputs

The implementation used:

- dataset `brownu/deform360` at
  `f804696d7a133908c7497ffdab43819d879b5cbc`;
- official processing `lhy0807/deform360` at
  `d8522a4403b766aeb387510c04e89032a56fdf35`;
- protocol SHA-256
  `55534067fb0b3d7965eb66438cbec2ac5b85bcf5378abd1a73785479a5cdbeab`;
- selection artifact SHA-256
  `dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82`;
  and
- Stage-1 implementation commit
  `f32f5c4fc8212cdfec578172286234f796d39d0d`.

The exact generated manifests are retained in
[`results/sota/deform360_official_hub_visuotactile_v1_stage1/`](../results/sota/deform360_official_hub_visuotactile_v1_stage1/).

| Artifact | Content identity | File SHA-256 |
| --- | --- | --- |
| Preflight | `c2e771e5407a36b9436974299b6e1aff67298e0c27cc2ed324e084f8b46f1d3d` | `84081f31e0dafcdba26394ef7c89e4a89b2460bf4d564b27ed598ee54513f6b4` |
| Download | `fafea9f42049976de0c9822cee0201bb6835b7a2fde93a115f6323cea797c6e3` | `063afbcef3b59a19422accb150fb436254d9933789bfc971dc9708afffb82da0` |
| Offline verification | `7e8198b82b1cbaeff96a925a56b06369934f4527140a80d6c159a449ac2be1fd` | `ad551eeabf1a3608d769032ad3b59d40702a785ba7094256576eabbb854bafef` |
| Processing view | `9b55437271aa8775fd31cce0f7a300224d59c53b6e72e1cbf5772022dc56c111` | `aad0895aee2e74442e770da00473713a4a600b5604a04f3cb070995bd55aeff4` |
| Processing report | `9cee90180f5b978ff3bd054958a626943514a72288c7f610a814958bb95799b7` | `90366f930d71d20932c4decc193652a6d5e2f59af0350e7afd8b76408fe3ef98` |

## Acquisition result

The preflight admitted exactly:

- 10 calibration objects;
- 908 files;
- 1,069,932,695 bytes;
- one exact-stem camera recording and timestamp pair per admitted camera;
- one exact-stem tactile recording and timestamp pair per sensor; and
- the latest tactile median baseline strictly preceding each recording.

Every payload file was verified against its Hub LFS SHA-256 or Git blob
identity. A second network-disabled pass reused and rehashed all 908 files. The
payload was then copied directly from `gpuserver4090` to
`gpuserver6000:/mnt/lexar4tb`; a checksum-mode `rsync` comparison reported no
differences. The temporary transfer credential was removed from both servers.

Released-data irregularities were retained rather than normalized silently:

- `026-sock-cloth` uses the released metadata object name `026-sock`, an
  explicitly allowlisted alias;
- six orphan camera timestamp stems for `186-monster` and two for `193-frog`
  were recorded and excluded because every selected MP4 still had an exact
  sidecar; and
- an invalid `nonprehensile` field in a nonselected `193-frog` episode was
  recorded. The locked selected episode remained valid.

No failed or malformed calibration object was replaced.

## Episode mapping

Selective staging leaves one recording in each stream. The pinned upstream
processor consequently assigns it local episode index zero even when its
original released episode ID differs. The typed processing-view manifest binds
that mapping for every object. Media, timestamps, calibration arrays, and
tactile baselines are read-only links to the verified payload. Only the view's
`metadata.json` is derived, with sequence zero copied from the locked original
episode. The original payload metadata remains unchanged.

The pinned upstream layout independently found exactly one camera episode and
one tactile episode in every stream in the view.

## Processing result

Official undistortion, camera synchronization, tactile normalization, and
tactile-to-camera alignment completed for all ten objects:

- 10/10 ordinary successes;
- 0 retained technical failures;
- 3,024 aligned frames;
- 32 calibrated cameras per object except `026-sock-cloth`, which had 36;
- four tactile sensors per object;
- 1,770 output files; and
- 1,890,966,840 output bytes.

The first dry execution and the clean committed execution produced identical
complete output-tree SHA-256 values for all ten objects. This establishes byte
reproducibility for the completed upstream stages under the pinned environment.

## Ordering correction

The finite-group calibration amendment at commit
`7ecd7d44bde7f3fe8abab4aa57a2469ada868d77` was merged after the calibration
preflight and payload download, but before any calibration score, provider
selection, contact-map fit, or confirmation access. Its statistical design
remains a valid pre-score freeze, but its
`calibration_payloads_opened=false` field cannot describe the global project
state at merge time.

The immutable correction artifact is
[`deform360_official_hub_visuotactile_v1_stage1_provenance.json`](../protocols/amendments/deform360_official_hub_visuotactile_v1_stage1_provenance.json).
The original method gates, cohort selection, and no-replacement rule were locked
before payload access and are unchanged.

## Next boundary

Before calibration values are inspected, the project must freeze:

1. the causal frame-window rule;
2. the exact camera/provider and physical-backbone construction;
3. the tactile/contact feature reduction and correlation groups;
4. the mapping candidates, bias controls, closure test, and regret guard; and
5. the independent-object calibration score and finite-group interval rule.

Only calibration objects may then run the remaining annotation and
Bayesian-PhysTwin stages. Confirmation payload access remains forbidden until
all eight required calibration artifact roles and the evidence-use ledger are
sealed in `Deform360CalibrationBundleV1`.

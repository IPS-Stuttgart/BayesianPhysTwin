# PokeFlex baseline-relative guard public paired v2

## Question

Does the frozen baseline-relative Bayesian state-update guard improve the
released PokeFlex Kinect checkpoint on a new public cohort, while returning the
checkpoint prediction exactly whenever a safe improvement is not supported?

The guard was developed after opening the source cohort and the public-paired
v1 cohort. Those results justify this test but cannot confirm it.

## Fresh cohort

The target-free selection audit fetched all repository refs, scanned every
reachable path and blob, and compared them with the 116 released PokeFlex take
identifiers. For each physical object with at least one never-referenced take,
it selected one take by a salted SHA-256 ordering. This yielded 12 takes from
12 objects. Six other objects were exhausted by prior history. Selection is
frozen in
`configs/sota/pokeflex_baseline_relative_guard_public_paired_v2_selection.json`;
replacement is forbidden.

Freshness means absent from all reachable Bayesian-PhysTwin repository history
at lock time. It does not mean that the public data have never been viewed by
any external researcher.

## Frozen method

For target frame `f`, the released checkpoint supplies the physical prior. The
candidate estimates a weak action-local graph correction from Kinect frames
`f-5` through `f-1` and robot history through `f-1`. A source-calibrated model
admits that correction only when its feature vector lies inside source support
and its upper predicted regret bound is below zero.

Rejected, unsupported, malformed, or out-of-support updates return the released
checkpoint vertices byte-for-byte. The selected target mesh at `f` and all
future target meshes are forbidden prediction inputs.

## Custody

1. Stage only the causal RGB-D/robot inputs and one explicit upstream template
   mesh for every selected take.
2. Generate all 12 prediction archives and seals at one clean implementation
   revision.
3. Validate the complete, checksum-bound prediction barrier.
4. Only then open target meshes and score once.

The dataset remains on `gpuserver6000`, while GPU prediction runs on
`gpuserver4090`. Bulk artifacts move directly over the servers' shared LAN;
the jump host is used only to initiate administrative SSH sessions and is not
part of the data path.

### Pre-seal schema amendment

The first target-free smoke prediction stopped before producing a prediction
archive or seal because the initial stager paired `volucam` calibration with
`realsense` depth. The frozen checkpoint interface instead consumes the
released `kinect` depth and its matching `kinect` calibration. The stager now
copies that exact sensor pair and rejects calibration without 3-by-3 depth
intrinsics or 4-by-4 depth extrinsics. No target mesh was opened, and all 12
predictions must be generated at one later clean implementation revision.

## Primary gates

The object-balanced paired comparison passes only if all conditions hold:

- mean relative `CD_UL1` improvement is positive;
- the 97.5% object-bootstrap upper bound on candidate-minus-baseline error is
  below zero;
- no object regresses;
- at least 10 of 12 objects improve; and
- at least 10 of 12 objects contain one or more admitted updates.

The breadth counts preserve the preregistered 80% requirement after the
all-history audit left only 12 eligible objects. The published 6.498 mm Kinect
value is non-gating cross-split context. Jaccard remains diagnostic.

## Claim boundary

A pass supports strict improvement over the released PokeFlex checkpoint on
this registered fresh-take public cohort. It does not reproduce the unavailable
internal PokeFlex split and is not an exact comparison with the paper's 6.498 mm
table entry. A failure remains informative: exact fallback tests whether the
guard can avoid harmful updates, while the registered accounting preserves all
unsupported or technically failed takes.

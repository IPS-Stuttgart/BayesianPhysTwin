# Deform360 cross-view guard: post-open result

## Status

This is post-open mechanism evidence on 7 already-open source cases and 4
already-open calibration cases. All camera supplements and guarded predictions
were checksummed before their existing outcomes were joined. No reserved
prospective target object, target media, or target future was accessed.

The experiment does **not** justify a fresh preregistered camera-only run.

## Question

Can disjoint camera subsets validate a PhysTwin-constrained state update and
reject the coherent camera failures seen in the prospective v2 experiment?

Two frozen variants were tested with the same eight-camera causal prefixes:

1. **Split triangulation:** triangulate independently within two four-camera
   halves, build one state proposal per half, and require bidirectional held-out
   reprojection improvement and correction agreement.
2. **Direct 2-D fit:** fit coefficients in the causal physical-response basis
   directly from pixel residuals. Per-camera constant offsets are centered out,
   each camera receives fixed total information mass, and the same bidirectional
   validation and agreement gate is applied.

Both variants preserve the selected baseline byte-for-byte on rejection.

## Result

| Variant | Source accepts | Calibration accepts | Source identity change | Source Chamfer change | Calibration identity change | Calibration Chamfer change |
|---|---:|---:|---:|---:|---:|---:|
| Split triangulation | 0/21 | 0/12 | 0.000 mm | 0.000 mm | 0.000 mm | 0.000 mm |
| Direct 2-D fit | 1/21 | 5/12 | +0.159 mm (+1.01%) | +0.100 mm (+0.68%) | +3.107 mm (+139.40%) | +2.071 mm (+133.29%) |

The split-triangulation path was unsupported rather than discriminative. Across
66 half-update fits, the number of centers retaining at least three inlier
cameras was 0 for 43 fits, 1 for 9, 2 for 10, 3 for 3, and 4 for 1. No interval
produced proposals in both halves.

The direct 2-D path had enough support, but all six accepted intervals were
harmful on at least one primary metric. This includes a particularly clean
counterexample on `170-spider-ep0000`: the two held-out directions improved
reprojection loss by 96.4% and 95.0%, the inferred correction fields had cosine
agreement 0.996, and their disagreement ratio was only 0.107. Nevertheless,
the future identity and Chamfer errors worsened by 3.823 mm and 2.399 mm.

The three accepted `078-fishing-line-ep0004` updates regressed future identity
error by 8.076-11.317 mm despite bidirectional held-out camera improvement.

## Interpretation

Disjoint cameras are not an independent modality for this failure mode. A
coherent correspondence/depth bias can transfer across camera subsets and can
be strongly consistent in current-frame reprojection while being wrong about
future material motion. This is the empirical counterpart of the ambiguity

```text
y = d + b + e,
```

where true deformation `d` and a coherent observation bias `b` are not
identifiable from camera observations alone.

The result rules out two attractive camera-only guards:

- minimum multiview redundancy plus split triangulation is too sparse after a
  four/four split;
- direct held-out-camera validation is not sufficient under shared bias.

It does not rule out state updates anchored by an independent metric modality.

## Decision

Stop the open-ended camera-stitching agenda. Keep Prob4D and camera tracking as
a versioned observation feeder, with exact baseline fallback, but do not launch
another camera-only prospective cohort.

The next admissible update experiment must include an independent causal anchor
such as current-frame metric depth, LiDAR, tactile/contact evidence, or measured
actuator constraints. The fastest available test is a sparse current-frame
depth anchor at the tracked query pixels on these already-open cases, followed
by a genuinely fresh-object protocol only if it rejects the known camera-only
failures while retaining source gains.

## Reproduction

The implementation commits are `1a55ec8` (split triangulation) and `a1c39f2`
(direct 2-D fitting). The exact server artifact root is:

```text
/mnt/corsair/florianpfaff/deform360-crossview-guard-v1-postopen
```

The archived result summaries are in
`results/sota/deform360_crossview_guard_postopen_v1/`. Their internal result
hashes are:

- split triangulation: `dd65982b4804f52c5359e1b2d32dfeffd2110ccf5e7b60c3682fc81ca5fb259d`;
- direct 2-D: `6769230ad35dba55b66c3952c376ebb6e528260b2fb5fad4e0cf23fc85bc2d58`.

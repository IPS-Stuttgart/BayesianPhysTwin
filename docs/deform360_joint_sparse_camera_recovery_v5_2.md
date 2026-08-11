# Deform360 joint-sparse camera recovery v5.2

## Scope

This is an additive, source-only amendment to the public Deform360 v5
experiment. Deform360 is already a real-world dataset: its released RGB,
camera calibration, robot state, and robot action streams are physical
measurements. This route requires no new recording and no human approval.
Every admission decision below is deterministic and content-addressed.

The v5.1 prediction batch remains immutable. Its forecasts were sealed before
the development suffix was opened, but one camera-local metric-gauge failure
was handled as an object-wide technical failure. Five objects therefore fell
back before their predictions could be meaningfully tested. No suffix,
confirmation payload, or target outcome was used to diagnose this execution
granularity defect.

## Frozen recovery rule

1. Audit each originally scheduled camera independently with the unchanged
   32-pixel cluster definition and eight-cluster metric-gauge threshold.
2. Retain every passing camera and record every rejected camera. Unexpected
   parser, checksum, or contract failures abort; only the two registered
   metric-gauge support failures are camera-local rejections.
3. Recovery is triggered only when fewer than two original cameras pass.
4. Candidate cameras are all released, non-reserved cameras not already
   attempted. Rank them using prefix robot geometry only, in this order:
   maximum independent-cluster count, number of qualifying causal frames,
   total projected-point count, and camera ID.
5. Run the unchanged pinned MotionCrafter/Prob4D provider for at most the first
   four ranked candidates. Audit those streams with the same gauge.
6. Form a replacement prediction batch in a new evidence root using all and
   only passing cameras. At least two independently passing cameras are
   required; otherwise the object uses exact `B0` physical fallback.

The camera threshold is not relaxed, a one-camera update is never admitted,
and the original v5.1 batch is neither overwritten nor reinterpreted. The new
batch must be completely sealed before any development suffix is opened.

### Source-independent v5.2.1 contract correction

The metric-prefix contract explicitly represents a camera with no projected
robot taxels as a valid, checksummed artifact with zero support. Such a camera
is deterministically ineligible and is never sent to the visual provider. The
original implementation rejected this representable state before the frozen
ranker could apply that rule, despite already allowing zero per-frame counts.
The correction changes neither the candidate roster nor any ranking, cluster,
camera-count, fallback, or outcome boundary. Positive-support v1 artifacts
remain valid and behavior-compatible.

The production integrity verifier counts every declared provider product: one
member for each baseline product and one member for each independently decoded
overlap window. The recovery merge derives its expected member count from that
frozen product roster. It does not mistake the two baseline archives for
unexpected evidence or validate only the overlap-window subset.

## Executable custody sequence

The registered commands are separated at the suffix boundary:

1. Run `audit-base`, then `rank-recovery`, then `build-provider-plan` with
   `materialize_deform360_joint_sparse_camera_recovery_v5_2.py`. These steps
   audit the original cameras, rank extra cameras, and freeze the recovery
   provider plan.
2. `run_deform360_joint_sparse_motioncrafter_source_v5.py` generates only the
   selected additional prefix providers. The recovery invocation must supply
   the base provider plan, recovery preflight, and amendment files. Parallel
   shards are allowed, but each frozen job may execute exactly once.
3. Run `merge-provider-runs` to reconstruct the frozen job order and reject a
   missing, duplicated, reordered, or lineage-inconsistent shard. Export each
   completed job with `export_prob4d_uniform.py`; fixed Prob4D/VGGT blending is
   forbidden.
4. Run `build-combined-audit-plan` and then `audit-base` on that generated plan.
   This deterministically joins the base and recovered camera artifacts and
   applies the unchanged gauge independently to every attempted camera.
5. Run `build-recovery-lineage` to validate and hash all nine named recovery
   artifacts, including the immutable base prediction batch and receipt.
   `materialize_deform360_joint_sparse_source_plan_v5_2.py` then consumes the
   combined camera-audit plan directly and combines its unchanged physical
   inputs with the final per-camera audit and generated lineage record.
6. `run_deform360_joint_sparse_source_predictions_v5_2.py` seals a new set of
   exactly 100 outcome-free forecasts and its receipt.
7. Only after that receipt validates may
   `materialize_deform360_joint_sparse_source_endpoint_plan_v5_2.py` name the
   development endpoint files and
   `score_deform360_joint_sparse_source_v5_2.py` read them.

There is no operator-choice field in these contracts. Human review is neither
required nor accepted as an admission signal; the public file hashes, fixed
camera ranking, fixed gauge threshold, and exact fallback determine the path.

## Claim boundary

The amendment repairs source-prefix camera failure accounting. It does not by
itself show predictive benefit, calibration, confirmation, safety, Causal4D
benefit, or state-of-the-art performance. A larger public evaluation remains
authorized only if the unchanged registered source gate passes on the newly
sealed batch.

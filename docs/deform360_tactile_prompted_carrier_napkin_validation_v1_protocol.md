# Deform360 tactile-prompted carrier independent validation protocol

## Question

Can the tactile-prompted, two-view, shared-bias-aware carrier developed on the
already-open sock case produce a valid observation carrier on one independent
calibration object without changing its policy or consulting prediction
scores?

This is an admission test for an observation artifact. It is not yet a test
of a Bayesian-PhysTwin state update or future-prediction accuracy.

## Source selection

The development object is `026-sock-cloth`, a bimanual sheet interaction. The
independent object was selected from the existing locked calibration list by
the following metadata-only rule:

> Choose the first still-unopened calibration object after the development
> object, in the same stratum, whose locked source episode is bimanual.

`031-cotton-cloth` is first but unimanual, which is incompatible with the
frozen two-gripper assignment mixture. The next candidate is therefore
`036-napkin-cloth`, source episode 9 and processed episode 0. This decision
used only manifest-bound metadata; no image, tactile array, MotionCrafter
array, or prediction score from either candidate was inspected.

The permitted prefix is source frames `[78,120)`, with registered contact
beginning at frame 114. Frames `[120,144)` remain untouched future. The fixed
camera panel is:

- `brics-odroid-010_cam0`;
- `brics-odroid-019_cam1`;
- `brics-odroid-022_cam1`.

## Frozen stages

The source-only pipeline is staged, and each failure terminates in exact
fallback:

1. Recover the two-gripper causal robot prefix using all three calibrated
   cameras. The existing robot quality gate must pass unchanged.
2. Preserve direct and swapped tactile-to-gripper assignments at equal prior
   mass. The existing tactile geometry gate must pass unchanged.
3. Fit the metric gauge independently for both assignments. Both assignments
   and all three cameras must pass the frozen held-prefix errors.
4. Use tactile object-side and robot-side prompts to select SAM2 automatic
   masks without a PhysTwin innovation.
5. Admit a camera pair only with at least five mutual fixed-block matches and
   p90 cross-view distance at most 20 mm.
6. Use one camera for the carrier mean. The second camera may only preserve or
   increase covariance through local disagreement and a shared-bias term.
7. Represent 128 backend nodes while retaining fixed `8x8` information-cluster
   identities; repeated nodes do not become independent evidence.
8. Keep each failed tactile assignment as an exact baseline fallback with its
   original prior mass.

The complete gates and policies are content-addressed in
`protocols/locks/deform360_tactile_prompted_carrier_napkin_validation_v1.json`.
The first operational stage is separately locked in
`protocols/locks/deform360_official_hub_causal_robot_prefix_napkin_validation_v1.json`.

## MotionCrafter provenance

This protocol consumes the already generated Stage 1 v6 provider bundle; it
does not rerun MotionCrafter. That bundle was produced under Torch
`2.13.0+cu130` with the base virtual-environment CUDA-13 libraries preceding
the system-site overlay in `LD_LIBRARY_PATH`. Invoking the virtual-environment
Python without that library composition is not an equivalent runtime.

The protocol binds:

- provider job manifest ID
  `9726e7ae12d442956ff81376fe52cdc2f8360fdcd3e5cccbc12543ca584b30f9`;
- provider run-report SHA-256
  `db94d78c9b5acd2c1290976f1ff9647c525df0bad7ab62f4621175ec0fc75383`;
- all three job IDs and video hashes;
- every consumed NPZ member through its generated prediction manifest.

## Information boundary

After both locks are pushed, the causal camera and tactile prefix and the
bound provider arrays may be opened only to execute these frozen gates.
Calibration prediction scores remain unopened. Confirmation payloads, target
outcomes, future frames, and all held-v8 artifacts remain prohibited.

An admitted carrier permits only the next preregistered state-update step. It
does not itself establish an accuracy gain or a state-of-the-art result.

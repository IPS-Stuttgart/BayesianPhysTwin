# Causal4D released visual-input artifact API v2

`bayesian_phystwin.causal4d_artifacts_v2` is the narrow boundary for released
PhysTwin visual inputs that Causal4D uses to prepare MolmoMotion queries.

The v1 artifact facade verifies the legacy `final_data.pkl` before loading the
released raw-track correspondence. It cannot by itself prove the identity of the
other files opened by the released preprocessing stack. API v2 closes that gap.

## Required identities

The caller must supply independently trusted SHA-256 digests for:

- `final_data.pkl`;
- `metadata.json`;
- `pcd/0.npz`;
- `calibrate.pkl`;
- every archive under `cotracker/*.npz`.

The CoTracker digest inventory must match the directory exactly. Missing,
unexpected, renamed, or differently ordered archives are rejected.

## Loading contract

`load_released_phystwin_visual_inputs(...)`:

1. discovers the released CoTracker archives without opening their payloads;
2. verifies every declared byte identity before any pickle, JSON, or NPZ payload
   is opened;
3. loads the two legacy pickle inputs only through the digest-bound trusted
   loader;
4. executes the released one-to-one raw-track correspondence implementation;
5. verifies every input digest again after loading, detecting in-place mutation
   during the operation;
6. checks that the correspondence's processed points and visibility equal the
   trusted final-data payload; and
7. returns `ReleasedPhysTwinVisualInputsV2`, whose NumPy arrays are defensive,
   read-only copies.

The returned artifact contains processed object observations, raw tracks and
visibility, source camera/track identities, source world points, initial-match
distances, intrinsics, camera-to-world calibration, image dimensions, frame
rate, mapping tolerance, every input digest, and a deterministic `artifact_id`.

## Security and evidence boundary

Digest verification establishes byte identity against a separately trusted
manifest. It does not sandbox Python pickle. Digests supplied alongside
otherwise untrusted pickle files are not sufficient.

This contract proves which released visual inputs were consumed. It does not
prove MolmoMotion accuracy, observation calibration, or physical-prediction
improvement. New artifacts should continue to use JSON/NPZ rather than pickle.

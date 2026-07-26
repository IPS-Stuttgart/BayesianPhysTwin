# PhysTwin Photometric Graph Prefix Source v1

## Purpose

The strongest open Bayesian-PhysTwin headroom result uses manual material
identities during an allowed response prefix. Automatic camera-only identity
channels have not transferred reliably: CoTracker3 remains well above the
manual ceiling, and a source-only static-background gauge improves background
tracks while worsening moving object identities. This smoke tests a different
observation path that does not first solve generic cross-frame identity.

PhysTwin already carries material appearance as Gaussian primitives attached
to its reconstructed object. The proposed adapter skins those frame-zero
Gaussians with the selected physical graph trajectory, renders the three
released cameras, and computes finite-difference image responses to a
low-rank graph-position field. Early prefix RGB fits the field; later prefix
RGB decides whether it transfers.

This first stage is a **photometric graph-position competence test**, not yet
a dynamically propagated state update. A pass only admits a separately
frozen official-Warp rerun that injects positions and estimated velocities.

## Frozen Smoke

- Case: `single_lift_cloth`, an already-open source interaction.
- Allowed images: frames 114 through 120, all before the exclusive training
  endpoint at frame 121.
- Fit frames: 114 through 117.
- Held-out prefix gate: frames 118 through 120.
- Cameras: `cam0`, `cam1`, and `cam2`.
- Appearance: the released frame-zero PhysTwin Gaussian splat.
- Physical carrier: the selected raw MatPhys/PhysTwin trajectory.
- Material attachment: fixed inverse-distance weights to the 16 nearest
  frame-zero graph nodes.
- Parameterization: rank-4 graph modes, independently displaced along world
  x, y, and z, for 12 coefficients total.
- Finite-difference scale: 5 mm maximum displacement per mode-coordinate
  direction.
- Rendering: 4x spatial downsampling, object mask eroded by two rendered
  pixels, and rendered alpha at least 0.1.

The graph update must lower robust, exposure-profiled held-out prefix RGB RMSE
by both at least 2% and at least 0.0005 in normalized RGB units. Rejection
returns exact all-zero graph weights.

## Causal And Calibration Boundaries

The object mask and rendered alpha determine support before the PhysTwin image
innovation is evaluated. The innovation is used once, by a robust likelihood
and the held-out prefix score. It is not recycled into prior reliability.

Dense pixels are not counted as independent evidence. Each frame has fixed
total information, divided equally across usable cameras and occupied 8x8
spatial blocks. Duplicating a camera or a correlated pixel block therefore
cannot create unbounded confidence. Per-frame, per-camera, per-channel affine
color terms absorb exposure and white-balance nuisance. The posterior
covariance is conditional on this deliberately conservative clustered model;
this one-case smoke cannot establish calibration.

No RGB, mask, manual trajectory, point cloud, or metric at or after frame 121
may be read. No held-v8 artifact may be opened or touched. All source hashes
must replace the prelock placeholders before the renderer runs.

## Prediction Lock

The implementation was frozen at commit
`edcd4f2f8e4b3a36e98ac3e480ac49132959ddb2`. The solver and remote renderer
are bound by SHA-256
`234e4a969894d2e5db4f32af17dc194ea83af0e936c74da01aecfd1929dbb57e`
and
`f0584616ffb32cb7df520be4d7f2ec83da4101262b46b05aca65b0e7ec77b2c0`,
respectively.

The physical trajectory, graph basis, Gaussian PLY, and camera JSON are bound
by SHA-256
`5e41ce3bfea780add79c20841084422ad7cad5e6e2443f3c2d2fca9729b8dd72`,
`e6d1d123efa934806aab9a5bf1d1ffcd6af72502b849248c852c59ce4ac50222`,
`4f7239de13961ce01edea3dc3ae2fee305329a83b9d4b4b732991f1e86a99fcf`,
and
`99b80635489980b814f9abd4fbc663a2677df2b13ae901f9512a6e2901f8e970`.
The sorted 42-file prefix RGB/mask manifest has SHA-256
`6a27b59c8ae8cd85156d93561adddd1a1541dd59fad08ebd03e22ff232425329`.
The protocol status is `locked-before-prefix-render`.

## Decision

A failure closes this exact Gaussian attachment, translation-skinning, rank-4
parameterization without tuning it on the observed result. A pass permits only
this next sequence:

1. seal the prefix selection;
2. lock a nonlinear official-Warp state-injection and future scoring protocol;
3. generate future predictions without future object observations;
4. seal predictions before opening already-public source evaluation metrics;
5. proceed to an object-disjoint source panel only if future track, Chamfer,
   late-horizon, and fallback gates pass.

Even a source-panel pass would not be an independent state-of-the-art result.
A genuinely fresh preregistered object cohort would still be required.

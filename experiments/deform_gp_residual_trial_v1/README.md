# Fixed nonlinear GP residual trial on DLO2 development data

This experiment tests whether a nonlinear residual model improves the existing
local ridge correction. It uses recorded trajectories only: no robot, probing,
new observations, simulator retraining, or changes to official protocols.

## Comparison

The baseline is the retained 6,400-update **DEFORM physics-plus-GCN hybrid**, not
bare rod physics. Both residual models use the existing action-local feature
builder and the same 40 historical fitting trajectories. Validation uses the
original eight validation trajectories over the full 498-step forecast. The
separate eight source-test trajectories and official evaluation are excluded.

The comparator is the existing nodewise local ridge predictor, ridge 1 and
shrinkage 0.25. The candidate adds a Matérn-3/2 Nyström feature block with 128
trajectory-balanced anchors, amplitude 3, length multiplier 1 (length equals
the square root of feature count in standardized coordinates), ridge 1,
seed 20260907, and shrinkage 0.25. All fitting targets are retained; only the
kernel basis is approximated. There is one fixed GP candidate and no GP
hyperparameter selection using validation outcomes.

The model is exact Gaussian inference for that finite-rank kernel, not a
variational GP. Nodes and coordinates are modeled separately. The optional
model-based covariance and trajectory-cluster sandwich covariance are distinct;
this accuracy trial does not establish calibration or graph-coupling value.
A gain in the GP mean is also a kernel-regression gain, not uniquely Bayesian
accuracy evidence.

## Native identities and data access

- Shared code: `9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d`.
- Training result SHA256: `1f8d092bc38b03f6cdd68ef38abcb7d403d914e38ba483698579deaeea8c2572`.
- Manifest SHA256: `7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98`.
- Checkpoint SHA256: `b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65`.
- Upstream DEFORM commit: `b73b8b8ecc033caefa693fab7898741d4e6dbeff`.

The exact retained manifest has a `trajectories` mapping. Upstream metadata
belongs to the training record, not the manifest. The recorded selected-checkpoint
validation L1 is `0.007912029745057225 m`; the trial must reproduce it within
`1e-6 m` before reporting a candidate comparison.

The current workstation1 interpreter is
`/home/florianpfaff/source-only/deform-bayesian-v1/venv/bin/python`, and its
upstream root is
`/home/florianpfaff/source-only/deform-bayesian-v1/DEFORM-b73b8b8`.
These were checked through runtime inventory and the downloaded, SHA256-verified
metadata artifact `10047382790` from Actions run `34205038782`.

Each allowed trajectory is checked against its manifest hash by the native
loader. A Python audit guard rejects source-test access, unlisted dataset paths,
and dataset writes. This is not an operating-system sandbox. The native
predictor receives two initial states and known future clamped-node motion;
future free-node values are fitting/scoring targets, not GP features.

## Checks and output

The zero-nonlinearity candidate must match the actual repository ridge mean
within `1e-9 m`. Clamped coordinates must remain exactly equal to the baseline.
Predictions are sealed before scoring. Outputs retain per-trajectory and
per-horizon errors, model arrays, prediction arrays, metadata, and any technical
failure. Technical failures are not scientific negatives.

The request-file workflow executes on `gpuserver4090`. Numerical tests run in
the retained runtime before the empirical comparison. Historical development
results are not fresh confirmation or evidence of arbitrary-object transfer.
The current draft PR is #967. Read the retained result artifact and result note
for the outcome rather than interpreting a green CI badge as improved accuracy.

# Cross-Branch Sparse-State Pilot

## Question

Can eight causal main-branch point observations improve a released native DEFT
forecast on two unobserved child branches, beyond an equally informed readout
correction? This tests whether the successful DEFORM state-update construction
has useful headroom on a different topology and native backend. It does not
claim that DEFT, harmonic interpolation, or a state reset is itself new.

The native adapter already passed six source-independent checks, including
bitwise monolithic/segmented/zero-update parity, at `9d21fdbf`. The result file is
bound by the new protocol. Existing DEFORM code and evidence stay unchanged.

## Frozen Design

Use exactly the previously declared first lexicographic robot-actuated BDLO1
training file, one 500-frame block from one original recording. No other block
is selected if this one is difficult. The released full checkpoint has training
exposure to this split; this is a capacity pilot, not independent confirmation
or a paper-metric reproduction. No significance test is justified at n=1.

All arms share two initial full states and prescribed future positions of the
four clamp points. The extra budget is parent nodes 2, 4, 6, 8 at prediction
frames 41 and 49, corresponding to raw frames 43 and 51. Forecasts score raw
frames [52,172). No child identity after initialization enters an update.

The primary arm transfers the existing gain-one DEFORM construction without
tuning: interpolate the two parent residuals with zero clamp correction,
extend each junction residual constantly down its child branch, and infer the
velocity increment from the residual slope over 80 ms. Apply those pose and
velocity increments to a private physical shadow of DEFT. That shadow retains
all released physical parameters and junction constraints but sets its learned
residual weight to zero. The candidate is the unchanged full-model forecast
plus the difference between updated and unchanged physical-shadow continuations.
This preserves the stronger full-model mean and gives an exact-zero fallback.

Controls are unchanged full DEFT, the physical-only shadow, matched persistent
and linear-velocity readout corrections, pose-only physical propagation,
parent-only propagation (moving duplicate roots but not free child identities),
and a direct state update inside full DEFT. Secondary arms cannot rescue the
primary gate. The parent-only arm helps distinguish graph interpolation from
subsequent cross-branch physical transport; no mechanistic conclusion is assumed.

The primary must lower RMSE by at least 5% against unchanged full DEFT and both
readout controls on **each** child branch, with non-increased coordinate L1 and
late RMSE on each. Report FDE and the equal-child-branch aggregate. Exclude
duplicate junction identities and padded vertices from all child scores.

## Information and Failure Boundaries

Freeze code before the restricted numeric stager decodes the source pickle.
The stager only publishes initial states, clamp streams, and eight parent
measurements. It necessarily decodes the source container, but does not compute
future residuals, choose a window based on motion, or expose future child values
to the predictor. The scorer independently reopens the training trajectory only
after all eight prediction arrays and their input/source digests are sealed.

Reject incomplete/nonfinite predictions and missing source data; do not replace
the recording. Retain any failure. Do not tune the gain, window, observation
identities, or physical parameters after scoring. No automatic broader study is
authorized even if this pilot passes. No public evaluation/test split, protected
DEFORM DLO3 evaluation or DLO4/DLO5, held-v8, Deform360 target, physical recording,
or Causal4D target is used. Evidence remains local/private-paper only.

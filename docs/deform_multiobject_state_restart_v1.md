# Matched Multi-Object Native-State Coupling

This is a new, exploratory transfer experiment, frozen after the positive DLO2
development result and before computing the new DLO1/DLO3 coupling outcomes.
It does not reopen or alter any earlier gate or successful DEFORM result.

## Question

Does the fixed coupling

`incumbent readout + updated physical continuation - unchanged physical continuation`

transfer to other rods without retraining, gain selection, or additional sensing?
The physical update preserves instantaneous velocity, twist, material frame, and
previous positions. It adds an interpolated position-residual increment and the
two-time residual slope to the physical endpoint. It does not replace simulator
velocity with an eight-frame average or double-count the learned readout.

## Objects and Inputs

- DLO1: all eight already-open source-test trajectories, frozen update-6400
  physical checkpoint and selected local readout (ridge 1, shrinkage 0.5).
- DLO2: the fourteen previously opened development/evaluation trajectories,
  excluding the pre-existing design trajectory `103.pkl` from every aggregate.
  Its fixed all-train checkpoint/readout is the discovery reference, not transfer.
- DLO3: all eight already-open source-test trajectories and the originally
  registered primary seed 42, not a seed chosen by its performance. The official
  DLO3 evaluation partition is not accessed.

There are 30 predicted trajectories and 29 analyzed trajectories, of which 16
belong to the two method-transfer objects. This is not an unseen-object-trained
model: each incumbent was previously fitted on that object's training data. It
tests transfer of the update rule, not zero-shot transfer of physical parameters.
All three data partitions were already opened historically, so even DLO1/DLO3
are exploratory method-transfer evidence, not a fresh confirmatory test.

The protocol binds exact source archives, checkpoints, manifests, and rosters.
No optimizer is run. No seed, checkpoint, readout, gain, or observation identity
is selected from the new comparison. DLO4/DLO5, held-v8, fresh Deform360 targets,
and physical Causal4D acquisition remain untouched.

## Matched Sensing

Every updated arm receives exactly eight three-dimensional prefix observations:
nodes 2, 4, 6, 8 at archive frames 41 and 49. Forecasting starts at frame 50 and
ends before frame 170, at 100 Hz. Archive frame 0 is raw dataset frame 2. Scored
hidden nodes 3, 5, 7, 9 never supply observations. Four end nodes are clamped;
their future motion is the known action input under the existing DEFORM contract.
Future free-node positions never enter any prediction function.

DLO1 has thirteen nodes rather than twelve. The primary comparison keeps exactly
the same four observed and four hidden material indices; node 10 remains part of
the dynamics but is not in the primary metric. A separately labeled all-hidden
analysis includes node 10 and cannot change the primary gate. This avoids silently
claiming that the matched four-node score covers the whole DLO1 free span.

The observations are released material identities, not an automatic camera or
tracker provider. The claim is controlled sparse-observation value, not deployed
perception competence. Whole public trajectory containers are decoded, but only
two initial states, the two permitted sparse observations, and clamped actions
are routed to inference; withheld future free-node truth is used only by scoring.

## Frozen Arms and Noise

Clean observations compare the unchanged incumbent, raw physical rollout,
persistent sparse readout correction, raw physical pose and pose/velocity resets,
paired pose-only update, paired pose/velocity update, and its quarter-gain version.
The primary gain is 1; the secondary gain is 0.25, exactly as in DLO2 development.
The secondary cannot rescue a failed primary transfer gate.

The incumbent, matched readout, full-gain coupling, and quarter-gain coupling are
also evaluated under the two previously frozen synthetic measurement conditions:
1 mm independent coordinate noise; and the same noise plus one 5 mm standard
deviation translation shared by all eight measurements of a trajectory. Sixteen
repetitions are fixed. Noise seeds are domain-separated by object; DLO2 retains
its original seed sequence for reproduction. These are stress tests, not measured
sensor calibration or robustness to arbitrary coherent bias.

## Controls and Evidence Order

Before any new comparison, each object must pass native-adapter versus legacy-CPU
parity, the existing archived-GPU tolerance, byte-identical zero-update continuation,
original-object return for zero paired readout, and known-perturbation recovery.
All clean and noisy forecasts for all three objects are then sealed, with a
complete global prediction barrier, before scoring. Technical failures are
retained, never silently replaced or omitted. A failed control stops the run;
there is no automatic same-method retry or post-outcome tolerance adjustment.

## Analysis and Interpretation

Report coordinate L1, Euclidean point RMSE, FDE, three equal horizon bins, per-case
wins and worst-case regressions, and per-object results. Average noise repetitions
within a trajectory before aggregation or resampling. Use 10,000 paired whole-
trajectory bootstrap samples per object. These intervals are conditional on the
opened object and are not evidence from thousands of independent point-frames.
Across objects, weight object means equally and show transfer-only DLO1/DLO3
separately from the three-object aggregate containing discovery DLO2. Two new
method-transfer objects do not support a population-level generalization claim.

The primary transfer gate requires both L1 and RMSE improvement, superiority to
the matched readout on both metrics, non-increasing late-horizon RMSE, and at
least five of eight joint trajectory wins on **each** of DLO1 and DLO3. Noise,
quarter-gain results, FDE, and the all-hidden analysis are reported separately;
none selects or repairs the primary method after seeing outcomes.

Outcome artifacts stay in a local/private evidence archive, respecting DLO3's
publication boundary. This experiment alone cannot establish SOTA, calibrated
uncertainty, robotic closed-loop performance, or counterfactual identification.

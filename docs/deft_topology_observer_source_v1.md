# Topology-Supported Sparse State: Source Test

## Scientific Question

Does the same eight-observation paired physical update work across a branched
topology when the observations directly constrain each child, rather than asking
parent observations to supply an unmeasured child correction? A positive result
would extend the existing DEFORM evidence to a second native hybrid simulator
and a different topology under matched observation budgets. Backend support or
synthetic recovery alone is not that contribution.

The failed parent-only pilot remains frozen at `04cf109c`. Its recording and all
other blocks from that recording are excluded here. No failed gain, window, cap,
parameter, or calibration rule is retuned. The new information is a measured
point on each child. These known material-point measurements come from the public
dataset, not a new recording or an automatic image tracker.

## Method and Controls

Retain the native DEFT checkpoint, the qualified CPU compatibility runtime, two
initial full states, future four-point clamp trajectory, prefix/future windows,
gain one, and residual-slope velocity rule. Only the observation placement and
its corresponding spatial interpolation change. Observe the two parent
junctions and the two child tips at raw frames 43 and 51, eight point observations
per corrected arm. Parent residuals interpolate linearly through the two
junctions with zero clamped endpoints. Each child interpolates between its
measured junction and measured tip. Duplicate roots remain equal; padding and
clamps receive exactly zero correction.

The primary is unchanged native full prediction plus the difference between
updated and unchanged physical-shadow continuations. The shadow retains all
released physical parameters and constraints; only its learned residual weight
is zero. It is a private model instance, not a modification of the incumbent.

Matched controls use exactly those same eight topology observations: persistent
readout, linear-velocity readout, pose-only physical transport, direct full-model
state injection, and periodic native position corrections at the two arrival
times. Periodic correction is a matched control, not a reproduction of a
published state observer. The old parent-only paired update is also run with its
own unchanged eight-observation placement. The stager therefore emits 12 unique
point observations for the two policies, but no arm may use their union. An
invariance test checks that changing the parent-control-only channel cannot
change any topology arm, and conversely.

Within the declared four-scalar-coefficient interpolation space, the parent-only
observation operator has rank two and the topology operator has rank four. The
synthetic check verifies this and exact pose/velocity recovery. This is elementary
linear algebra, not a new observability theorem, and not a claim that the full
nonlinear rod state, twist, or internal forces are identified. No Gaussian or
calibration claim is attached to the deterministic point-update experiment.

## Prospective Source Roster

Use the first block of the first three lexicographic robot-actuated BDLO1
training recording IDs after excluding the opened track1244 recording. These
are three original recordings of one physical object, not three new objects.
The exact filenames and Git/SHA-256 identities are bound in the JSON lock before
decoding. The released checkpoint has training exposure; this is source capacity
and transfer-between-recordings evidence, not independent model generalization.

The source protocol and independent metric implementation must be committed
before the restricted numeric stager decodes a trajectory. The stager publishes
only the two initial states, clamp stream, and permitted observations. Every
case and every arm is sealed before the scorer reopens any future free-node
values. No difficult recording is replaced and no empirical retry is allowed.

Score only identities disjoint from both observation placements: five parent,
three child-1, and two child-2 identities. In particular, the newly observed tips
are excluded from future scores. Duplicate roots and padded points are excluded.
Report each recording and branch, then equal-recording averages. The primary
aggregate first weights both child branches equally within a recording.

For each child, the primary must reduce average RMSE by at least 5% versus native
DEFT, both readout controls, periodic native correction, and parent-only paired
correction, without increasing coordinate L1 or late RMSE. It must jointly win
RMSE/L1 on at least two of the three recordings against every comparator. No
recording may have more than 10% RMSE regression versus native DEFT. Secondary
arms cannot rescue a failed primary. With three recordings of one object, there
is no significance test or population-level confidence claim.

## Custody and Scope

The source qualification continues to use Python 3.10.12, Torch 2.0.1+cu118,
NumPy 1.24.3, Theseus 0.2.1, Numba 0.59.1, PyTorch3D 0.7.7, CPU/float64 and one
thread. It is the previously qualified compatibility runtime, not a claim to
reproduce the upstream paper's best number or recommended environment.

An initial partial-clone Git archive attempted a larger promisor pack and was
stopped before completion. Its partial archive is retained. The three declared
files were instead downloaded by exact raw URLs and hashed without decoding.
As with the prior source audit, no claim is made that no unselected public bytes
ever entered Git's object cache. No unselected trajectory content is inspected.

No public test/evaluation content, protected DEFORM DLO3 evaluation or DLO4/DLO5,
held-v8, Deform360 target, or physical Causal4D dataset is opened. Existing DEFORM
and all failed study implementations, protocols, and results remain unchanged.
Even a pass does not automatically authorize a target evaluation. Evidence stays
local/private-paper only.

Upstream: [DEFT code](https://github.com/roahmlab/DEFT) and
[DEFT paper](https://arxiv.org/abs/2502.15037).

# Frozen reference-centered DEFORM transport screen

## Question and boundary

The successful paired update measures a prefix innovation against the learned
incumbent readout, but propagates it about the raw native DEFORM trajectory.
This new source-only experiment asks whether transporting the same innovation
about the learned reference instead improves prediction. It changes neither the
incumbent, its checkpoint/readout, the observation budget, nor any prior result.
It is not a retry of weak-constraint inference, constraint-normal persistence,
or a failed DLO-Lab controller. No new material or statistical model is fitted.

Only the fourteen already-open DLO2 trajectories from the prior paired study
are predicted. The existing design case 103.pkl is excluded from all aggregates.
DLO1/DLO3 transfer, DLO4/DLO5, official DLO3 evaluation, Deform360 reserves,
held-v8, physical Causal4D, GPUs, and new recordings remain out of scope.
The experiment is local/private, with no push or main merge. A passing source
gate motivates, but does not authorize, a separately frozen transfer study.

## Fixed method

Let b_t be the frozen learned forecast and a_t its archived native component.
Let s_t be a fresh, parity-checked native replay, and r_t = b_t - a_t. These are
predictions computed before the sparse observation update, not future truth.
The lifted reference uses x_ref,t = s_t.x + r_t and
v_ref,t = s_t.v + (r_t - r_(t-1))/dt, with dt = 0.01 s.
Offsets at prescribed clamp nodes must be exactly zero.

At prefix endpoint 49, initialize two native states at that reference. Add the
unchanged interpolated pose and residual-slope velocity innovation to one.
Advance both with the same registered future clamp action. Return
`b_(t+1) + x_updated,(t+1) - x_nominal,(t+1)`.
Before each subsequent native step, translate both branches by the same amounts
to re-center the nominal branch at the next lifted reference position/velocity.
Each branch retains its own twist, material frame, and previous positions;
these are evolved by the unmodified native solver. Clamp positions are exact;
native clamp-velocity bookkeeping is left unchanged. The nominal/candidate
state difference is preserved by common translation up to floating arithmetic.

This is a defect-corrected, reference-centered error-transport construction,
not a new generic Bayesian theorem. The corrected arrays remain readouts;
neither the lifted reference nor the delivered mean is claimed to be exactly
inextensible, physically feasible, or an identified physical mechanism.
No UQ, planning, fresh-confirmation, or official SOTA claim is tested here.

## Matched arms and information

1. Unchanged incumbent.
2. The existing paired pose/velocity update, byte-reproduced from its frozen run.
3. Reference-initialized paired update, with no further re-centering (diagnostic).
4. Reference-centered paired update (the sole primary).

Every updated arm uses the same eight 3D material-point observations: nodes
2,4,6,8 at archive frames 41,49. Hidden nodes 3,5,7,9 supply no inference data.
The prefix ends at 49; score frames 50:170, split 50:90,90:130,130:170.
The two initial full states and prescribed end-node trajectories are identical
to the prior contract. Raw archive frame 0 corresponds to dataset frame 2.
Whole already-open source containers may be decoded; only two initial states,
the eight allowed measurements and clamp actions enter prediction. Future
free-node truth is loaded for metrics only after the complete prediction seal.
Unit tests poison all non-permitted truth entries to check this routing.

## Native controls and evidence order

Freeze clean implementation/source hashes, exact upstream revision, checkpoint,
parent prediction archive, source manifest/roster, CPU runtime, and this protocol
before native execution. One fresh root/attempt only. No retry, replacement,
post-score tuning, or secondary-arm promotion. All numerical/state arrays must
be finite. The previous incumbent and paired forecasts must reproduce exactly.
An exact-zero reference offset must reduce to the previous paired propagation;
an exact-zero observation innovation must return the original incumbent object.
All nominal/candidate traces and every primary/secondary prediction are sealed
before opening source future truth. Retain any technical failure separately;
it fails the whole source gate and cannot be counted as a successful forecast.

## Source gate

Compute trajectory-level coordinate L1, point RMSE and FDE on the four hidden
identities, then equal-trajectory averages over the thirteen non-design cases.
The primary must satisfy all of:

- At least 2% lower mean L1 and mean RMSE than the existing paired update.
- Non-increasing late-horizon mean RMSE.
- At least eight of thirteen joint L1/RMSE trajectory wins.
- No trajectory RMSE ratio above 1.05 relative to the existing paired update.
- A paired whole-trajectory bootstrap RMSE-difference 95% upper limit below zero
  (10,000 replicates, NumPy default_rng seed 260929, lexicographic case order).
- Every native control and all fourteen prediction seals pass, with zero
  technical failures, unsealable cases, omitted cases, or replacements.

All arms and horizon results are reported even on failure. The already-open
cases make this a development screen, not independent confirmation; bootstrap
intervals are conditional on this one opened object. A separate arithmetic
implementation checks metrics, readout and centering identities and the decision.
It is a second implementation, not independent human review.

# Constraint-separated readout: fixed DLO2 source screen

## Question

Does the successful eight-query paired DEFORM update improve further if its
increment is split using the native rod's local inextensibility constraints?
This is a new exploratory screen, not a rerun or retuning of a failed arm. The
earlier paired, weak-constraint, sensing, and uncertainty results stay immutable.
Only the same already-open 14 DLO2 trajectories are used; the original design
trajectory is excluded from the 13-trajectory mean. No DLO1/DLO3 transfer, DLO4/5,
official DLO3 evaluation, protected Deform360, held-v8, or physical Causal4D data
are accessed. No recording, GPU, training, or native replay is needed.

The author implementation at DEFORM commit
`b73b8b8ecc033caefa693fab7898741d4e6dbeff` applies inextensibility constraints after
each step and recomputes velocity from the corrected positions. This motivates a
test, not an attribution of the previous errors to constraints. The upstream
[implementation](https://github.com/roahmlab/DEFORM/blob/b73b8b8ecc033caefa693fab7898741d4e6dbeff/DEFORM_sim.py)
and checkpoint are not modified or redistributed.

## Fixed construction

Let `b_t` be the frozen incumbent, `d_t = paired_t - b_t` its already-sealed
paired native correction, and `r` the persistent sparse-prefix position offset.
All use the original eight measurements: identities 2/4/6/8 at frames 41/49.
The persistent offset is recovered from the frozen readout comparator; its
constancy is checked to 1e-12 m. This introduces no new observation.

For every predicted nominal native centerline `q_t`, form rows
`A_i dq = unit(q_{i+1}-q_i) dot (dq_{i+1}-dq_i)` with the prescribed clamps
excluded from the free coordinates. `P_t` is the Euclidean orthogonal projection
onto `null(A_t)`, computed by SVD at relative tolerance 1e-10. It uses the model's
future prediction, never observed future geometry. The primary readout is

```
b_t + P_t d_t + (I - P_t) r.
```

This is an incremental, linearized **readout construction**. It does not restart
a feasible native state, enforce nonlinear edge lengths, identify observation
bias, or prove a physical cause. The fixed world-frame offset is reprojected at
each predicted geometry. No mass or time metric is inferred. There is no gain,
rank, noise scale, or horizon fit. Projection is established mathematics, not
itself a novel contribution. A useful result would be evidence that this
particular decomposition improves sparse-information prediction.

## Controls and gate

Six arms are sealed together: unchanged incumbent; unchanged paired update;
unchanged persistent readout; a fixed 50/50 blend of the last two corrections;
tangent-only propagation (ablation); and the primary complementary split. The
ablation cannot rescue a failed primary arm. Geometry failure retains the paired
array exactly and counts as a technical fallback, never a dropped trajectory.

The primary must achieve at least 2% lower mean hidden-identity RMSE than **each**
of incumbent, paired, readout, and half-blend; lower coordinate L1 than each;
at least 9/13 joint L1/RMSE wins over paired; no higher late RMSE than incumbent
or paired; no trajectory RMSE above 1.05 times incumbent; and a negative upper
95% paired bootstrap bound against each control. All 14 predictions must be
ordinary. The four hidden identities are 3/5/7/9, forecast frames 50:170, with
fixed 40-frame early/middle/late blocks. No observed identity is scored.

Metrics average events within trajectory, then equally across the 13 analysis
trajectories. Bootstrap resamples whole trajectories (10,000 draws, seed 260913).
These are descriptive development intervals on a repeatedly opened object, not
fresh confirmation or an independent-object confidence statement. These point
metrics are not Chamfer distance. No uncertainty/calibration result is claimed.

## Custody and interpretation

Clean committed source, the exact parent artifacts, runtime, configuration, and
output root are bound before prediction. A write-once attempt marker is consumed
before reading any forecast input. Every prediction and exact comparator is
sealed before the separate scorer opens the source truth member. Frozen arrays
are read only. The prediction process has no source-truth loader. The scorer
validates the complete roster, shapes, array hashes, comparator bytes, and source
identity alignment. Synthetic tests cover projection, rigid equivariance,
complementarity, exact fallback, causal geometry, and fail-closed custody.

Failure closes this exact construction; no alternate rank/blend/threshold will
be selected from its results. A pass permits proposing a separately registered
native/transfer experiment, not automatically running it or claiming SOTA.

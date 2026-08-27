# Native DEFT Source Qualification

This is an isolated extension of the public-data sparse-state research line.
It does not replace DEFORM, modify its successful result, or revive any failed
forecast-sensing, weak-constraint, or covariance gate.

## Scientific Motivation

DEFT is the authors' branched discrete-elastic-rod simulator, not DEFORM or
DeformMaster. A promising new question is whether sparse measurements on a
parent branch improve prediction on unobserved child branches. That would test
topology-dependent state transfer rather than another mean correction on the
same unbranched examples. Native backend support alone is not that contribution.

Primary references: [official DEFT code](https://github.com/roahmlab/DEFT) and
[the DEFT paper](https://arxiv.org/abs/2502.15037). The pinned source commit is
`5781c70c7737fb84b8bd43261e3ed00ef2fd0fbc`.

## Qualification Before Data

The present lock permits only source code, the declared released checkpoint,
the reference geometry written in upstream source, and synthetic clamp motion.
It does not permit a trajectory accuracy score. A subsequent source experiment
needs a separate committed protocol after native parity passes.

The adapter imports the exact upstream implementation. An AST transformation
adds a pause/restore boundary after a completed native timestep; it does not
rewrite its force, integration, twist, neural residual, or constraint equations.
The transformation is source-hash-bound and rejects a changed loop shape.
Cloned state retains positions, previous positions, velocity, Bishop-frame
direction, twist, optimization masks, rod orientations, and the three junction
iteration-memory arrays. Restart does not reinitialize those dynamic states.

The constructor is extracted from the released BDLO1 source before all dataset
loaders. It uses the released full checkpoint and native checkpoint-loading
semantics, including the source's constructor-time derived arrays. It does not
silently refresh those arrays or claim a new reproduction of the paper's best
number. The batch-specific adjacency buffer is the one permitted omitted key;
every other checkpoint key must match.

The prediction wrapper accepts exactly two initial full states and prescribed
four-point clamp motion. Future free-node positions cannot enter its API. This
is important because upstream `iterative_predict` accepts a fully populated
array as its GNN input, unlike the explicitly masked training routine.

Required synthetic tests compare the unmodified native routine, the resumable
routine, a segmented continuation, and an exact-zero correction. Positions,
velocities, and final internal memory must agree bitwise where declared. Every
input checkpoint is cloned, and finite outputs and clamp preservation are
checked. CPU, one thread, float64, fixed seed, and 20 native constraint iterations
are used. The optional sparse import shim is inherited from the DEFORM CPU
runner; it raises if invoked and does not replace the native dense solver.
The compatibility runtime is Python 3.10.12, Torch 2.0.1+cu118, NumPy 1.24.3,
Theseus 0.2.1, Numba 0.59.1 and pinned PyTorch3D 0.7.7 pure-Python transforms.
It is deliberately reported separately from the upstream README's recommended
Torch 2.5.1+ environment. No shared environment is changed.

## Prospective Data Boundary

If qualification passes, the names-only development candidate is the first
lexicographic BDLO1 training file whose name starts
`BDLO_data_kinova_and_panda`. It is not decoded by this lock. Such a pilot is
training-split capacity evidence, not independent generalization, since the
released checkpoint was trained on that split. The public repository splits
recordings into blocks; blocks must not be called independent sessions.

Acquisition note: a names-and-size Git tree query during the source audit caused
the partial clone to cache some public upstream `tests/trajectories` and
`tests/visualization` blobs. Their contents were never decoded, viewed, printed,
or checked out. The sparse working tree contains only source directories and
root files. Thus the boundary is no evaluation/test content inspection, not a
claim that no test bytes ever entered Git's object cache.

No protected DEFORM DLO3 evaluation, DLO4/DLO5, held-v8, Deform360 target, or
physical Causal4D dataset is opened. No hardware recording is needed. The old
DEFORM modules and frozen evidence remain unchanged.

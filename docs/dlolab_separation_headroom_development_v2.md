# DLO-Lab Separation Headroom Development Screen v2

V2 is a custody-only successor to the terminal v1 launch failure. The public
DLO-Lab task, nine rotated worlds, eleven-member action batch, native reward,
qualification thresholds, and development gates are inherited unchanged from
v1. No v1 native world or outcome exists.

Before creating any output or consuming the one permitted attempt, the v2
runner validates the clean Git revision, all bound source files, pinned public
DLO-Lab source, Python/package runtime, CPU-only variables, and the locally
bound OSMesa library. It then atomically publishes an external attempt ledger
before creating the registered result root. Workers revalidate both the ledger
and lock.

The registered invocation uses:

```text
CUDA_VISIBLE_DEVICES=""
PYOPENGL_PLATFORM=osmesa
LIBGL_ALWAYS_SOFTWARE=1
LD_LIBRARY_PATH=/home/fpfaff/source-only/dlo-lab-decision-v1-assets/native-libs/root/usr/lib/x86_64-linux-gnu
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
PYTHONPATH=src
```

A complete pass is still only bounded public-simulator development evidence. It
does not authorize source transfer or a prospective run automatically. Any
native, custody, or value-gate failure is terminal, receives no retry, and
leaves the unchanged best fixed action as the exact fallback.

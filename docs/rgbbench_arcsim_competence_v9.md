# RGBench ARCSim competence v9

v9 repeats the frozen v8 target-free numerical gate after correcting exactly
two provenance paths:

- `dependencies/lib/libjson.a`;
- `dependencies/lib/libalglib.a`.

v8 stopped before ARCSim invocation because it named those same hash-bound
files under non-existent build subdirectories. v9 preserves the source case,
physical parameters, two-replay design, numerical thresholds, and all forbidden
outcome boundaries byte-for-byte at the JSON value level.

Passing v9 authorizes only a separately frozen full-horizon target-free
qualification. It does not authorize source point-cloud scoring.

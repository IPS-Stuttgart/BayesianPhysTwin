# DEFORM DLO2 two-seed all-train refit v1

This stage exists only for a fresh DLO2 ensemble that passes every locked source
gate. It cannot run from the exploratory DLO1 result and it cannot authorize
either seed independently.

The authorization validator requires both fresh DLO2 source models to pass their
single-member source gates, the selected two-seed ensemble to improve validation
and source error by at least 1%, at least five of eight source-case wins, the
candidate parity and persistence gate, a non-fallback selection seal, and a
finite validation-fitted variance calibration.

Seeds 42 and 43 are then retrained independently on all 56 DLO2 training
trajectories with the same 50-frame horizon, batch size 32, 6,400 updates,
official SGD parameter groups, and seed-specific deterministic schedules. Each
run materializes exactly the checkpoint update selected for that seed on fresh
DLO2. Neither run may alter the ensemble operator, seed weights, checkpoint
updates, or calibration.

`assemble_deform_dlo2_deep_alltrain.py` verifies both checkpoint payloads,
schedules, method specifications, source results, selection seal, runtime, and
protocol hashes. Only then does it emit a combined final method. The assembler
does not read evaluation data and does not by itself open or score the official
partition.

The frozen files are:

- `configs/sota/deform_dlo2_deep_alltrain_refit_v1.json`
- `scripts/remote/run_deform_dlo2_deep_alltrain_seed.py`
- `scripts/remote/assemble_deform_dlo2_deep_alltrain.py`

An official evaluator remains a separate one-shot stage and must bind the
assembled artifact explicitly before any DLO2 evaluation trajectory is read.

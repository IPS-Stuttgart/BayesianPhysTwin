# DEFORM DLO2 two-seed official evaluation v1

This is the final, one-shot target stage for the frozen two-seed DEFORM route.
It cannot execute unless both fresh DLO2 seed runs pass their source gates, the
two-seed predictive mean improves fresh DLO2 validation and source transfer by
the preregistered margins, both all-56 refits complete, and their assembled
artifacts preserve the exact selected weights, member updates, comparison seed,
and source-fitted variance calibration.

Before touching the evaluation directory, the runner independently verifies:

- both source protocol and source-result lineages;
- the frozen ensemble result and its selection seal;
- both all-train seed results, schedules, method specifications, and checkpoint
  payloads;
- the assembled method against a duplicated immutable selection summary; and
- the upstream commit plus the PyTorch and CUDA runtime identities.

Only after writing `authorization.json` may the runner enumerate and hash the 14
official DLO2 trajectories. It evaluates every sorted trajectory exactly once,
also reports the frozen canonical with-replacement draw used by the published
DEFORM reference, and seals any post-open failure with retries disabled.

The candidate is the preselected predictive mean of the two all-56 seed
checkpoints. Its comparison arm is the preselected lower-validation seed. No
target selection, calibration, retry, or case replacement is permitted. The
claim gate requires all of the following:

- mean L1 below 9.7 mm on all 14 unique trajectories;
- mean L1 below 9.7 mm under the canonical published-reference draw;
- at least 1% improvement over the comparison seed; and
- at least 8 of 14 paired case wins.

The between-seed predictive variance is evaluated with the source-validation
scale and variance floor unchanged. Coordinate coverage, interval width,
Gaussian NLL, NEES, and early/middle/late horizon diagnostics are reported, but
no official outcome may recalibrate the distribution.

Frozen files:

- `configs/sota/deform_dlo2_deep_official_eval_v1.json`
- `scripts/remote/run_deform_dlo2_deep_official.py`

The official stage remains closed until the fresh DLO2 source and all-train
artifacts exist and pass every upstream gate.

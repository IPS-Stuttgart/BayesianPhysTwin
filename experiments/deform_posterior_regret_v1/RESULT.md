# Posterior Regret Admission V1: Source Gate Failed

Date: 2026-10-01. This is a source-competence discovery experiment on
historically exposed recorded public DLO2 data, not official benchmark scoring
or fresh-object confirmation. The registered Bayesian-risk direction is closed.

## Primary Result

All 40 authorized fitting recordings completed. There are 120 primary contexts
(three prefix endpoints per recording), 360 contexts across all three horizons,
and zero technical failures. Every selector receives exactly the same unchanged
open-loop GP forecast B and prefix-conditioned forecast C.

At 50% acceptance and the primary 100-frame horizon:

| Selector | Harmful accepted / 60 | Calibrated Brier | Raw Brier | Deployed coordinate RMSE (mm) |
| --- | ---: | ---: | ---: | ---: |
| Posterior risk | 8 | 0.147965 | 0.172618 | 18.430898 |
| Matched predictive distribution | 8 | 0.147965 | 0.172618 | 18.430898 |
| Expected regret, no covariance | 12 | 0.150801 | 0.183333 | 18.471422 |
| Constant diagonal | 9 | 0.149583 | 0.183069 | 19.123970 |
| Trajectory-cluster sandwich | 9 | 0.148859 | 0.172943 | 18.480780 |
| Whole-trajectory residual bootstrap | 10 | 0.151646 | 0.406667 | 19.113030 |
| Direct empirical regret | 12 | 0.157121 | 0.161386 | 19.255588 |
| Ridge regret predictor | 14 | 0.161268 | 0.216043 | 18.833696 |
| Kernel regret predictor | 9 | 0.150548 | 0.149151 | 19.155361 |

The strongest independent control under the frozen rule is sandwich covariance.
Posterior risk accepts one fewer harmful correction: 11.11% reduction, below
the required 20%. Its Brier improvement is 0.000893823, below the required
0.005. The panel has 98 beneficial and 22 materially harmful contexts, so it
passes the context-count prerequisites but fails both superiority thresholds.
These are discovery criteria, not significance claims.

**Study One: failed. Study Two: not run and not authorized.** No DLO4/DLO5
replication data, historical validation, original source-test recordings,
official evaluation, held-v8 or other reserved cohort was opened by this study.
Secondary horizons cannot rescue the failed primary gate.

## What This Does And Does Not Show

Prefix conditioning is useful for point prediction: always-B coordinate RMSE is
19.769447 mm versus 18.038659 mm for always-C on this cross-fitted source panel.
That is not the tested Bayesian admission contribution. The posterior's small
advantage over a strong empirical covariance does not satisfy the registered
standard, and the matched distribution reproduces its decisions exactly.
Retain the separate residual-adaptation evidence without claiming that this
study established Bayesian superiority.

Conditional-C marginal coordinate coverage is 88.705% at nominal 90%, mean
coordinate NEES is 1.3833, and mean 90% interval width is 49.811 mm. These are
marginal coordinate diagnostics, not joint trajectory coverage or joint NEES.
The model retains full cross-time covariance for regret risk but assumes
independent node/coordinate blocks in its fixed local frame.

At horizon 25 posterior/sandwich accept 17/16 harmful contexts; at horizon 200
they both accept 12. Neither secondary is a new confirmation or a rescue gate.
The complete recording, not a node or point-frame, is the statistical unit.
These coordinate errors must not be compared directly with published official
DEFORM errors from another split or metric.

## Evidence And Reproduction

- Frozen execution commit: `9600807f9bba6c0d0aaa6f9db3c93c6b41968c0a`.
- Original GP source: `1b0622c823d6d927ab9d4863911caab67e98ebaf`, unchanged.
- [Protocol](PROTOCOL.md), [source gate and all curves](results/report.json),
  [recording outcomes](results/per_recording.json),
  [context outcomes](results/per_context.json), and
  [prediction-only selection seal](results/selection_seal.json).
- [Independent compact audit](results/independent_audit.json),
  [independent sealed-array audit](results/large_pair_audit.json), and
  [runtime/provenance](results/provenance.json).

The dedicated CPU run took 886.805 seconds on gpuserver6000, using one BLAS
thread and no GPU. The large sealed forecast pairs remain in the dedicated
source-only run directory and are SHA-256 bound by the five prediction seals;
all 360 recorded outcomes were independently recomputed from those arrays.

Verification: 24 focused tests pass locally and on native Linux; the full local
suite passes 8,202 tests with 50 skips. Changed-file Ruff, formatting and
diff-check pass. GitHub's full test matrices on Python 3.10/3.12/3.14, core,
provider and coverage jobs pass. Its quality/preflight jobs are blocked solely
by an unchanged inherited temporary workflow expiring on 2026-09-15; package
build/install jobs consequently did not run. Do not report all CI green.

Two pre-outcome technical attempts are retained: an incorrect clamp-admission
check failed before fitting, and a training-only attempt was interrupted before
held predictions to freeze a numerically equivalent covariance contraction
optimization. The protocol records both amendments. No completed gate was
retried, no observed outcome changed the frozen method, and no default or
original GP artifact/workflow was modified.

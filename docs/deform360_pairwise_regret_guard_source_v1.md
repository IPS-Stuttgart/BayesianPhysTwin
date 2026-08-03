# Deform360 Pairwise Regret Guard: Source V1

## Decision

The source gate passes and authorizes one no-refit accuracy evaluation on
genuinely fresh Deform360 physical objects. This is post-open method
development on five source objects, not independent confirmation or a SOTA
claim.

The candidate combines the strongest raw-prefix observation result with the
failure mechanism exposed by the prospective camera-only study:

1. select the physical or persistence backbone from the current prefix only;
2. construct the unchanged pairwise-consensus RBF camera correction;
3. require non-trivial action-conditioned physical response;
4. cap correction RMS relative to that physical response;
5. admit the bounded correction only when its source-fitted regret upper bound
   is strictly negative;
6. otherwise return the selected backbone bit-for-bit.

The runtime builder accepts neither a target nor an outcome. Source outcomes
enter only this separate qualification stage.

## Source Result

Metrics are object-balanced across the five physical source objects. Lower is
better.

| Method | Hidden identity RMSE (mm) | Hidden Chamfer (mm) |
| --- | ---: | ---: |
| Selected physical/persistence backbone | 8.807 | 7.888 |
| Outer-LOO guarded update | **8.561** | **7.619** |
| Relative improvement | **2.79%** | **3.42%** |

The outer leave-one-object-out guard admitted four of 49 target-free eligible
intervals across two held-out physical objects. All four admitted intervals
improved both co-primary metrics; none was harmful. The bounded candidate
without the regret guard improved the aggregate more broadly, but still had 22
harmful intervals, so it is not the deployable method.

## Control Arms

The primary gate was calibrated with two controls using the same feature and
cross-fitting implementation:

| Control | Result |
| --- | --- |
| 1,024 within-object outcome permutations | 0 full gate passes; 7.52% produced any acceptance; 1.95% produced a nonempty zero-harm acceptance |
| Known -2 mm regret injection on production feature rows | Pass; 42 intervals across all five objects admitted, 11.81% identity and 13.18% Chamfer improvement |

The full source payload was regenerated directly from the checksummed physical
prediction seals, causal raw-camera measurement artifacts, and explicitly open
source outcomes. Every candidate feature, report, score, and interval regret
matched the earlier development payload exactly across all 27 cases.

## What Is New

The raw pairwise-clique RBF decoder was already a source-positive observation
path: relative to the selected physical/persistence backbone it improved hidden
identity RMSE by 15.74% and Chamfer by 14.10% on the opened Deform360-27 panel.
It was not safe to deploy. A later prospective camera-only run regressed every
one of twelve object means because a coherent camera bias is observationally
indistinguishable from true discrepancy when persistence is already nearly
exact.

This successor adds the missing Bayesian-PhysTwin contribution:

- physical and action-conditioned support for whether a discrepancy update is
  plausible;
- a correction magnitude bound relative to the physical response;
- explicit common-mode and camera-redundancy features;
- a baseline-relative source regret certificate;
- exact fallback outside physical support, source feature support, or negative
  regret support.

Two-view triangulations remain eligible but expose zero three-view redundancy
to the certificate. This avoids the technical failures caused by treating a
hard three-view plan as an admission contract while still allowing redundancy
to affect trust.

## Limitations

Five source groups are not enough for a calibrated safety statement. The outer
four-group folds have finite-sample coverage 3/5 = 60%; the final five-group
deployment certificate has coverage 3/6 = 50%. The result says the fixed rule
transfers within the opened source panel and beats its matched placebo controls.
It does not say the rule is uniformly non-worsening or better than SOTA.

The improvement is also selective: only four intervals from two objects are
admitted. A fresh evaluation must therefore report both accuracy and admission
coverage, preserve every technical failure, and avoid interpreting exact
fallback as a method win.

## Next Gate

Before any fresh object is selected:

1. union all hash-only exclusions from Prob4D, MolmoMotion, held-v8 source
   history, open-27, the prospective camera-only cohort, and every other opened
   or reserved Deform360 study;
2. preflight metadata enums, required streams, episode length, camera-panel
   sufficiency, and the physical backend's minimum node count;
3. lock the physical-object cohort, episodes, failure accounting, code commit,
   source certificate, and admission thresholds;
4. build and hash every causal-prefix prediction before opening any future
   object outcome;
5. proceed to scoring only if the prediction cohort barrier passes without
   replacement.

The fresh result must beat the unchanged selected physical/persistence
backbone on both object-balanced hidden metrics and retain a nonzero admission
rate. It must be reported as a new-object Deform360 result, not official Table-4
parity unless the official benchmark contract is reproduced separately.

## Evidence

- `configs/sota/deform360_pairwise_regret_guard_source_v1.json`
- `results/sota/deform360_pairwise_regret_guard_source_v1/source_payload.json`
- `results/sota/deform360_pairwise_regret_guard_source_v1/source_qualification.json`
- `results/sota/diagnostics/deform360_raw_alltracker_pairwise_gate_v1/summary.json`
- `docs/deform360_selective_virtual_sensing_result.md`

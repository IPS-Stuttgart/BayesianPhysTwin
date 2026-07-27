# Graph-spectral discrepancy source result

Run date: 2026-07-27

Status: source gate failed; exact endpoint persistence retained.

## Result

The frozen low-capacity graph-spectral family was evaluated by whole-case
cross-fitting on 17 already-open source interactions. Every scored interaction
was excluded from the fit of its source transition prior, and its local update
used only the first 75% of the released training interval. Known controller
actions, but no future object observations, extended through the validation
suffix.

The selected arm is blend zero, which is exactly dense endpoint persistence.
No target artifact was read.

| Source validation arm | CD | Manual track | Change vs persistence |
|---|---:|---:|---:|
| Raw physical baseline | 7.948 mm | 15.518 mm | - |
| Dense endpoint persistence | **6.441 mm** | 14.001 mm | reference |
| Best nonzero spectral arm | 6.475 mm | **13.964 mm** | +0.53% / -0.27% |

The best nonzero arm uses rank 16, temporal smoothing 0.5, source-prior
strength 10, and dynamic blend 0.25. On the case-ratio selection objective it
changes CD by +1.06% and track error by -0.58%, for a balanced regression of
0.24%. It wins CD in 5/17 cases, track error in 12/17, and both metrics in only
4/17. No complete held-out fold wins both metrics in every case. Its worst
single case/metric ratio is 1.0658, above the frozen 1.05 safety bound.

All source gates fail:

- required balanced improvement: at least 3%; observed: -0.24%;
- required joint-win folds: at least 2; observed: 0;
- both aggregate metrics must improve; CD regresses;
- maximum case/metric ratio: at most 1.05; observed: 1.0658.

## Post-open headroom

A diagnostic per-case oracle over the complete frozen candidate bank improves
the balanced ratio by only 0.78%; its mean CD ratio is 1.0003 and track ratio
is 0.9840. Restricting the oracle to candidates that do not worsen either
metric in a case yields only 0.51% balanced headroom and selects a nonzero arm
in 5/17 cases.

This is too little model-class headroom to justify another rank, smoothing,
prior-strength, or blend sweep. The failure is not merely a poor global
selector.

## Interpretation

The spectral continuation reveals a repeatable tradeoff: small
action-conditioned changes can improve material-identity tracking while moving
the predicted surface enough to hurt Chamfer distance. The same geometry versus
identity tension appeared in the redundant-view CoTracker3 experiments.

Endpoint persistence remains unusually strong once the early residual is
known. Shared scalar dynamics over low graph frequencies do not supply the
missing SOTA-scale gain, even with causal hierarchical prefix adaptation.
Prob4D must not be added as an initializer for this rejected dynamics family.

The remaining credible route is an observation/state belief update with
physical and action support, an explicit coherent-bias nuisance model, a
source-calibrated regret guard, and exact fallback. That route is separate from
this source gate and must be evaluated on fresh objects under its own lock.

## Provenance

- frozen implementation commit:
  `590c9de2da0b62654c0a0ea34ea9f14f05ab8b76`
- deployment bundle SHA-256:
  `d42d988dfd9147ac5606931c847fe09c970ba6dd40a2fa86998485d1d7cab178`
- full summary SHA-256:
  `83ce90e6414aa5a41805e5a6c07e169d00735d76a9ddf89751dd5b4d1019f574`
- run log SHA-256:
  `eef48af0847b0d2622f86b2aa7ee0e06a4a7605617e2ecc4433311d157312470`
- remote execution root:
  `gpuserver4090:/home/florianpfaff/bpt-graph-spectral-source-v1`
- archived evidence:
  `results/sota/diagnostics/phystwin_graph_spectral_residual_source_v1/`

No held-v8 artifact, process, query, score, barrier, target, or outcome was
inspected or modified.

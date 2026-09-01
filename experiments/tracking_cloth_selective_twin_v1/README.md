# Selective Digital Twin: real-cloth feasibility screen

This experiment asks a narrow physical question:

> Can a gate fitted on other cloth materials decide, before a held-out material's
> outcome is used, when the maintained spring-mesh twin should replace
> persistence for a particular action, physical query, and horizon?

It reuses the complete public Tracking Cloth Deformation dataset and the
maintained spring-mesh implementation. The outer split is leave-one-material-out.
The simulator candidate is the source-selected Bayesian spring-bank mean. The
fallback is persistence.

The registered queries are full free-marker shape, free-marker centroid,
bottom-edge centroid, and shape radius at 0.25, 0.5, 1, 2, and 5 seconds. A
context is accepted only when all three training materials show nonpositive mean
regret, pooled improvement is at least 1%, and practical harmful use is at most
10%. Every rejection copies the fallback loss exactly.

This is **retrospective feasibility evidence**. The twisting outcomes were
already opened by workflow run `33302686759`; the experiment is not fresh
confirmation. Query/horizon rows are repeated task views. Materials are the
outer resampling unit. The result cannot establish deployment safety, calibrated
joint uncertainty, unseen-action transfer, or state of the art.

The real-data run is authorized only by adding the canonical one-file request:

```text
.github/requests/tracking-cloth-selective-twin-v1.json
```

and runs on `[self-hosted, Linux, X64, gpuserver4090]`.

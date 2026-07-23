# Deform360 lifecycle-equivalence execution audit

Date: 2026-07-22--23 CEST

Scope: development-only admission work for the resource-bounded Deform360
trainer adapter. This is not a held scientific result and is not evidence of
state-of-the-art forecasting performance.

## Frozen numerical rule

Commit `27fc0cf819895819bc0cbf8b74078773fd5e219a` first committed the
analyzer, its tests, and the disclosure incident record. The numerical rule
remained unchanged in every later commit:

- exact matched structured-array equality is the primary check;
- otherwise all 14 predeclared metrics must satisfy both
  `cross median <= max(within-original p95, within-wrapped p95)` and
  `cross p95 <= max(within-original max, within-wrapped max)`;
- linear quantiles and fixed pair counts are used; and
- the renderer uses the fixed 21-camera gsplat contract.

The existing 5+5 GPU-0 cohort was already partially unblinded as documented
in `deform360_resource_lifecycle_equivalence_incident_20260722.md`. It is a
diagnostic cohort only. No artifact from it can satisfy the fresh v2
qualification or any held attempt.

## Technical attempts

Each attempt used a new detached clean checkout and a new output root. Failed
roots were retained; none was repaired or reused in place.

1. `27fc0cf819895819bc0cbf8b74078773fd5e219a`: manifest preparation
   rejected the copied dataset because its transforms retained the canonical
   absolute seed-Ply path. No manifest or result was published.
2. `2247e5b4ff0bc6c6547bb1ee087340ee2285f7c5`: the exact materialized
   canonical seed alias was bound successfully, but live preflight rejected
   Python 3.12's boolean `sys.flags.safe_path`. A signed manifest was created;
   no result was published and no cohort render occurred.
3. `7affcc5ce1bb8614c24a40ec7914a486ea7b2a3e`: rendering completed in
   memory, but post-render validation rejected the renderer-added
   `CUDA_MODULE_LOADING=LAZY` environment variable. No result was published;
   no numerical values were inspected or reconstructed.
4. `af3f7ce43475361d0b1b07c37f55d45851c8b2ed`: rendering completed in
   memory, but post-render validation rejected transient import paths created
   by the frozen runtime. No result was published; no numerical values were
   inspected or reconstructed.
5. `91732f6805c0520d7bdecb8ef8c1fa3364c8a5f3`: a synthetic, target-free
   lifecycle gate found that cold module import and renderer setup create
   their transient paths in separate phases. No cohort manifest or result was
   created.
6. `b78db2ec80519b9bd7ed9381cb2bf2f7784b9dfc`: the target-free lifecycle
   gate first passed with exact pre/post execution bindings and an empty
   temporary directory after exit. The subsequent development analysis
   completed and published a signed result.

All post-preregistration changes were provenance, type-compatibility, or
resource-lifecycle fixes. They did not change the PLY parser's metric values,
the metric set, the pair construction, the renderer contract, the quantile
method, or either acceptance inequality.

## Accepted development artifact

Root:

```text
/mnt/corsair/florianpfaff/bpt-resource-lifecycle-equivalence-dev-b78db2ec8051
```

Bindings:

```text
analyzer commit       b78db2ec80519b9bd7ed9381cb2bf2f7784b9dfc
analyzer source SHA   43056e39ff7ea5f760f18420784db0edbb75523031dba7f3a19eca0c6951c128
manifest file SHA     8e6f928707cb635be2a8d2c0daf5fafa6ff988e09f6b2cca61caa37935c0aff8
result file SHA       cd1149087925e2dfc99dd9d492d6de1bcb1d06730532b951d3e396b698adf8d9
result artifact SHA   bf0753e61726f63c4ec0ee0d9bbec3c3faa854def94fdc14e9ed59e07f372775
```

The result is mode `0400`, has link count one, has a valid self-signature, and
the detached analyzer checkout remained clean. Its temporary directory was
empty after process exit. The exact primary check failed, as expected for
independent stochastic fits. The unchanged secondary gate passed all 14
metrics over 10 within-original, 10 within-wrapped, and 25 cross-mode pairs.
Pre/post source, runtime, execution, manifest, and transitive-input state were
exactly equal.

The tightest observed secondary boundary was the p95 of cross-mode maximum XYZ
distance: `0.02781774153611858 m` against the within-mode maximum limit
`0.027821092410326594 m`. Therefore the development result supports running a
fresh qualification, but its narrow margin is also a reason not to treat it as
formal admission evidence.

### Post-acceptance label-sensitivity diagnostic

After the signed decision was published and read, an exact diagnostic
enumerated all 126 unique partitions of the ten fits into two unlabeled groups
of five. For each partition it reconstructed the same 10/10/25 pair groups and
applied the unchanged 14-metric gate. Only 5 of 126 partitions (`3.97%`)
passed. The actual original/wrapped partition had a worst relative slack of
`0.00012044366046426057` and ranked fourth-highest of 126 (`97.6`th empirical
percentile).

This is a post-hoc calibration diagnostic, not a new acceptance rule. It shows
that the engineering envelope is highly conservative and label-sensitive at
five repeats per mode. Under an exchangeability interpretation, a fresh
failure could therefore be a false rejection of an equivalent wrapper. The
v2 protocol retains the frozen rule, but such a failure must be described as
an inconclusive admission failure rather than evidence of scientific model
failure or wrapper inequivalence.

## Next admissible step

The v2 qualification must generate a fresh 5+5 cohort on physical GPU 1 from
one clean H1 tree, apply this unchanged analyzer in `same-as-analyzer` mode,
and then pass the 243-fit resource soak. Only a sealed, self-contained v2
qualification may admit a fresh held attempt.

The watcher for H1 commit `756a57733d3710f7cf211231947d56dec1832859`
was stopped before GPU 1 became available and before the qualification root
was created. A subsequent freeze audit found a separate held-outcome promotion
and terminal-sealing gap. That H1 is therefore retired without a qualification
result. The analyzer and numerical equivalence rule above remain frozen, but a
new H1 must bind the repaired promotion, raw-score recomputation, terminal
inventory sealer, and exact `RLIMIT_NOFILE=1024` boundary before the one-shot
qualification is queued again.

The new H1 must additionally bind the post-GO confirmation-source operator.
That operator may publish only the exact six preregistered cases from
`brownu/deform360@7fea8e20231a47641d1d2bc8791920ec4e62ec5e`: tactile files
remain metadata-only, while exactly 168 camera/shared payloads are processed
under Deform360 commit `0fe36f0b7a7a917ba62b5f8cee707299a9a4a317`.
Publication is cohort-atomic and the source manifest is recursively revalidated
through role and terminal sealing. Its single held-root runtime path must be
fresh and empty after success; cleanup failure is a fail-closed integrity
failure. These requirements are prospective and produced no score or gate
result.

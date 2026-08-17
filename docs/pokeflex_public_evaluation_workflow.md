# PokeFlex Public Evaluation Workflow

`pokeflex-public-evaluation.yml` provides the Actions entry point for the
checked-in PokeFlex source experiment. It deliberately exposes only two modes:

- `contracts` validates the command registry, frozen protocol hashes, causal
  information boundary, and failure-accounting contract on a GitHub-hosted
  runner. It does not access PokeFlex data.
- `source-validation` runs the already-open frozen source panel on a
  self-hosted GPU runner. It is retrospective/exploratory evidence, not the
  official 18-object comparison, prospective confirmation, or a SOTA claim.

The workflow never downloads or caches the public dataset. The self-hosted
runner must provide an extracted PokeFlex root, the exact upstream
`pokeflex-dataset/reconstruction` checkout, the three released Kinect
checkpoints, and a Python environment containing PyTorch, PyVista, and Trimesh.
Paths can be supplied as dispatch inputs or configured with repository
variables:

```text
POKEFLEX_DATA_ROOT
POKEFLEX_UPSTREAM_CHECKOUT
POKEFLEX_CHECKPOINT_ROOT
POKEFLEX_PYTHON
```

Run the data-free contract job with:

```bash
gh workflow run pokeflex-public-evaluation.yml \
  --repo IPS-Stuttgart/BayesianPhysTwin \
  --ref main \
  -f profile=contracts
```

Run the complete frozen source panel with:

```bash
gh workflow run pokeflex-public-evaluation.yml \
  --repo IPS-Stuttgart/BayesianPhysTwin \
  --ref main \
  -f profile=source-validation
```

An optional comma-separated `take_ids` input can reduce a smoke run to a subset
of the 20 source-panel takes. The registered CLI rejects every take outside that
panel. A complete panel automatically runs the checked-in compact source
analyzer and publishes its object-balanced baseline/candidate table and frozen
gate decision. Subset runs are execution smokes and deliberately do not produce
an aggregate scientific result. Calibration, reserved, prospective, and
sealed-target stages are not workflow options.

## Causal and custody boundary

The workflow resolves the experiment through:

```bash
bpt experiment run evaluate-pokeflex-public
```

Before data access, the command validates the source and registration protocol
hashes, the exact upstream revision, all checkpoint hashes, the selected take
set, and the checked-in runner hashes. The execution manifest fixes those bytes
before the source runner starts.

Candidates may use Kinect, RealSense, and robot evidence only through frames
`f-5` to `f-1`. The deformed mesh at frame `f` is scoring-only. Target error is
never an admission or reliability feature. Failed takes are retained as
`failed-no-replacement`, and an update rejected by the method retains the
released checkpoint exactly. An incomplete panel keeps its progress and logs
but returns a failing workflow status; it is never presented as a completed
evaluation.

Each run uses a unique directory below `RUNNER_TEMP`. Actions uploads only the
execution manifest, progress and summary records, environment description,
compact source analysis, console log, and hashes. Full per-take predictions
remain on the self-hosted runner, avoiding multi-gigabyte artifact uploads and
any contact with unrelated active experiment roots.

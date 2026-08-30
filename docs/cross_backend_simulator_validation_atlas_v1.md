# Cross-Backend Simulator Validation Atlas v1

## Contribution

A backend name is neither a prediction guarantee nor a useful unit of
scientific evidence. The same simulator can execute correctly and still lack
decision headroom, lose against a stronger source comparator, or fail on a
complete action horizon. Conversely, a failure on one exact query does not
erase a separately certified query.

The cross-backend atlas makes this distinction executable. For an exact query

```text
(simulator, task, observation policy, action or action bank,
 metric, world distribution, statistical unit),
```

it records the ordered validation vector

```text
(runtime, native, full horizon, decision headroom,
 source value, prospective value).
```

Each component is `passed`, `failed`, `not_evaluated`, or `not_applicable`.
A pass cannot appear after an unmet applicable prerequisite. Every evaluated
component carries the originating Git revision, repository path, file SHA-256,
and available artifact identity. The atlas copies the compact evidence bytes
from their original commits and verifies their original hashes; it does not
rewrite frozen metrics or rerun a simulator.

This extends the DLO-Lab query certificates from one simulator family to a
common validation-domain protocol spanning public DLO-Lab and RGBench
evidence. It complements the integration-oriented five-backend support matrix:
`fully supported` means maintained software and fallback, while this atlas says
how far an exact scientific query actually progressed.

## Frozen Results

| Exact public query | Runtime | Native | Full horizon | Headroom | Source value | Prospective value | Atlas decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DLO-Lab wrapping | pass | pass | pass | pass | pass | pass | **prospectively certified** |
| DLO-Lab slingshot | pass | pass | pass | fail | fail | fail | rejected |
| DLO-Lab coiling | pass | pass | pass | fail | fail | -- | rejected |
| DLO-Lab separation | pass | fail | -- | -- | -- | -- | rejected |
| DLO-Lab unknotting | pass | fail | -- | -- | -- | -- | rejected |
| ARCSim Dirichlet on RGBench | pass | pass | pass | n/a | fail | -- | rejected |
| Codim-IPC on RGBench | pass | pass | fail | n/a | -- | -- | rejected |
| LibuIPC ensemble on RGBench | pass | pass | fail | n/a | -- | -- | rejected |
| MatPhys pinned runtime on RGBench | fail | -- | -- | n/a | -- | -- | rejected |

Across nine exact queries and five backend identities:

- runtime execution passes in 8/9;
- native qualification passes in 6/9 and fails in 2/9;
- complete-horizon qualification passes in 4/9 and fails in 2/9;
- source value passes in 1/9 and fails in 3/9;
- prospective value passes in 1/9 and fails in 1/9; and
- only the exact DLO-Lab wrapping query is prospectively certified.

The denominators above include `not_evaluated` and `not_applicable` entries and
are descriptive accounting, not a cross-backend statistical ranking.

## Positive Evidence That Survives

The atlas does not reduce the project to a list of failures.

### Prospective decision value

The DLO-Lab wrapping query remains prospectively certified on 288 fresh public
simulator worlds. Mean native decision reward improves by `0.004721`, with a
paired 95% interval `[0.003894, 0.005597]`, one harmed world, and a one-sided
95% harm upper bound of `0.016365`. No other query inherits that certificate.

### A mechanically competent alternative prior

ARCSim completes two byte-identical 1,636-step full-resolution replays, retains
all 9,865 identities, and tracks the registered pins to
`3.817e-13 m`. On the sealed source case it improves one-sided real-to-sim L1
Chamfer from `59.394 mm` for remeshed PyBullet to `53.225 mm`, a `10.39%`
gain. This is useful evidence that the thin-shell prior is mechanically and
numerically viable.

It is not a source-value pass. The same `53.225 mm` is `7.21%` worse than the
frozen selected dynamic comparator (`49.643 mm`) and `27.03%` worse than the
published GarmentDynamics cell (`41.900 mm`). The comparator-bound gate, not
the favorable pairwise comparison, determines the atlas decision.

### Diagnostic localization

Codim-IPC demonstrates exact short-horizon actuation and deterministic replay,
then stalls at step 48 of the complete trajectory. LibuIPC completes three
full-horizon processes but exceeds its replay-spread and pin-tracking limits.
MatPhys stops still earlier at pinned-runtime import. These are different
engineering and scientific failure locations; treating all three as merely
"backend failed" would discard the information needed to choose the next
method.

The three additional DLO-Lab negatives are equally distinct: coiling and
slingshot reach complete native rollouts but lack transferable decision value,
whereas separation and unknotting fail physical qualification before value
analysis. This is the empirical basis for query-conditional admission.

## Exact Fallback

`select_prospectively_validated_candidate` admits a candidate complete belief
only when the exact query has a prospective certificate and current inference
is admissible. Source support, full-horizon qualification, local qualification,
unknown queries, and every failed stage return the original baseline object by
identity. A technically impressive rollout therefore cannot become a deployed
belief merely because later evidence was not collected.

## Claim Boundary

This is a public-data/public-simulator evidence synthesis and an executable
validation protocol. It adds no new outcome, recording, target access, or
solver retry. It is not:

- an official RGBench or DLO-Lab leaderboard;
- a cross-backend accuracy ranking;
- evidence of backend-wide competence;
- a new point-metric state-of-the-art result;
- a physical-robot safety certificate; or
- independent human review.

Its paper contribution is methodological and empirical: a content-addressed,
comparator-bound validation ladder that preserves a real prospective positive,
localizes heterogeneous failures, and turns unsupported scope transfer into an
executable exact fallback rather than a prose caveat.

## Reproduction

The builder rehashes its atlas implementation plus all eight immutable evidence
inputs (the DLO-Lab v4 atlas and seven RGBench capsules), validates their
registered fields, reconstructs all derived decisions, and writes a
content-addressed atlas:

```bash
PYTHONPATH=src python3 scripts/build_cross_backend_validation_atlas_v1.py \
  --output /tmp/cross-backend-validation-atlas-v1.json
```

The committed artifact is
`results/source/cross_backend_validation_atlas_v1/atlas.json`, with artifact ID
`f68efd7c9279219be470464a828720830ec0275c7647ce91aea115ec62656967`.

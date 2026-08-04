# Frozen Full-22 Bayesian Anchor Reproduction

This capsule records a portable, fail-closed orchestration path for the frozen
Bayesian-anchor comparison on the official ordered 22-case PhysTwin cohort. It
addresses the missing source-command record without reinterpreting or replacing
the existing evidence artifacts.

## Frozen identities

- Bayesian-PhysTwin source revision:
  `e393bb6ff61d44815afd8d09dfc5334cb55d5524`
- Bayesian-anchor protocol ID:
  `ee11310a84b92ff2158018a13ef09989e641e7c0ea84733fe8a6abf267093c65`
- PhysTwin source revision:
  `2b6630528141b9cba5a7677c8b88b2129b4a8390`
- Portable trajectory-data identity SHA-256:
  `f67534421ee2f81ec823171427fb0ac66d3ac1762eb1f5b7624ddda92d057ffc`
- Historical full retrieval-manifest SHA-256:
  `c986f9fffe99e63f842bb48eb1d394a6b87663f5c4a4fb99f2a58855875fb125`

The source checkout must be clean and at the exact recorded revision. The data
identity binds the two public archive sources, the ordered 22-case cohort, and
the archive member, byte count, CRC32, and SHA-256 of every required
`final_data.pkl`, `gt_track_3d.pkl`, `split.json`, and `inference.pkl` file. The
capsule also hashes the actual files before fitting.

The raw historical manifest digest is retained for provenance, but it is no
longer used as the portable admission key: retrieval timestamps, cache-reuse
flags, and unused checkpoint/optimization records do not change the scientific
data identity. Both `evaluation_subset_manifest.json` and the minimal
`trajectory_evaluation_manifest.json` are accepted when their normalized
scientific content is identical.

## Run

Install the current checkout with its data and development dependencies. Then
use that checkout containing this capsule, a separate checkout at the frozen
source revision, and the retrieved evaluation subset:

```bash
python -m pip install -e ".[dev,data]"
python reproductions/full22_anchor_v1/reproduce.py \
  /path/to/Bayesian-PhysTwin-e393bb6 \
  /path/to/phystwin-eval \
  /path/to/output/full22-anchor-v1 \
  --workers 8
```

The output directory must be new or empty. Use `--force` only when intentionally
discarding a previous, unadmitted bundle; the command removes that directory
before starting rather than mixing artifacts from two executions.

The frozen source checkout executes the historical confirmation and absolute
22-case comparison modules through `PYTHONPATH`. The current checkout is used
only to validate the portable data identity and create the strict
`RunManifestV2` evidence record.

The hosted workflow `.github/workflows/full22-anchor-reproduction.yml` restores
or selectively downloads the public trajectory subset, creates a detached
worktree at the frozen source revision, runs the capsule with two CPU workers,
validates the evidence bundle, and uploads the result. No self-hosted runner or
private dataset path is required.

## Produced bundle

The output contains:

- the exact two-stage source command in `source-command.txt`;
- a copied capsule and expected-metric contract;
- the source retrieval manifest and a stable `data_identity.json`;
- all 22 per-case Bayesian-anchor trajectories and summaries;
- `full22_comparison.json` with released and Bayesian-anchor metrics;
- `verification.json`, which fails closed on metric drift beyond `5e-7 m`;
- repository, runtime, information-boundary, and claim identifiers; and
- a validated `run_manifest.json` binding every compact input and output.

The expected values cover equal-case and frame-weighted Chamfer distance and
track error for both released PhysTwin and the Bayesian anchor. They are rounded
paper-facing values, so the verifier uses the recorded half-unit tolerance at
the seventh decimal place rather than requiring an artificial byte identity
across numerical runtimes.

## Claim boundary

This capsule reproduces the Bayesian correction layer beginning from released
`inference.pkl` trajectories. It does **not** reproduce PhysTwin's original
inverse-physics optimization. A successful run supports only the recorded claim
that the Bayesian anchor improves the re-evaluated released PhysTwin predictor
under this official 22-case contract. It does not establish overall state of the
art, calibrated raw posterior covariance, or a dynamically admissible simulator
state correction.

After a successful independent execution, copy the compact manifest and
verification artifacts into `FlorianPfaff/BayesianPhysTwin-Paper`, update the
claim registry and evidence ledger together, and remove the source-command
promotion blocker only after their digests have been validated there.

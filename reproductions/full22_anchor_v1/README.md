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
- Evaluation-subset manifest SHA-256:
  `c986f9fffe99e63f842bb48eb1d394a6b87663f5c4a4fb99f2a58855875fb125`

The source checkout must be clean and at the exact recorded revision. The data
manifest must have the exact digest and ordered 22-case content. The capsule
fails before fitting when either identity differs.

## Run

Install the current checkout with its graph and development dependencies. Then
use that checkout containing this capsule, a separate checkout at the frozen
source revision, and the retrieved evaluation subset:

```bash
python -m pip install -e ".[dev,graph]"
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
only to create and validate the strict `RunManifestV2` evidence record.

## Produced bundle

The output contains:

- the exact two-stage source command in `source-command.txt`;
- a copied capsule and expected-metric contract;
- the frozen data-manifest copy and method/configuration locks;
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

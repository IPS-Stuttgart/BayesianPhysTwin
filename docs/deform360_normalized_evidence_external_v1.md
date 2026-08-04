# Deform360 normalized-evidence external study v1

This study is the first BayesianPhysTwin evaluation that selects mounted
Deform360 objects without consulting their numerical trajectories. It is a
narrow external transfer experiment, not an official Deform360 leaderboard or
full physical-twin comparison.

## Scientific question

The preceding full-22 PhysTwin study found that cumulative component evidence
collapsed a 15-component endpoint mixture to effectively one component. A
source-selected scalar temperature of 128 restored mixture uncertainty, but it
sat at the edge of the search grid and slightly harmed point prediction. This
study tests the direct mechanism implied by that diagnosis:

\[
  w_{ik} \propto \pi_k
  \exp\left(\ell_{ik}/\max(n_i,1)\right),
\]

where \(\ell_{ik}\) is cumulative predictive log evidence and \(n_i\) is the
number of supported prefix updates for track \(i\). Dividing by the update count
uses mean evidence per supported observation. It has no fitted temperature,
hyperparameter search, or target-dependent calibration.

The frozen comparison contains four arms:

1. persistence;
2. last supported residual;
3. the unchanged cumulative-evidence model average V1;
4. the same component bank with per-observation normalized evidence.

## Information order and cohort sealing

The protocol was committed before any workflow was permitted to open a numeric
payload. The workflow then executes two separate processes.

The first process performs a **names-only seal**. It may inspect directory and
file names but may not open or hash mounted dataset files. It:

- discovers object identifiers and candidate `.npz` paths from names;
- excludes every explicitly reserved, calibration, source, or previously opened
  object;
- additionally excludes identifiers mentioned in committed Deform360 evidence,
  result, protocol, evaluator, and test paths;
- chooses one archive per eligible object by a frozen SHA-256 ranking rule;
- writes a content-addressed `selection.json` artifact.

Only after that artifact exists does a separate evaluator process start. It may
open exactly the sealed paths and fails closed if an object identity, path,
protocol digest, repository revision, or selection digest changes. Unsupported
sealed archives remain recorded support failures; replacement is prohibited.

This establishes that selection does not depend on numerical target outcomes.
It does not establish that the selected objects were never inspected outside
this repository.

## Accepted representations

### Fixed-identity trajectories

A trajectory must contain a finite array of shape `(T, N, 3)` under a key that
also declares metres or millimetres. Unit inference from numerical magnitude is
not allowed. Predictions are scored by identity RMSE and symmetric Chamfer
RMSE.

### Packed visual hulls

A packed hull archive must contain integer `frame_indices`, integer
`point_offsets`, and metric `points_world_m`. Because point identity changes
between frames, prediction is restricted to global centroid translation. It is
scored by centroid error and symmetric Chamfer RMSE.

Archives satisfying neither contract are retained as unsupported selected
units and are not replaced.

## Predictive evidence

For every rolling one-step forecast, both Bayesian arms report:

- Gaussian negative log likelihood;
- nominal 90% ellipsoid coverage;
- NEES divided by state dimension;
- predictive standard deviation;
- effective component count and entropy;
- between-model covariance fraction.

Point metrics are aggregated within archive and then with equal object weight.
Paired uncertainty uses 10,000 object-level bootstrap resamples with frozen seed
`20260804`.

## Registered gates

The external diagnostic passes only when:

- at least six sealed objects satisfy one declared numerical representation;
- normalized evidence improves mean Gaussian log score and absolute 90%
  coverage error relative to cumulative evidence; and
- normalized evidence is noninferior to cumulative evidence in every available
  representation-specific point metric under the paired 95% upper bound.

Persistence and last-residual baselines are always reported regardless of these
gates.

## Execution

The workflow `deform360-normalized-evidence-external.yml` uses
`[self-hosted, Linux, X64, nvidia-smi]` and the mounted dataset root configured
by `DEFORM360_DATA_ROOT`, falling back to
`/home/github-runner/.cache/datasets/deform360`.

All outputs are written outside the checkout. The workflow fingerprints mounted
names before and after execution, verifies the repository is clean, and uploads
only compact protocol, selection, metric, environment, and integrity evidence.
Raw third-party arrays are never uploaded.

## Claim boundary

A successful run supports only a prospective external-data statement about
rolling one-step endpoint-motion prediction on repository-unmentioned mounted
Deform360 object identities selected from names before numerical access. It is
not official benchmark parity, deployment calibration, intervention prediction,
a complete Bayesian physical twin, or a state-of-the-art claim.


## Upstream provenance and structure seal

The names-only selector admits archives only under the declared upstream release
prefix `data-7fea8e2/replication-v1/observations/`. Generated experiment roots
are excluded even when their object identifier has not appeared in a previous
result. The seal binds every admitted upstream file and directory name, not only
the selected candidate paths, and the identity is rechecked before later stages.

A separate process then opens only NumPy headers plus integer `frame_indices`
and `point_offsets`. Coordinate values remain unopened. Empty packed hulls stay
in archive custody but are excluded from prediction. Each archive is partitioned
into maximal constant-frame-stride segments, and only segments with at least four
usable frames are admitted. Unsupported objects are retained as support failures
without replacement. The content-addressed structure seal must reach six objects
before any coordinate array can be decoded.

For fixed-identity trajectories, the deterministic comparator is the last
**supported** residual independently for each track. Tracks without any supported
residual fall back exactly to persistence. This avoids treating an invalid
immediately preceding transition as an observation.

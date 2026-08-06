# Contributing to Bayesian PhysTwin

Bayesian PhysTwin is both a Python package and an evidence-producing research
codebase. A green unit test establishes an implementation property; it does not
by itself establish an accuracy, calibration, transfer, deployment, or
state-of-the-art claim.

## Before changing code

Classify the proposed change as one of the following:

- **Stable contract:** a versioned artifact, provider, CLI, or installed-package
  interface covered by the support policy.
- **Prospective method:** a method intended for a future frozen experiment.
- **Diagnostic or control:** an audit, ablation, negative control, or
  non-promotable analysis.
- **Frozen reproduction:** code or metadata cryptographically bound to existing
  protocols or evidence.
- **Infrastructure:** packaging, CI, workflow, provenance, or developer tooling.

Do not edit a frozen implementation, protocol, lock, result, or reproduction in
place merely to simplify current development. Add a versioned successor and a
protocol amendment unless the owning evidence record explicitly authorizes an
in-place repair.

## Information-order rules

For prospective and confirmatory work:

1. Define the statistical unit before outcome access. Frames, points, and tracks
   are not independent units when the registered unit is an object, session, or
   execution block.
2. Freeze source, calibration, and target partitions before target payloads are
   opened.
3. Fit covariance, reliability, nuisance priors, guards, thresholds, and model
   selection on the declared source or calibration evidence only.
4. Record every technical failure and preregistered exclusion. Do not silently
   replace failed target units.
5. Preserve exact physical fallback whenever an update is rejected.
6. Do not use an optional method, oracle, semantic prior, or public-data branch
   to rescue a failed registered primary gate.

## Numerical changes

Claim-bearing numerical paths must fail closed. Do not add implicit jitter,
eigenvalue clipping, pseudoinverses, probability clipping, lossy identifier
coercion, or fallback regularization without a versioned contract and explicit
protocol authorization.

A numerical refactor should normally include:

- agreement with the previous implementation on well-conditioned fixtures;
- residual, symmetry, positive-definiteness, and permutation checks;
- near-singular and invalid-input failures;
- unchanged accept/reject and exact-fallback behavior where parity is claimed;
- an explicit statement about whether artifact identities change; and
- exact source and numerical-runtime evidence.

## Local development

Create an isolated environment and install only the extras needed by the path
being changed:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Common checks are:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy src/bayesian_phystwin
python -m pytest -q
python -m compileall -q src tests
python -m pip check
```

Install `.[dev,graph]` or `.[dev,graph,vision]` only when the affected path
requires those optional dependencies. Heavy GPU, private-data, or physical-data
execution belongs in the registered self-hosted workflow, not in an unreviewed
pull-request script.

## Public interfaces and commands

Bayesian PhysTwin installs one executable, `bpt`. Add new operations through the
typed grouped command registry; do not add another top-level console script.
Stable interfaces require installed wheel and source-distribution coverage.
Research operations must be classified as experiment, diagnostic, or archived
and must name their owning protocol or milestone.

Cross-repository code must use the versioned Prob4D and Causal4D boundaries.
Do not import another repository's private implementation modules into a stable
or claim-bearing path.

## Pull requests

Keep one scientific or engineering concern per pull request. The description
must state:

- what changed and why;
- whether the change is stable, prospective, diagnostic, frozen, or
  infrastructure-only;
- the exact fallback and compatibility behavior;
- the tests and exact-head evidence run;
- any artifact, schema, command, or protocol identity that changes; and
- the permitted claim boundary.

Temporary writer workflows, publication capsules, hard-coded pull-request heads,
and validation-only branches must be removed or closed rather than merged.
Workflow changes require explicit review and must use least-privilege permissions,
immutable action revisions, exact checkout identities, and no persisted checkout
credentials unless a narrowly documented publication step requires them.

## Evidence promotion

Results belong in the canonical paper-notes repository only after the producing
run records exact source revisions, configuration and input identities, output
digests, statistical units, access boundaries, failure accounting, and the
registered decision. Negative results and rejected gates are scientific evidence
and must not be discarded or retuned on the same target cohort.

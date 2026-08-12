# Public API policy

BayesianPhysTwin 0.4 separates two compatibility surfaces with different
purposes:

1. the large historical package-root convenience API; and
2. the deliberately small `bayesian_phystwin.v1` integration API used by new
   ecosystem consumers.

Both surfaces have exact, ordered export snapshots. The snapshots are
compatibility ratchets, not claims that every research helper has the same
support level.

## Historical package-root surface

BayesianPhysTwin 0.4 retains a large historical convenience surface at the
package root. Removing or silently changing those imports would create avoidable
breakage for notebooks and companion repositories, while continuing to add every
new contract to the root would make later compatibility work harder.

The exact observed root surface is recorded in
`api/root-public-api-v0.4.json`.

For the `0.4.x` line:

- existing root exports remain available in their recorded order;
- additions or removals require an explicit manifest update and compatibility
  explanation in the same pull request;
- new functionality should normally use an explicit module or namespace rather
  than expanding the package root; and
- a fail-closed correction for causal leakage, provenance ambiguity, or an
  invalid scientific claim may still override the ordinary deprecation period,
  as described in `SUPPORT.md`.

The root is a lazy compatibility layer. `import bayesian_phystwin` parses the
private historical import declarations without executing the recorded estimator,
Deform360, or experiment modules. Accessing
a historical export imports only its defining module and caches the resolved
object. `from bayesian_phystwin import *` remains compatible and therefore still
resolves the entire recorded surface.

Static consumers retain the historical type surface through
`src/bayesian_phystwin/__init__.pyi`. Runtime laziness must not be implemented by
removing type information or changing `__all__`.

The generated migration map
`api/root-public-api-migration-v0.5.json.gz` records every historical export,
its defining module, and whether `bayesian_phystwin.v1` is already the preferred
stable import. The gzip stream is deterministic; pass `--stdout` to the generator
for review-friendly JSON. This prepares an explicit `0.5` migration without
contracting the `0.4` surface or changing scientific behavior. The `0.4`
snapshot remains immutable evidence of the earlier import contract.

## Versioned integration surface

New Prob4D, Causal4D, and independent integrations should prefer
`bayesian_phystwin.v1`. Its exact ordered surface is recorded in
`api/versioned-public-api-v1.json` and is intentionally much smaller than the
package root.

Within the BayesianPhysTwin `0.4.x` line:

- the recorded `bayesian_phystwin.v1.__all__` order and symbol set are frozen;
- a compatible implementation may fix behavior behind those contracts without
  changing their documented semantics;
- adding an export requires an explicit snapshot update and justification that
  it belongs to the long-lived integration boundary;
- removing or semantically changing an export requires a new versioned
  namespace rather than silently altering `v1`; and
- research estimators, experiment runners, and optional candidate methods stay
  in explicit modules unless a demonstrated consumer need justifies promotion.

Importing `bayesian_phystwin.v1` must not import optional graph, vision, data,
Deform360, or experiment modules. The isolated-import regression runs with
Python's `-I` flag and also rejects accidental imports of `cv2`, `h5py`,
`remotezip`, or SciPy.

Provider-specific modules and artifact schemas may remain narrower than
`bayesian_phystwin.v1`. A green API check establishes compatibility only; it is
not evidence of estimator accuracy, covariance calibration, physical transfer,
or deployment safety.

## Validation

From a source checkout with the package importable:

```bash
python tools/quality/check_public_api.py
python tools/quality/check_public_api.py \
  --manifest api/versioned-public-api-v1.json
python tools/quality/generate_root_api_migration.py --check
```

Add `--json` to either public-API checker command for a machine-readable report.
The checker validates the manifest schema, schema-policy pairing, symbol
uniqueness, literal module identity, exact `__all__` order, existence of every
exported attribute, and the project minor-version line.

The complete Python test matrix exercises both snapshots, root laziness,
first-access caching, unknown-attribute behavior, the typing stub, migration-map
drift, and isolated `v1` imports. `MANIFEST.in` requires all API manifests and
their generators in the source distribution.

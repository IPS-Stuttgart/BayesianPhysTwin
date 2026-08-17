# Public API policy

BayesianPhysTwin 0.4 separates three compatibility surfaces with different
purposes:

1. the large historical package-root convenience API;
2. the deliberately small `bayesian_phystwin.v1` artifact-integration API; and
3. the explicit `bayesian_phystwin.inference.v1` guarded-inference API.

All three surfaces have exact, ordered export snapshots. The snapshots are
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

The legacy names are implemented as a lazy compatibility shim. Importing
`bayesian_phystwin` alone does not import the research, Deform360, graph, vision,
or experiment modules that own those names. Accessing a recorded root export
imports its owning module once, returns the original object, and caches that
object on the package. The symbol set, order, and object identities therefore
remain compatible while the default import path becomes NumPy-only and cheap.

The ownership table is recorded in `api/root-export-migration-v1.json`. It maps
every frozen `0.4` root symbol to the explicit module import that new code should
use and binds the ordering to `api/root-public-api-v0.4.json`. The table targets
the planned `0.5` compatibility line, but it does not itself remove a root name,
activate a deprecation warning, or authorize a version change.

A future `0.5` compatibility line may contract the root after documenting
replacement imports and deprecating supported interfaces where practical. The
`0.4` snapshot remains immutable evidence of the earlier import surface.

## Versioned artifact-integration surface

New Prob4D, Causal4D, and independent artifact integrations should prefer
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

Provider-specific modules and artifact schemas may remain narrower than
`bayesian_phystwin.v1`. A green API check establishes compatibility only; it is
not evidence of estimator accuracy, covariance calibration, physical transfer,
or deployment safety.

## Versioned guarded-inference surface

New inference consumers should prefer `bayesian_phystwin.inference.v1`. Its exact
ordered surface is recorded in `api/inference-public-api-v1.json`. It exposes the
supported strict Prob4D candidate-inference entry point together with explicit
complete-belief selection and exact fallback.

The inference namespace deliberately keeps deployment policy outside the
estimator. Candidate construction, nonlinear closure, and the source-frozen
regret guard remain separate caller-owned steps. `finalize_guarded_update` then
records the candidate-inference identity, verifies that numerical admission
agrees with the guard decision, and validates complete-belief selection, selected
object identity, and exact fallback in one immutable result.

Within the BayesianPhysTwin `0.4.x` line:

- the recorded `bayesian_phystwin.inference.v1.__all__` order and symbol set are
  frozen;
- configuration objects are accepted only as `None` or their declared runtime
  types, never through truthiness-based fallback;
- rejected routing returns the exact baseline object;
- optional vision, graph, dataset, and experiment modules remain outside the
  supported import boundary; and
- a new estimator, guard policy, or covariance meaning requires a separately
  reviewed contract rather than a silent behavior change behind `v1`.

A valid inference record is implementation and provenance evidence. It does not
establish provider competence, calibrated covariance, unseen-object transfer,
Causal4D benefit, deployment safety, or state of the art.

## Public module lifecycle

The public API snapshots define symbol compatibility, while
`api/public-module-lifecycle-v1.json` classifies importable public modules as
stable, historical compatibility, or experimental. The registry is bound to the
root-export migration and the two versioned API manifests.

Stable modules own documented artifact, guarded-inference, or provider
boundaries. Historical compatibility modules retain `0.4.x` root behavior
without being promoted into the versioned API. Dataset- and benchmark-specific
root owners remain explicitly experimental. Unregistered modules have no
compatibility promise.

The registry does not move modules, remove historical imports, activate
deprecation warnings, or create scientific evidence. Its complete policy is
documented in `docs/public_module_lifecycle_v1.md`.

## Validation

From a source checkout with the package importable:

```bash
python tools/quality/check_public_api.py
python tools/quality/check_public_api.py \
  --manifest api/versioned-public-api-v1.json
python tools/quality/check_public_api.py \
  --manifest api/inference-public-api-v1.json
python tools/quality/check_root_export_migration.py
python tools/quality/check_public_module_lifecycle.py
```

Add `--json` to any checker for a machine-readable report. The public-API
checker validates the manifest schema, schema-policy pairing, symbol uniqueness,
literal module identity, exact `__all__` order, existence of every exported
attribute, and the project minor-version line. The migration checker additionally
validates exact root-snapshot coverage, runtime owner mappings, and object
identity between every lazy root export and its explicit owning module. The
lifecycle checker validates disjoint lifecycle categories, complete root-owner
coverage, required stable boundaries, dataset classification, and source-file
identity.

The complete Python test matrix exercises all three export snapshots, isolated
root, artifact-v1, and inference-v1 import boundaries, the migration map, and
the module lifecycle registry. `MANIFEST.in` also requires all five manifest
files and all three checkers in the source distribution.

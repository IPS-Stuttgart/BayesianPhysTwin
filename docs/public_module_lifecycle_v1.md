# Public module lifecycle v1

BayesianPhysTwin exposes several kinds of importable modules for different
purposes. A module being importable does not by itself make it a supported
integration boundary. The machine-readable registry at
`api/public-module-lifecycle-v1.json` separates the current `0.4.x` surface into
three explicit lifecycle categories while preserving every historical import.

## Stable modules

Stable modules own versioned artifact contracts, guarded inference, or public
Prob4D and Causal4D integration boundaries. Within the documented compatibility
line, their public semantics, units, shapes, failure behavior, and artifact
interpretation cannot be silently changed. Incompatible changes require a new
versioned namespace, provider API, or compatibility line.

The stable category includes:

- `bayesian_phystwin.v1` and its artifact owners;
- `bayesian_phystwin.inference.v1` and its guarded-inference owners;
- `bayesian_phystwin.inference.components_v1` for separate point-mean and
  covariance admission;
- `bayesian_phystwin.inference.component_beliefs_v1` for semantic validation of
  the five complete-belief arms;
- [`bayesian_phystwin.physical_cause_selection_v1`](physical_cause_selection_v1.md)
  for source-calibrated routing
  among baseline, observation-bias, readout-discrepancy, physical-parameter, and
  physical-state complete beliefs;
- `bayesian_phystwin.structured_point_covariance` and
  `bayesian_phystwin.structured_point_covariance_operator_v1` for exact
  block-local plus labeled low-rank covariance representation and matrix-free
  linear algebra;
- `bayesian_phystwin.identifiability_report_v1` for content-addressed reporting
  of physically reachable state modes distinguishable from a declared bias
  subspace;
- `bayesian_phystwin.causal4d_guarded_belief_provider_v1` for exact Prob4D
  runtime, candidate-construction, guard, and selected-belief identities;
- the public Causal4D provider-v1 and provider-v2 modules; and
- the Prob4D causal-lineage validation bridge.

The component-admission, physical-cause, structured-covariance, and
identifiability modules remain explicit direct imports. Registering them as
stable does not by itself add symbols to `bayesian_phystwin.v1`; symbols exposed
through `bayesian_phystwin.inference.v1` remain governed by its exact API
snapshot. The guarded Causal4D provider is likewise an additive integration
facade; its implementation modules are not separate downstream compatibility
boundaries.

Stable status does not promote every implementation dependency imported by
those modules. It applies only to the explicitly registered module identities
and their documented public contracts.

## Historical compatibility modules

Compatibility modules own symbols retained by the frozen `0.4.x` package-root
convenience surface. Their historical imports remain available, and the lazy
root shim must continue to return the same owning objects. The lifecycle label
does not promote these research-oriented modules into the smaller versioned
integration API.

The compatibility roster and symbol-to-owner table are themselves loaded only
when root exports are inspected or resolved. A plain package import therefore
does not eagerly import the generated roster or any historical owner module.

The warning and removal schedule is deliberately version-gated:

- the complete `0.4.x` root surface remains warning-free and unchanged;
- once the installed distribution enters the `0.5` line, first access to each
  historical root symbol emits a `DeprecationWarning` naming its exact owning
  module and replacement import;
- direct imports from owning modules and the versioned APIs do not warn;
- lazy resolution still returns and caches the original owning object; and
- no root export is removed by the warning policy, with removal not scheduled
  before the `0.6` compatibility line.

The gate is derived from installed distribution metadata. Consequently, adding
the dormant policy to `0.4.x` does not rewrite frozen wheels, tags, evidence, or
runtime behavior. A later `0.5` version decision activates the warnings without
requiring a second ad hoc root rewrite.

## Experimental modules

Experimental modules are dataset-specific, benchmark-specific, or
research-lifecycle surfaces. The current registry classifies every Deform360
root owner, `synthetic_benchmark`, and the direct-import
[`query_identifiability_certificate_v2`](query_identifiability_certificate_v2.md)
as experimental. The query certificate formalizes a local kernel-inclusion test,
but its finite-tolerance diagnostics and coordinates remain study-frozen rather
than a compatibility promise. Exact revisions and frozen artifacts remain
reproducible, but current-main forward compatibility is not promised outside
separately versioned contracts that those modules may consume.

Experimental status does not authorize target access, confirmation access,
retuning, deployment, or a scientific claim. It is software lifecycle metadata
only.

## Unregistered modules

An importable module absent from the registry is internal or experimental and
has no compatibility promise. Underscore-prefixed modules are always internal
and cannot be added to the public lifecycle registry. The generated private
`_root_exports_v0_4` module exists only to preserve the frozen root symbol table
and static re-export information; it is not a supported import surface.

New public modules should be registered only when their intended lifecycle is
clear. Adding a stable module requires a documented consumer boundary and
appropriate installed-artifact tests. Adding a historical compatibility module
requires an explicit package-root compatibility decision. Dataset or benchmark
surfaces should normally remain experimental.

## Validation

Run the fail-closed checker from the repository root:

```bash
python tools/quality/check_public_module_lifecycle.py
```

Add `--json` for a machine-readable report. The checker validates:

- strict JSON syntax and exact schema fields;
- canonical, sorted, unique, and disjoint module lists;
- exact bindings to the root-export migration and stable API manifests;
- complete one-category coverage of every historical root owner;
- experimental classification of dataset and benchmark root owners;
- required stable integration identities; and
- one regular, non-symlinked source file for every classified module.

The root-deprecation regressions additionally prove that `0.4.x` access is
warning-free, `0.5` access names the exact replacement import, lazy object
identity is preserved, repeated access is cached, and unknown attributes do not
produce misleading warnings.

The changed-source quality ratchet executes the lifecycle checker on every pull
request. The manifest, checker, and this policy document are also included in
the source distribution.

A successful lifecycle or deprecation-policy check establishes software-policy
consistency only. It does not establish estimator accuracy, covariance
calibration, provider competence, unseen-object transfer, physical-query
benefit, deployment safety, or state of the art.
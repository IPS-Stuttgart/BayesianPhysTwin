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
- the public Causal4D provider-v1 and provider-v2 modules; and
- the Prob4D causal-lineage validation bridge.

Stable status does not promote every implementation dependency imported by
those modules. It applies only to the explicitly registered module identities
and their documented public contracts.

## Historical compatibility modules

Compatibility modules own symbols retained by the frozen `0.4.x` package-root
convenience surface. Their historical imports remain available, and the lazy
root shim must continue to return the same owning objects. The lifecycle label
does not promote these research-oriented modules into the smaller versioned
integration API.

A future compatibility line may deprecate or reorganize these modules only
after documenting replacement imports. The registry itself emits no warnings,
moves no files, and changes no runtime behavior.

## Experimental modules

The explicit experimental category covers dataset-specific, benchmark-specific,
or research-lifecycle modules that also own symbols on the historical package
root. The current registry classifies every Deform360 root owner and
`synthetic_benchmark` as experimental. Exact revisions and frozen artifacts
remain reproducible, but current-main forward compatibility is not promised
outside separately versioned contracts that those modules may consume.

Experimental status does not authorize target access, confirmation access,
retuning, deployment, or a scientific claim. It is software lifecycle metadata
only.

## Unregistered modules

An importable module absent from the registry is internal or experimental and
has no compatibility promise. Underscore-prefixed modules are always internal
and cannot be added to the public lifecycle registry.

The external-physics adapter, backend registry, Genesis and JAX-FEM producers,
and source-only qualification module deliberately remain unregistered. They do
not own any historical package-root exports, and registering them in the
explicit experimental list would falsely imply root ownership. Their
content-addressed records remain reproducible, but the modules themselves carry
the manifest's `internal-or-experimental-no-compatibility-promise` status.

A listed backend profile is only a compatibility target, a valid runtime is only
a candidate rollout, and a passing source-only qualification is not
independent-object accuracy, calibrated uncertainty, deployment authorization,
or Causal4D intervention evidence. Stable promotion requires a demonstrated
consumer boundary and the corresponding installed-artifact and scientific
evidence gates.

New public modules should be registered only when their intended lifecycle is
clear. Adding a stable module requires a documented consumer boundary and
appropriate installed-artifact tests. Adding a historical compatibility module
requires an explicit package-root compatibility decision. A dataset, benchmark,
or unqualified backend surface without root ownership should normally remain
unregistered under the fail-closed policy.

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

The changed-source quality ratchet executes this checker on every pull request.
The manifest, checker, and this policy document are also included in the source
distribution.

A successful lifecycle check establishes software-policy consistency only. It
does not establish estimator accuracy, covariance calibration, provider
competence, unseen-object transfer, physical-query benefit, deployment safety,
or state of the art.

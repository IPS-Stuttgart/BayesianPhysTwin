# Repository-Owned Three-Repository Integration Tests

## Purpose

The installed-wheel golden path validates the released package boundary

```text
Prob4D -> BayesianPhysTwin -> Causal4D
```

Each repository may now own tests for the assumptions it makes at that boundary.
The runner discovers those tests from the exact clean `git archive` snapshot used
to build each wheel, stages them outside all source trees, and executes them
against only the three installed wheels.

This replaces the earlier asymmetric rule under which the golden path executed
only tests stored in BayesianPhysTwin.

## Discovery contract

A repository contributes tests through:

```text
integration_tests/**/test_three_repository_*.py
```

Support modules and fixtures beneath that repository's complete
`integration_tests/` tree are copied together with the tests. A repository may
contribute no tests, which permits incremental adoption. The combined run fails
closed when none of the three exact source snapshots contributes a matching test.

The staging helper receives explicit owner-to-root bindings:

```text
bayesian_phystwin=<exact BayesianPhysTwin source snapshot>
prob4d=<exact Prob4D source snapshot>
causal4d=<exact Causal4D source snapshot>
```

Owner labels and source roots must be unique. Output paths must be outside every
source repository. Symbolic links and non-regular entries are rejected rather
than followed or silently omitted.

## Collision-safe staging and execution

Tests are staged below an owner namespace:

```text
run/
  bayesian_phystwin/
  prob4d/
  causal4d/
```

The runner consumes a deterministic list of explicit test paths. Every test file
runs in a separate Python-isolated Pytest process with `--import-mode=prepend`.
Pytest therefore adds only that staged test file's directory for repository-local
helper imports; no source checkout is exposed on `sys.path`. Running one explicit
file per process also allows different repositories to use the same test-module
name without import collisions. Because the parent of an owner directory is not
added, an owner label such as `bayesian_phystwin` cannot shadow the installed
package of the same name.

The path list and the versioned JSON inventory are created with exclusive
no-clobber semantics. The inventory records every owner, the sorted relative
test paths it contributed, and the complete test count. It is execution
diagnostic metadata; it is not added to the scientific evidence bundle.

## Validation

The focused workflow checks:

- Ruff lint and formatting for the helper and policy tests;
- deterministic multi-owner staging and support-file preservation;
- duplicate owner and duplicate root rejection;
- empty-suite, symbolic-link, occupied-output, and unsafe-output rejection;
- exclusive manifest publication;
- exact golden-path source bindings;
- collision-safe explicit per-file Pytest invocation; and
- shell syntax for the installed-wheel runner.

The full installed-wheel golden path remains authoritative for compatibility of
the three exact package revisions. The focused lane protects the discovery and
staging mechanism from regressing between full runs.

## Ownership boundary

Prob4D should own producer and provider-contract assertions. BayesianPhysTwin
should own guarded-update, exact-fallback, and physical-belief assertions.
Causal4D should own accepted-belief consumption and intervention-boundary
assertions. Tests may share immutable artifacts, but one repository should not
reach into another repository's private implementation modules.

Passing these tests establishes package and contract interoperability only. It
does not establish provider competence, uncertainty calibration, physical-query
benefit, Causal4D intervention benefit, deployment safety, or state of the art.

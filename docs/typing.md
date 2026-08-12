# Installed typing contract

Bayesian-PhysTwin ships inline type information under [PEP 561]. The
`bayesian_phystwin/py.typed` marker is included in both wheels and source
distributions, and project metadata declares the `Typing :: Typed` classifier.
This lets installed consumers such as Prob4D and Causal4D use the public
annotations instead of treating the package as untyped.

The supported typing boundary is the public package surface. Modules whose names
start with an underscore remain implementation details, and the marker does not
turn experimental scripts into stable APIs.

## Strict integration boundary

The portable artifact validators, `bayesian_phystwin.v1`, and the public Prob4D
and Causal4D bridge modules are checked as an explicit strict-typing surface.
They are always included in the changed-source MyPy run and are also passed to
`mypy --strict` on every pull request, even when a change touches unrelated
files.

`pyproject.toml` repeats the per-module-capable strict options for the same
surface. This makes ad-hoc and editor MyPy runs use the intended policy rather
than depending solely on the CI command line. The CI invocation remains
authoritative for global-only strict flags and checks that the declarative module
list cannot drift away from the executed file list.

This strictness is an interface-quality guarantee, not scientific evidence. It
does not establish provider competence, calibrated uncertainty, physical
transfer, deployment safety, or state of the art.

A release archive can be checked with:

```bash
python scripts/release/verify_pep561_distribution.py dist/*.whl dist/*.tar.gz
```

The installed-distribution workflow also creates a clean virtual environment,
installs only the built wheel and its dependencies, copies an external consumer
fixture outside the repository, and runs strict mypy checking there. This keeps
the check from succeeding accidentally through the source checkout.

[PEP 561]: https://peps.python.org/pep-0561/

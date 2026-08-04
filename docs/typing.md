# Installed typing contract

Bayesian-PhysTwin ships inline type information under [PEP 561]. The
`bayesian_phystwin/py.typed` marker is included in both wheels and source
distributions, and project metadata declares the `Typing :: Typed` classifier.
This lets installed consumers such as Prob4D and Causal4D use the public
annotations instead of treating the package as untyped.

The supported typing boundary is the public package surface. Modules whose names
start with an underscore remain implementation details, and the marker does not
turn experimental scripts into stable APIs.

A release archive can be checked with:

```bash
python scripts/release/verify_pep561_distribution.py dist/*.whl dist/*.tar.gz
```

The installed-distribution workflow also creates a clean virtual environment,
installs only the built wheel and its dependencies, copies an external consumer
fixture outside the repository, and runs strict mypy checking there. This keeps
the check from succeeding accidentally through the source checkout.

[PEP 561]: https://peps.python.org/pep-0561/

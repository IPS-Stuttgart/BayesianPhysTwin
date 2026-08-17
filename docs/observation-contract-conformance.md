# Observation-belief contract conformance

`phys4d.observation_belief` version 1 is shared by Prob4D,
Bayesian-PhysTwin, and Causal4D. The repositories deliberately keep
independent producer and consumer implementations, but they no longer define
the wire contract independently.

The installed package contains a data-only normative bundle under
`contract_data/observation_belief_v1`. The bundle fixes:

- the closed descriptor and NPZ member sets;
- exact NumPy dtypes and symbolic shapes;
- causal, identity, covariance, probability, and grouping invariants;
- JSON and NPZ serialization rules;
- array and aggregate content-address algorithms;
- two valid vectors, including a rank-zero covariance factor; and
- ten invalid mutations covering semantic, closed-schema, dtype, and digest
  failures.

The canonical development repository is `IPS-Stuttgart/Prob4D`. The normative
bundle was originally frozen under `FlorianPfaff/Prob4D`, and historical
content-addressed artifacts may correctly retain that repository identity.
Every participating repository carries a byte-identical copy so ordinary
installation and CI remain offline and do not require access to another private
repository. The copies are bound by the bundle SHA-256

```text
a62c693a14c227daa1f4c8db850e691a1d0081df0c853cf0174c33d0b8504ce9
```

The small `observation_contract_bundle` module verifies every member hash
before exposing the schema or vectors. Its hashing functions are also the
reference implementation used by the local writer or consumer. Provider-
specific metadata validation remains separate from this neutral wire contract.

Report the installed bundle with the package-local module:

```bash
python -m prob4d.observation_contract_bundle
python -m bayesian_phystwin.observation_contract_bundle
python -m causal4d.observation_contract_bundle
```

Each repository runs its implementation against the same valid and invalid
corpus on Python 3.10, 3.12, and 3.14. A schema change requires a new bundle and
wire version; editing one local validator without updating the normative corpus
is intentionally insufficient.

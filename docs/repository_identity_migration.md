# Repository identity migration

The active project repositories are:

- `IPS-Stuttgart/BayesianPhysTwin`;
- `IPS-Stuttgart/Prob4D`;
- `IPS-Stuttgart/Causal4D`.

Package metadata, documentation links, and cross-repository CI use these
canonical organization repositories.

## Frozen Prob4D artifact compatibility

Before the organization transfer, Prob4D observation artifacts recorded
`FlorianPfaff/Prob4D` in their content-addressed `source_repository` field. That
field cannot be rewritten without changing the artifact identity and invalidating
frozen evidence. Bayesian-PhysTwin therefore recognizes both identities:

```text
canonical: IPS-Stuttgart/Prob4D
frozen:    FlorianPfaff/Prob4D
```

The canonical identity is exposed as `PROB4D_SOURCE_REPOSITORY`. The frozen
identity remains available as `PROB4D_LEGACY_SOURCE_REPOSITORY`.

Recognition is deliberately narrow. The stream ID, causal lineage, metric-anchor
contract, covariance semantics, source revision, and artifact digests must still
validate. For provider-v2 artifacts, the observation descriptor and embedded
provider manifest must use exactly the same supported repository identity. A
canonical descriptor paired with a frozen manifest identity, or the reverse, is
rejected.

The historical semantic validator remains unchanged for byte-compatible frozen
provider-v1 reproduction. The public compatibility boundary presents a temporary
in-memory legacy descriptor to that validator when reading a canonical artifact,
then reports and propagates the original canonical identity and artifact ID.

## Producer policy

New producer releases should emit `IPS-Stuttgart/Prob4D`. Existing frozen
artifacts and manifests retain their original repository strings. Experiments
must bind the exact producer revision and must not normalize repository strings
inside a content-addressed artifact after publication.

New claim-bearing evidence should also record the complete canonical
Prob4D–BayesianPhysTwin–Causal4D commit tuple and the installed wheel identities,
so repository redirects cannot obscure the software boundary that actually ran.

## CI policy

The three-repository installed-wheel workflow checks out the canonical
organization repositories. Credentialed runs still require read access to the
private Prob4D repository. Producer-neutral fixtures continue to validate frozen
legacy artifacts so repository transfer does not erase reproducibility.

# Release claim synchronization v1

BayesianPhysTwin keeps release-facing scientific wording synchronized through a
small, machine-readable contract. The check is deliberately narrower than a
scientific analysis: it verifies that required evidence numbers, comparators,
uncertainty limitations, and independent-validation boundaries remain present
in the release-facing documents that users are likely to read.

The normative contract is `release/claim_contract_v1.json`. It names:

- the release-facing documents that must retain specific literal statements;
- the frozen evidence reports whose exact bytes are bound into the check report;
  and
- the claim boundary carried by every generated synchronization report.

Run the validator from the repository root:

```bash
python tools/release/check_release_claim_sync.py \
  --output release-claim-sync.json
```

The validator rejects duplicate JSON keys, non-finite JSON values, ambiguous or
escaping paths, symlinked required files, duplicate document declarations, and
missing release-claim literals. Literal matching collapses consecutive Markdown
whitespace to one space, so harmless line wrapping does not weaken or break the
contract. Digests and byte counts are nevertheless computed from the exact
original bytes of the contract, every checked release-facing document, and every
frozen source document. The complete report receives its own deterministic
content ID.

## What the gate establishes

A passing report establishes only that the checked repository revision carries
the registered release wording and binds it to exact source-document bytes. It
does not establish that the underlying point prediction, covariance,
independent-object transfer, downstream intervention benefit, or deployment
safety is correct. Those remain questions for their registered evidence and
prospective protocols.

## Change procedure

A scientific result or boundary change must update the owning evidence report
first. The release claim document, README, support policy, changelog, and
machine-readable contract can then be changed together in one reviewed pull
request. Removing a required literal solely to make the synchronization gate
green is not an evidence update.

The repository test suite executes the validator against the real contract and
release-facing files, so ordinary pull-request CI rejects claim drift. Release
preparation should additionally retain the generated JSON report next to the
release-candidate and compatibility-evidence artifacts.

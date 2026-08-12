# Prospective protocol authority registry v1

## Purpose

`bayesian_phystwin.prospective_protocol_authority_v1` resolves which immutable
prospective-study protocol is authoritative for each scientific claim. It is a
separate governance artifact layered on top of
`ProspectiveStudyProtocolV1`; it does not modify protocol bytes, target-access
state, estimator behavior, or previously frozen evidence.

The registry addresses a common failure mode in long-running projects: several
issues, paper notes, or experiment specifications may remain visible after the
scientific method has changed. Individually valid protocols can then contradict
one another while each still looks active. The authority registry makes that
ambiguity machine detectable.

## Entry contract

Each `ProspectiveProtocolAuthorityEntryV1` binds:

- one scientific `claim_id`;
- the human-readable `protocol_id`;
- the exact `protocol_content_id` from `ProspectiveStudyProtocolV1`;
- one literal authority status;
- the content identity of the authority decision; and
- an optional successor protocol identity for superseded protocols.

The supported statuses are:

| Status | Meaning |
| --- | --- |
| `authoritative` | Sole protocol currently governing the claim. |
| `superseded` | Former authority replaced by a registered successor. |
| `historical` | Archival protocol outside the active successor chain. |

An authoritative or historical entry cannot name a successor. A superseded
entry must name one and cannot point to itself.

## Registry invariants

`ProspectiveProtocolAuthorityRegistryV1` fails closed unless:

1. every `(claim_id, protocol_content_id)` pair is unique;
2. every `(claim_id, protocol_id)` pair is unique;
3. each claim has exactly one authoritative protocol;
4. every successor is registered under the same claim;
5. no supersession chain enters a historical protocol;
6. every supersession chain terminates at the sole authoritative protocol; and
7. the graph contains no supersession cycle.

The registry sorts entries canonically, so input order does not change its
content identity. One exact protocol may govern more than one distinct claim,
but each claim receives its own authority classification and decision record.

## Example

```python
from bayesian_phystwin.prospective_protocol_authority_v1 import (
    ProspectiveProtocolAuthorityRegistryV1,
    build_prospective_protocol_authority_entry,
    lock_authoritative_prospective_study,
    require_authoritative_protocol,
    validate_authoritative_prospective_study_chain,
    write_prospective_protocol_authority_registry,
)

old_entry = build_prospective_protocol_authority_entry(
    old_protocol,
    claim_id="fresh-object-covariance",
    authority_status="superseded",
    authority_decision_id=authority_decision_sha256,
    superseded_by_protocol_content_id=current_protocol.protocol_content_id,
)
current_entry = build_prospective_protocol_authority_entry(
    current_protocol,
    claim_id="fresh-object-covariance",
    authority_status="authoritative",
    authority_decision_id=authority_decision_sha256,
)
registry = ProspectiveProtocolAuthorityRegistryV1(
    entries=(old_entry, current_entry),
    metadata={"owner": "paper claim registry"},
)

require_authoritative_protocol(
    registry,
    claim_id="fresh-object-covariance",
    protocol=current_protocol,
)
state = lock_authoritative_prospective_study(
    registry,
    claim_id="fresh-object-covariance",
    protocol=current_protocol,
)
validate_authoritative_prospective_study_chain(
    registry,
    claim_id="fresh-object-covariance",
    protocol=current_protocol,
    states=(state,),
)
write_prospective_protocol_authority_registry(
    registry,
    "prospective-protocol-authority.json",
)
```

The authority-aware lock wrapper records the registry, entry, and claim IDs in
the first lifecycle state. The matching chain validator checks both ordinary
lifecycle ancestry and that exact design-lock binding. Caller metadata cannot
override the reserved authority fields.

The writer uses the repository's atomic no-clobber publication path by default.
Loading reparses every entry, recomputes all content identities, and reruns the
complete graph validation.

## Recommended use

Before target payload access, the study orchestration should bind both:

- the exact `ProspectiveStudyProtocolV1.protocol_content_id`; and
- the authority registry identity containing that protocol as the sole authority
  for the intended claim.

Paper issues and planning notes can remain available as historical records.
Their visible status should agree with the machine-readable registry, but issue
metadata is not itself the authority artifact.

Changing the authoritative protocol requires a new authority-decision artifact
and a new registry identity. It does not rewrite the old protocol or its
lifecycle. A protocol whose target outcomes have already opened must not be
silently reclassified to justify a new confirmatory run.

## Scientific boundary

Authority classification is governance and provenance evidence only. It does
not establish provider competence, uncertainty calibration, physical benefit,
independent-object transfer, Causal4D intervention benefit, deployment safety,
or state of the art. A registry marked authoritative also does not authorize
target access or a paper claim; those remain governed by the prospective-study
lifecycle, evidence decision, claim bundle, and review boundaries.

# Cross-stage evidence-use ledger

## Purpose

A BayesianPhysTwin state update may consume visual, tactile, force, robot-state,
or other raw factors before its belief is passed to Causal4D. A downstream
intervention update must not multiply the same factor, or a correlated proxy,
again as if it were independent.

`bayesian_phystwin.evidence_use_ledger` provides a dependency-neutral,
content-addressed record of every raw factor admitted to one inference stage. It
can be produced by BayesianPhysTwin without importing Prob4D or Causal4D and can
be translated into the corresponding downstream ownership contract.

This is evidence-accounting infrastructure. It does not establish observation
accuracy, calibration, physical-state benefit, intervention benefit, or safety.

## One raw factor, one independent path

Each `EvidenceUseV1` binds:

- an exact evidence-artifact identity;
- a stable raw-factor identity and SHA-256 of the underlying bytes;
- source repository, exact revision, and source-file digests;
- sensor family, stream, clock, and exclusive causal interval;
- one or more correlation groups;
- one inference role; and
- immutable finite metadata.

The supported roles are:

- `state_update`;
- `actuator_abduction`;
- `contact_abduction`;
- `joint_state_intervention_update`;
- `calibration_only`; and
- `evaluation_only`.

`EvidenceUseLedgerV1` binds the entries to one protocol, case, and causal frame
stop. Its identity is independent of input order. Construction fails before
inference when:

- one entry, evidence artifact, or raw-factor identity is repeated;
- identical raw bytes are relabelled with another raw-factor identity;
- an entry crosses the admitted causal prefix;
- one correlation group is consumed in both a state update and an independent
  intervention update; or
- a joint state/intervention factor is also consumed through an independent
  path.

Multiple correlated factors may still be handled jointly inside one stage. For
example, actuator and contact rows can share a correlation group when Causal4D
evaluates them as one intervention-stage likelihood.

## Contact-anchor integration

The helper below converts a validated `Deform360ContactAnchorV1` into an
explicit state-update use record. The caller supplies the stable raw-factor
identity and raw-byte digest because these belong to the acquisition or
preprocessing boundary, not to the reduced anchor itself.

```python
from bayesian_phystwin.deform360_contact_anchor import (
    Deform360ContactAnchorV1,
)
from bayesian_phystwin.evidence_use_ledger import (
    EvidenceUseLedgerV1,
    attach_evidence_use_ledger,
    evidence_use_from_deform360_contact_anchor,
)

contact_use = evidence_use_from_deform360_contact_anchor(
    anchor,
    raw_factor_id=raw_factor_id,
    raw_factor_sha256=raw_factor_sha256,
    stream_id="episode-3-contact",
    clock_id="robot-clock",
    causal_frame_start=contact_start,
)
ledger = EvidenceUseLedgerV1(
    protocol_id="deform360-official-hub-visuotactile-v1",
    case_id=f"{anchor.object_id}:episode-{anchor.episode_id}",
    causal_frame_stop=anchor.causal_frame_stop,
    entries=(contact_use, *visual_uses),
)
batch = attach_evidence_use_ledger(batch, ledger)
```

The complete ledger record is embedded in batch metadata under
`evidence_use_ledger_v1`. The attachment fails if a ledger is already present or
if its cutoff differs from the observation batch.

## Persistence and downstream use

```python
from bayesian_phystwin.evidence_use_ledger import (
    load_evidence_use_ledger,
    save_evidence_use_ledger,
)

save_evidence_use_ledger(ledger, "evidence-use-ledger.json")
verified = load_evidence_use_ledger("evidence-use-ledger.json")
assert verified.ledger_id == ledger.ledger_id
```

Loading rejects duplicate JSON keys, unknown or missing fields, non-finite JSON,
coercible scalar aliases, malformed revisions or digests, and changed content
identities. Persistence uses a temporary file, `fsync`, and atomic replacement;
existing evidence is not overwritten unless explicitly requested.

For a Causal4D consumer, preserve the exact raw-factor ID, raw-byte digest,
correlation group, causal interval, source lineage, and role while translating
the record. A factor intentionally shared by state and intervention inference
must be represented once as a joint likelihood rather than relabelled as two
independent observations.

## Claim boundary

A valid ledger proves that declared factor reuse is internally consistent. It
does not prove that the declared factors are competent, independent in the
physical world, correctly calibrated, or beneficial to either BayesianPhysTwin
or Causal4D. Those remain separate object/session-level empirical gates.

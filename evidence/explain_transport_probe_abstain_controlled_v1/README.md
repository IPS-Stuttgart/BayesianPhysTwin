# Explain--Transport--Probe--Abstain controlled evidence

Generate the deterministic integration result with:

```bash
PYTHONPATH=src python \
  scripts/science/run_explain_transport_probe_abstain_controlled_v1.py \
  --output evidence/explain_transport_probe_abstain_controlled_v1/result.json \
  --report evidence/explain_transport_probe_abstain_controlled_v1/report.md
```

The study exercises every semantic branch of the composed physical-twin
diagnosis state machine. It requires:

- a unique registered explanation to transport;
- a set-valued cause explanation with a fully invariant target to transport
  without a cause label;
- target-specific probes for two noninvariant targets;
- target-directed mean intervention cost at least two thirds below full cause
  identification;
- an omitted cause to return `none_of_the_above` without probing;
- an unresolvable target to abstain; and
- a below-threshold residual to create no correction.

This is controlled local-linear mechanism evidence. It does not establish natural
physical causes, nonlinear closure, real probe validity, held-intervention
transport, deployment, or safety.

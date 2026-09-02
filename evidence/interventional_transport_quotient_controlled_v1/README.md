# Controlled transport-quotient evidence

Generate the frozen result with:

```bash
PYTHONPATH=src python \
  scripts/science/run_interventional_transport_quotient_controlled_v1.py \
  --trials 10000 \
  --seed 20260902 \
  --output evidence/interventional_transport_quotient_controlled_v1/result.json \
  --report evidence/interventional_transport_quotient_controlled_v1/report.md
```

The study keeps state and gauge coefficients exactly confounded under the source
observation. Their sum nevertheless has a unique held-intervention effect, while
their difference does not. The transport quotient uses no diagnostic probe for
the invariant target and one probe for the sensitive target. Full cause
identification probes every trial. The registered result requires equal
prediction quality within tolerance, half the probe use, and zero false
unprobed transport of the sensitive target.

This is controlled local mechanism evidence. It does not establish real-data
transport, natural physical cause labels, nonlinear closure, or deployment
safety.

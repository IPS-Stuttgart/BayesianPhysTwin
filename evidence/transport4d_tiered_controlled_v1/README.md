# Controlled Transport4D tier separation

Generate the deterministic evidence with:

```bash
PYTHONPATH=src python \
  scripts/science/run_transport4d_tiered_controlled_v1.py \
  --output evidence/transport4d_tiered_controlled_v1/result.json \
  --report evidence/transport4d_tiered_controlled_v1/report.md
```

The seven cases exercise every transport tier, action-level descent from an
uncertain stronger tier, target-outcome contamination, and complete fallback.
The controlled result is not public real-data validation of a tier decision.

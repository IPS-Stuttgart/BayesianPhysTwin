# Controlled target-directed intervention evidence

Generate the frozen result with:

```bash
PYTHONPATH=src python \
  scripts/science/run_target_directed_intervention_design_controlled_v1.py \
  --output evidence/target_directed_intervention_design_controlled_v1/result.json \
  --report evidence/target_directed_intervention_design_controlled_v1/report.md
```

The source response identifies only the sum of state and gauge coordinates. The
finite diagnostic roster contains one state/gauge-separating probe, one material
probe, and one cheap but redundant probe. Three target queries are registered:
the already visible sum, the state/gauge difference, and the material effect.

The exact target-directed selector must use no probe for the sum and exactly one
query-relevant probe for each other target. Full cause identification requires
both informative probes for every target. The registered controlled decision
requires all targets to be identified, the redundant probe never to be selected,
and mean intervention cost to fall by at least two thirds relative to full cause
identification.

This is controlled local linear mechanism evidence only. It does not validate a
physical probe, natural cause label, nonlinear closure, real-data transport, or
safe execution.

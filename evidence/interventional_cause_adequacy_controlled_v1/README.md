# Controlled interventional cause-family adequacy evidence

Generate the frozen controlled result with:

```bash
PYTHONPATH=src python \
  scripts/science/run_interventional_cause_adequacy_controlled_v1.py \
  --trials 10000 \
  --seed 20260902 \
  --output evidence/interventional_cause_adequacy_controlled_v1/result.json \
  --report evidence/interventional_cause_adequacy_controlled_v1/report.md
```

The study deliberately includes an unregistered hysteretic response direction.
It compares forced attribution to the five registered causes against the
cause-family adequacy gate. A positive controlled decision requires the gate to
retain registered causes, detect the unregistered direction, prevent false
physical promotion, and lose attribution quality when the action relation is
broken.

This directory does not itself authorize a natural-data cause claim. A retained
result must be generated on the exact reviewed implementation revision and must
remain bounded to the controlled linear mechanism.

# Diagnostic-driven change policy

The architecture represented by `v0.3.0-causal4d-aip` is immutable. The tag,
artifact manifest, registered gates, generated results, and vault must not be
rewritten in place.

After this milestone, an architectural change must cite a failed experimental
diagnostic. Its experiment record must state:

1. the failed metric, gate, or oracle-gap component;
2. the falsifiable explanation being tested;
3. the smallest proposed change that targets that explanation;
4. the comparison and ablation that can reject the change;
5. whether the new evidence changes a controlled, factual, counterfactual, or
   calibration claim.

Current diagnostic priority: model discrepancy, especially frame, rest-geometry,
or graph-smooth structural error inside PhysTwin. The existing audit does not
justify widening the intervention bank, adding more theta particles, or adding
semantic architecture first.

Corrections to release instructions or integrity tooling require a new patch
milestone. Changes to research behavior require a new minor milestone and must
retain `v0.3.0-causal4d-aip` as the comparison baseline.

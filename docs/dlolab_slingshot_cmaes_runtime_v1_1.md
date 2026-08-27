# CMA-ES Worker Runtime Binding Amendment

The original v1 attempt at `0390595f629f9f231bfd535d63bbb4d4581e303c`
failed before its first worker created a native execution claim. Importing
the official `trajopt.cmaes` helper imports Genesis, whose particle utility
appends its mesher directory to `LD_LIBRARY_PATH`. The fresh child inherited
that changed string, while the qualified runtime requires the original local
OSMesa path. This was a launcher/runtime defect, not a controller observation.
There were zero native evaluations, generation rewards, or selected controllers.

The original root and its failure, lock, proposal bundle, and log remain
unchanged. The v1.1 launcher verifies their exact hashes and refuses the repair
if a native output directory or later generation exists. It launches children
with the exact six environment values from the qualified pre-import runtime;
the parent's upstream import is not changed, and upstream source is untouched.
Each worker independently rechecks that runtime before native initialization.

This is one prospective runtime-only replacement attempt under a fresh root.
The source warm start, CMA-ES seed, population, sigma, 64-candidate budget,
projection, native task/reward, replay, and all competence thresholds remain
as in v1. Any new failure is retained without retry. The frozen scientific
negative results remain negative, and no Bayesian, target, hardware, or SOTA
claim is authorized.

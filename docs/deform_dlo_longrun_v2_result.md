# DEFORM DLO1 long-run and checkpoint-posterior result

The frozen 6,400-update seed-42 run completed and passed its DLO1 source gate.
Its validation-selected update was 6,400. On the eight post-open DLO1 source
trajectories it obtained 9.642 mm mean coordinate-wise L1, compared with the
published DLO1 reference of 10.1 mm and 58.486 mm for exact persistence. It beat
persistence on all eight source trajectories. This is source evidence, not an
official benchmark result.

The registered checkpoint-posterior evaluator then compared only the frozen
parameter-mean, predictive-mean, and coordinate-wise predictive-median arms.
The best validation candidate was `predictive_mean::tail_4_uniform`, at
9.2498 mm versus 9.2562 mm for the selected single checkpoint: a 0.0696%
improvement. That is below the preregistered 1% validation gate. The evaluator
therefore used its exact fallback, did not evaluate a posterior candidate on the
source-test split, and did not authorize fresh DLO2 work for this route.

The first evaluator invocation was a technical no-outcome failure: the script
was launched from the remote home directory although its frozen lineage identity
is repository-relative. It stopped before any candidate rollout. The preserved
attempt-2 invocation changed only the working directory to the exact
`f8e9e3a` archive root; no protocol, candidate, threshold, or data changed.

The independent seed-43 run also completed all 6,400 updates and passed its
DLO1 source gate. It selected update 5,200, obtained 9.0305 mm validation L1,
10.4045 mm source L1, and beat persistence on all eight source trajectories.

The preregistered two-seed evaluator then compared equal-weight and frozen
validation-softmax predictive means against the lower-validation seed. The best
candidate was the validation-softmax mean at 8.9714 mm validation L1 versus
9.0305 mm for seed 43, a 0.6543% improvement. This missed the locked 1% gate.
The evaluator therefore selected the exact seed-43 fallback, reported zero
source-transfer improvement and zero candidate wins by construction, and did
not authorize fresh DLO2 work.

No DLO2 source trajectory or official DEFORM evaluation trajectory was opened.
The checkpoint-posterior and independent-seed ensemble routes are both closed;
the prepared DLO2 source, all-train, and one-shot evaluation machinery remains
unexecuted.

Compact checksummed evidence is in
`results/sota/deform_dlo_longrun_v2/summary.json`; the complete source artifacts
remain immutable under `/home/florianpfaff/source-only/deform-bayesian-v2/runs/`
on `gpuserver4090`.

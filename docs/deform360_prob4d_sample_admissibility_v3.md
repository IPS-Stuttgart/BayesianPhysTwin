# Deform360 Prob4D sample-admissibility v3

This protocol repairs a source-independent contract mismatch exposed by the
frozen v2 public Deform360 source run. The v2 metric stage admitted a camera
stream when any released robot projection was visible, while the downstream
sample materializer required enough joint provider/metric support in every
overlapping causal-prefix window. V2 therefore stopped technically before its
automated calibration gate and produced no model-performance result.

V3 keeps v2's exact target-free camera-visibility decision, then evaluates the
same structural support consumed by sample materialization. For each window it
uses only integrity-bound MotionCrafter `valid_mask` values, released Deform360
metric `valid_mask` values, and frame indices. It does not load provider or
metric point values, score prediction residuals, fit calibration, inspect future
frames, or access confirmation or target outcomes.

The frozen policy requires at least 8 first-frame metric-gauge
correspondences, 8 independent 32-pixel spatial clusters, and 32 held-prefix
rows per window. A stream that misses any requirement is retained as a
support-negative exclusion without replacement. Corrupt or unverifiable input
is a terminal technical failure. The plan is emitted only when all 10 public
calibration objects retain at least 2 streams and at least 90% of the original
324 admitted streams remain sample-admissible.

The source calibration and its transfer/calibration gate remain unchanged. A
passing sample-admissibility gate merely permits that source gate to run; it is
not evidence of prediction benefit. Confirmation remains closed unless the
separate automated source gate explicitly authorizes it. This path uses only
released real-world Deform360 measurements, requires no new acquisition, and
requires no human approval.

# Slingshot Active-Bayes v2 Runtime Preflight

The terminal v1 study reached no simulator episode because it was launched with
the decision-assets interpreter, which lacks `mediapy`. This separate preflight
qualifies the parent benchmark-assets interpreter before any v2 study attempt
can exist.

It runs one prefix-only 300-step native batch using eight copies of an
already-open registered particle world and the parent passive action. It checks
the exact Python path, package versions, parent lock, v1 terminal bytes, controls,
world realization, causal stop, fixed endpoints, and absence of reward/future
simulation. It uses no new continuous world and cannot score the v2 hypothesis.

The preflight has one attempt and no retry. A passing result may be bound by a
separately frozen v2 study; failure leaves that study unattempted. It uses no
protected data, held-v8, DLO4/DLO5, target, GPU, or recording.

# Anytime-valid simulator admission v1 — pre-execution status

## Current state

The statistical core, delayed-outcome controller, frozen fresh-stream protocol,
claim-bearing runner, theorem note, and focused tests are implemented in PR
#872.

The source-formatting and protocol-hardening pass completed successfully at
revision `2931d28da3c9a51575177e1b530f6a22114391da`.

## Frozen evidence boundary

- Development seed domains: `100000:100200`.
- Fresh claim-bearing domains: `200000:200400`.
- Fresh outcomes opened at this status commit: **no**.
- Fresh retries, replacements, and target-dependent threshold changes: **not
  authorized**.
- Candidate: `guarded_recursive`.
- Exact physical fallback: `physical_baseline`.
- Statistical unit: one complete independent seed domain aggregated over all
  eleven registered stress conditions.
- Delayed outcome schedule: frozen before fresh generation.
- Lifetime gain-null budget: `0.025`.
- Lifetime harmful-rate-null budget: `0.025`.
- Composite bad-regime budget when the active null may differ by epoch: `0.05`.

## What remains

Repository CI must pass on a human-authored head revision. Only then may a
one-shot evidence workflow execute the frozen fresh roster. A complete result
must report the candidate, fallback, and actually selected stream; first
authorization time; harmful selected episodes; exact-fallback identity; both
continuous-monitoring null simulations; and all information-order identities.

No positive anytime-valid empirical claim is authorized by this status file.

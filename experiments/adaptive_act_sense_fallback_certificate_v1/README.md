# Adaptive stochastic Act–Sense–Fallback certificate

This controlled finite-hypothesis study requires a depth-two adaptive sensing
policy. A complete policy tree is selected before any probe outcome is observed.
The first noisy probe routes the second probe to one of two task-relevant latent
bits; a cheaper and more accurate nuisance probe is deliberately ignored.

For every retained policy tree, the implementation recursively computes its
expected loss under each physical hypothesis, including probe costs, and then
applies the exact query-quotient worst-case-regret certificate over all
prior-supported compatible complete beliefs.

## Strict separation

- Direct actions: fallback; minimax regret `0.45`.
- Any one-probe policy: fallback; minimax regret `0.45`.
- Best fixed nonadaptive two-probe sequence: worst-case loss `0.475`.
- Adaptive depth-two policy: certified; worst-case regret `0.129`.
- Complete physical state: remains unidentified.
- Nuisance probe selected: false.

The implementation enumerates 1,473 raw trees and retains 860 distinct,
non-dominated loss vectors. Probe outcomes are stochastic and conditionally
independent given the registered finite hypothesis; each probe is nonrepeatable
along a policy path.

Reproduce with:

```bash
PYTHONPATH=src:. \
python -m experiments.adaptive_act_sense_fallback_certificate_v1.run --check
python -m pytest -q tests/test_adaptive_act_sense_fallback_certificate_v1.py
```

## Claim boundary

This is an exact controlled mechanism result for the supplied finite physical
support, quotient masses, terminal losses, conditionally independent probe
models, costs, nonrepeatable finite probe roster, depth limit, retained policy
class, and regret tolerance. It does not validate physical probe models, reset
semantics, support completeness, exchangeability, target transport, online
robot performance, deployment, or safety.

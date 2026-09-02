# Exact support-miss Act–Sense–Fallback phase diagram

This controlled mechanism converts the support-robust certificate from a few
selected epsilon examples into a complete exact phase diagram.

## Mechanism

Four represented hypotheses differ in hidden tether side. Terminal actions are
`pull_left`, `pull_right`, and the caller-owned fallback `hold`.

Two probes reveal tether side on represented support:

- `quick_tug` has zero registered cost but a broad unknown-physics complete-plan
  upper loss of `3.0` for its correct contingent mapping;
- `camera` costs `0.1` but has a tighter unknown-physics upper loss of `1.0` for
  its correct contingent mapping.

All plan lower bounds are zero. Other contingent mappings retain the generic
upper bound `4.0`. The regret tolerance is `0.25` and the phase interval is
\(\epsilon\in[0,0.25]\).

## Exact transitions

For the correct quick-tug plan, robust regret is \(3\epsilon\). For the correct
camera plan it is \(0.1+0.9\epsilon\). Consequently:

\[
\epsilon_{\rm quick\to camera}
=\frac{0.1}{3.0-0.9}
=\frac{1}{21}\approx0.047619,
\]

and

\[
\epsilon_{\rm camera\to fallback}
=\frac{0.25-0.1}{0.9}
=\frac{1}{6}\approx0.166667.
\]

The exact output is therefore:

| Support-miss region | Output |
| --- | --- |
| \(0\le\epsilon\le 1/21\) at the tie convention | Sense with `quick_tug` |
| \(1/21<\epsilon\le1/6\) | Sense with `camera` |
| \(1/6<\epsilon\le0.25\) | Exact `hold` fallback |

At the probe-switch equality the deterministic lower-index plan selects the
quick tug. At the tolerance equality the camera remains admissible. The checked
`result.json` records every additional mathematically irrelevant breakpoint as
well as the compressed output regions.

## Reproduction

```bash
PYTHONPATH=src:. python -m \
  experiments.support_robust_phase_diagram_v1.run --check
```

Use `--write` only when intentionally regenerating the checked result after a
reviewed scientific change.

## Paper role

The experiment establishes an exact, non-grid phase diagram showing that model
support uncertainty can:

1. leave a task-directed cheap probe admissible;
2. make a costlier but support-safer probe optimal; and
3. force an exact fallback after a computable critical miss probability.

It therefore supports the central paper claim more strongly than isolated
examples at epsilon values chosen by hand.

## Boundary

This is a finite controlled mechanism. The unknown complete-plan loss box and
support-miss probability are declared, not estimated. The mechanism does not
validate physical probes, reset semantics, target transport, online execution,
calibration, deployment, or safety.

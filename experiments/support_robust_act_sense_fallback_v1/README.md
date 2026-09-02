# Support-robust act--sense--fallback mechanism

This controlled experiment extends the exact finite contingent-plan certificate
with an explicit upper bound on probability mass outside the represented
physical support.

## Ambiguity model

Let `Delta_0(p,b)` be the exact represented-support worst-case loss gap between
plans `p` and `b`. Up to `epsilon` probability mass may lie on unknown physics.
The unknown complete plan-loss vector is constrained only to a registered
axis-aligned interval `[lower, upper]`, so

```text
M(p,b) = upper[p] - lower[b].
```

The exact worst-case gap over every represented belief, every unknown loss
vector in this box, and every unknown mass `rho in [0, epsilon]` is

```text
Delta_epsilon(p,b)
  = Delta_0(p,b) + epsilon * max(0, M(p,b) - Delta_0(p,b)).
```

The formula follows because the pairwise gap is linear in `rho`; its maximum is
therefore attained at `rho=0` or `rho=epsilon`. The represented and unknown
support functions are themselves attained at vertices of their registered
convex sets.

The implementation also returns, for every complete direct or contingent plan,
the largest support-miss probability for which its robust regret remains below
the registered tolerance.

## Checked phase diagram

The finite mechanism contains terminal actions `pull_left`, `pull_right`, and
caller-owned fallback `hold`. Two probes reveal the represented tether side:

- `quick_tug` is free under represented physics but has a wide unknown-domain
  cost interval;
- `camera` costs 0.16 but has a tighter unknown-domain envelope.

At regret tolerance 0.25 the exact certificate gives:

| Information state | Support-miss bound | Output | Robust regret |
|---|---:|---|---:|
| tether side resolved left | 0.10 | act: `pull_left` | 0.08 |
| side unresolved | 0.00 | sense: `quick_tug` | 0.00 |
| side unresolved | 0.10 | sense: `camera` | 0.24 |
| side unresolved | 0.20 | fallback: `hold` | best plan 0.32 |

Thus misspecification changes not only whether sensing is worthwhile, but which
sensor is admissible. The result is intentionally decision-directed: a sensor
is evaluated through its complete outcome-conditioned action plan rather than
through state entropy.

Regenerate and verify the content-addressed result with:

```bash
python -m experiments.support_robust_act_sense_fallback_v1.run
python -m experiments.support_robust_act_sense_fallback_v1.run --check
```

## Claim boundary

The result is an exact finite mechanism for the declared represented support,
unknown-support probability bound, deterministic probe maps, loss intervals,
and tolerance. It does not estimate the support-miss probability or unknown
loss envelope; validate probe physics, reset semantics, or target transport;
or establish calibration, deployment safety, or real-robot performance.

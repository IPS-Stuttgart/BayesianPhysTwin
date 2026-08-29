# Slingshot Active-identification Particle Result

## Decision

**The 70% frontloaded probe narrowly failed the frozen full-particle value gate,
so the run stopped before continuous truth worlds.**

The sole registered attempt generated four new prefix-only native batches and
combined them with the sealed development slice and unchanged 27-particle reward
table. All four batches passed the 300-frame, fixed-endpoint, padding, command,
and information-boundary checks. The resulting bank contains both passive and
active histories for all 27 material/placement particles.

| Prefix policy | Expected Bayes reward | Gain over blind | Expected MAP reward | Mutual information (nats) |
|---|---:|---:|---:|---:|
| Original passive | 7.165749 | 0.000015 | 7.160141 | 1.077255 |
| Active frontload 70% | **7.170392** | **0.004659** | 7.150318 | **1.910103** |

The blind reward was 7.165734 and the finite-particle oracle was 7.176108,
leaving 0.010375 reward headroom. Active sensing captured 44.90% of that
headroom, improved over passive sensing by 0.004644, and changed the posterior
Bayes action from the blind action in 44.80% of the registered noisy draws. It
passed the native-prefix, active-over-passive, oracle-fraction, and information
checks. Its absolute gain was 0.004659, however, which was 0.000341 below the
preregistered 0.005 threshold. The aggregate particle gate therefore failed.

The frozen boundary was honored: no continuous truth probe, decision, task
future, or score was generated. The output root is terminal and will not be
retried.

## Bayesian Decision Signal

The sealed finite-prior arithmetic exposes a sharper follow-up hypothesis. The
active posterior Bayes rule improved over blind, while choosing an action from
only the maximum-a-posteriori material particle produced expected reward
7.150318, or 0.015416 below blind. The Bayes rule exceeded that plug-in MAP rule
by 0.020074.

This does not establish a continuous-world control gain: the expectation is
under the same registered 27-particle prior used by the selector. It does show
why uncertainty should be propagated into the decision rather than collapsed to
one inferred material. A distinct prospective study may test this newly
generated Bayes-versus-MAP hypothesis on fresh continuous material worlds, but
it must not relabel this failed gate or reuse this output root.

## Verification

The read-only verifier reconstructed the full active history from the four new
native prefix artifacts plus the sealed development carrier, checked the passive
history and reward table against the original bank, reproduced all 4,096-draw
Bayes/MAP calculations and frozen checks, and confirmed that no truth stage
exists. It performed no native replay and passed.

- Frozen implementation: `001b9403bc198bc70171fc15e74df3ef89e5de47`
- Attempt ID: `2a5deea9edb5af7fe7574eaeb524a90155e6fd380053a89bbfb163b5e78e2c34`
- Lock ID: `3dde6f7ec8aed5a68f040f387eb54dfc11a117341c82a282213169abe20d50ed`
- Particle-bank ID: `17b96572a07a3d20818e19f3f31fec4afff98429aea8628f0872e70a3788c22a`
- Result ID: `b202020e4e9e73a92b83a416a09d252394890b7ab02bcd188ac73889e92c3005`
- Particle arrays SHA-256: `d2ec1f6fc9e8495a1eb99c20d2c8815868dadd7ce9f4449904fb3daf39d15e20`

## Claim Boundary

This is prospective public-simulator source evidence, not real-data evidence or
a benchmark/SOTA result. It used the public native Genesis ROD CPU path, no new
recording, GPU, protected target, held-v8 artifact, DLO4/DLO5 data, or official
DLO3 evaluation. It does not change the successful DEFORM forecast, promote the
active controller, authorize continuous truth automatically, or establish
general Bayesian control improvement.

# Slingshot v5 prospective component result

## Status

The registered Slingshot component completed all 128 calibration worlds and all
320 fresh evaluation worlds with zero technical failures. Its own verifier
reproduced the result before the portable component record was exported.

This is a complete positive component result, not yet a joint portfolio claim.
The Wrapping component and the exact two-query barrier remain required.

## Primary guarded result

| Quantity | Result |
|---|---:|
| Evaluation worlds | 320/320 |
| Guard deployments | 65 |
| Exact fallbacks | 255 |
| Mean reward gain over incumbent | +0.00633863 |
| Ordinary bootstrap 95% CI | [0.00435353, 0.00851365] |
| Registered familywise-adjusted gain lower bound | +0.00382284 |
| Harmful worlds beyond the 0.002 margin | 1 |
| Registered familywise-adjusted harm upper bound | 0.01809391 |
| Registered harm budget | 0.05 |

The component therefore passes both of its predeclared portfolio gates. The
adjusted bounds use its outcome-independent query allocation from the v5 joint
protocol, not a confidence level selected after observing the result.

## Matched controls

The always-deployed posterior action obtained a larger mean gain
(`+0.01596792`) but harmed 67/320 worlds beyond the registered margin. The
matched simultaneous-regret guard updated 23 worlds, had mean gain
`-0.00007028`, and harmed 9 worlds. The query-specific policy-gain guard thus
retained positive decision value while sharply reducing downside, and it beat
the matched simultaneous guard on mean gain.

These controls are component-level evidence. No reward is pooled with Wrapping,
and no cross-task effect size is reported.

## Provenance

- Slingshot result artifact ID:
  `7f5622d544d2c8f14c054c28014686b1113af427cae4eb57ee4d92ac6f2cd52d`
- Portable component artifact ID:
  `c7556807bb9c0fb8a78c410be58f13a8ff40f56190a70332e41be80dd69d3c70`
- Portable component file SHA-256:
  `c7ff4fa1adfb8f50b5076b00e0265c9c016c478c9d545e8cb8e65c1fed67ea69`
- Result file SHA-256:
  `f390e64af7a0236cef36b8c5dc246b8b26a22eac644abf46805f3fd43c0cacfd`
- Frozen method/source revision:
  `2395eea8b8d6cead954f0c952b0c36574f5bbf69`
- V5 protocol ID:
  `d1c5f9a7b52281d0762b597f2cb3143891b63f0d063093a6b2706a808a9f9ed6`

The complete world-level component record is retained so every aggregate above
can be independently recomputed without reopening simulator trajectories.

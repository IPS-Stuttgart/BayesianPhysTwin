# PokeFlex All-18 Robust-Scale Source Protocol v4

## Purpose

The successful v3 repeated-action calibration covers twelve of the eighteen
PokeFlex object identities. Six objects still use the global multiplier one in
the public-13 exploratory result because they were not members of either
fresh12 source panel. This source-only protocol evaluates whether the unchanged
v3 maximin rule can provide robust scales for those six objects.

This is the fastest credible route to a complete object-level scale map. It
does not change the radius-0.4 correction field, association, likelihood,
candidate multiplier bank, or byte-identical fallback.

## Frozen Source Selection

The source cases were fixed before running the exact v4 multiplier bank. Every
case had already been exposed by earlier source-development or exposure audits,
but none is the corresponding official validation take.

| Object | Source actions | Excluded official action |
| --- | --- | --- |
| 3dPrintedBunny | T4, T2 | T1 |
| 3dPrintedHeart | T6, T4 | T14 |
| FoamDice | T1, T4 | T3 |
| MemoryFoam | T4, T6 | T2 |
| PlushOctopus | T4, T5 | T6 |
| ToiletPaperRoll | T4, T5 | T1 |

Within each predeclared eligible inventory, the two actions are the smallest
values of

```text
SHA256("pokeflex-action-robust-all18-source-v4" || NUL || take_id).
```

The exact selection and evidence boundary are checksummed in
`configs/sota/pokeflex_action_robust_all18_source_v4.json`.

## Source Run

Each case runs the existing causal checkpoint-registration implementation with
only `action_local_state_relative_0.4` and effective scales

```text
0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5.
```

Scale zero is an exact released-checkpoint control and is not a candidate
multiplier. The source wrapper validates the exact-take allowlist from the v4
protocol and adapts only the already-validated in-memory cohort view supplied
to the generic runner. The frozen generic runner remains byte-identical and
retains its original locked development-cohort behavior for every other call.

The wrapper is:

```bash
python scripts/remote/run_pokeflex_action_robust_all18_source.py \
  TAKE_ROOT OUTPUT_JSON \
  --upstream-checkout POKEFLEX_CHECKOUT \
  --checkpoint-root CHECKPOINT_ROOT
```

The twelve immutable JSON files are then placed in one directory as
`<take_id>.json` and combined with:

```bash
python scripts/development/build_pokeflex_action_robust_all18_scale.py \
  SOURCE_ARTIFACT_ROOT OUTPUT_CALIBRATION_JSON
```

The builder binds every source-artifact file hash and preserves all twelve v3
object rows exactly.

## Gate

The extension passes only when:

- at least three of the six new objects choose a non-global scale;
- the selected scale does not regress against multiplier one on either source
  action for any object;
- the mean source-action improvement is positive;
- the unchanged v3 synthetic positive and placebo controls pass.

Objects with missing support or conflicting action preferences use multiplier
one exactly. A failed gate produces no calibration artifact.

## Claim Boundary

The source actions are opened development evidence. A passing calibration is
not a PokeFlex state-of-the-art result. The public-13 target outcomes were
already opened under frozen v3 and cannot confirm v4 prospectively. Five
official validation records still lack a reproducible public mapping, so an
official-18 evaluation remains prohibited until the PokeFlex authors provide
that mapping or the processed validation set with hashes.

No Deform360 held-v8 artifact, target, process, or identity is used by this
protocol.

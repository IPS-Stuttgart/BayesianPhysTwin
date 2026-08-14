# MatPhys part-aware reconstruction control result

Status: capacity gate passed; predictive use remains forbidden.

## Locked result

The sole registered 200-epoch run completed from the fixed terminal checkpoint
on the already-open `single_lift_sloth` case. The checkpoint used all released
frames, including the future evaluation interval, so this is an offline
reconstruction-capacity control rather than a causal prediction experiment.

| Released-test metric | Released PhysTwin | Part-aware control | Change |
| --- | ---: | ---: | ---: |
| Chamfer distance | 17.860 mm | **11.317 mm** | **-36.63%** |
| Manual-track error | 24.926 mm | **17.356 mm** | **-30.37%** |

Both metrics improve, so the predeclared capacity gate and backend-export gate
pass. The result does not reach the rounded `8/15 mm` future-prediction point
reported by MatPhys, and the comparison is not head-to-head: this checkpoint
was fitted to the same future frames on which it was scored. It therefore
cannot support a state-of-the-art, forecasting, transfer, or calibration claim.

## Backend evidence

- terminal checkpoint SHA-256:
  `c8407722279b04022804e032fb433b2f6156863057363d656ead970628116f26`;
- `16,800/16,800` optimizer steps accepted, with zero rejected steps and zero
  non-finite model or optimizer values;
- all `32,768` zero-initialized DINO projection weights moved and remained
  finite;
- export contains `110,833` positive object/controller springs and all seven
  finite global parameters;
- five graph parts have a `66.25x` maximum-to-minimum ratio between their
  geometric-mean spring values.

The spatial audit matters because the previously rejected causal graph-part
models mostly collapsed to global softening. Here the adapter produces a
genuinely part-dependent field, although the field was learned with future
observations and is not reusable for prediction.

## Decision

This positive control justifies a separately frozen source-only causal design.
It does not authorize this checkpoint or its spring field for any predictive
run. The successor should preserve the exact PhysTwin fallback and pinned
upstream replay, train only from registered source outcomes, expose only an
allowed target prefix to selection, and gate before any fresh future is opened.

The old five-part continuous correction and shared global hierarchy should not
be repeated: both already failed their source gates. The most informative new
family is a part-aware MatPhys proposal with source-calibrated shrinkage and an
explicit part-conditioned topology candidate bank, because the public pinned
implementation exposes only global topology controls and the prior continuous
part-pair fit barely moved. A fresh source panel must pass before a larger
preregistered evaluation is justified.

## Reproducibility

Compact evidence is archived under
`results/sota/diagnostics/matphys_part_aware_reconstruction_control_v1/`.
The locked decision artifact has SHA-256
`4a47e5f41a8838dde5aa0abfb405f508ad169c2056c07d80e836408ebb4f7af3`,
and the descriptive spatial audit has SHA-256
`329945a6fa2cfc13ab1ce9e9dc6bddfbb5562c75f6245d4053ff7fd3bdcb52eb`.

MatPhys primary references: [paper](https://arxiv.org/abs/2605.19386) and
[public implementation](https://github.com/Yrainy0615/MatPhys).

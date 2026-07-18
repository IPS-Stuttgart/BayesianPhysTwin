# Hierarchical mechanics plus discrepancy: frozen 19-case result

Run date: 2026-07-18

Status: complete negative exploratory result. The hierarchy must not replace
the discrepancy-only component.

## Locked question

The experiment tested whether a random-effects posterior over object- and
controller-spring log scales adds value on top of the frozen
action-conditioned discrepancy. Raw profiles used the first 75% of each
released observation prefix, the remaining prefix selected the discrepancy,
and the official future interval was opened once at the final summary stage.

The protocol ID is
`dab6d6b3c1fd593fe23348f531a4fbf6aea8052ae001172c8bd040bd9d9da169`.
The code was fixed at `bc66431a2661e7249ba51b07808c4336e8dca0b1` and the
official PhysTwin source at `2b6630528141b9cba5a7677c8b88b2129b4a8390`.

## Result

Lower is better. Percent changes use the predeclared paired bootstrap; the
primary incremental comparison is against the frozen discrepancy component.

| Comparison | Equal-object CD change | Equal-object track change |
|---|---:|---:|
| Discrepancy component vs released PhysTwin | -8.25% | -4.74% |
| Hierarchical mechanics vs released PhysTwin | +0.77% | +3.81% |
| Combined vs released PhysTwin | -6.32% | -0.33% |
| Combined vs discrepancy component | **+2.00%** | **+4.70%** |

For combined versus component, the object-clustered 95% intervals are
`[-1.50%, +5.40%]` for CD and `[-0.75%, +10.59%]` for track error. The
case-macro track interval is `[+0.61%, +7.98%]`, so the degradation is not a
small point-estimate ambiguity.

The random-effects posterior placed essentially all deviation-scale mass at
`0.30`, while the shared object log scale stayed near zero. Case-specific
stiffness evidence therefore disagrees too strongly for this pooling model to
transfer. The matched raw hierarchical trajectory also loses to its
deterministic zero replay, confirming that the failure is not created only by
the discrepancy layer.

Even a future oracle choosing independently among released PhysTwin, the
discrepancy component, hierarchical mechanics, and their combination reaches
only `9.933 mm` CD and `19.974 mm` track error. Those values remain about 24%
and 33% above the published MatPhys point. This family lacks enough oracle
headroom to be the route to SOTA.

## Reproducibility

- full summary: `results/sota/phystwin_hierarchical_combined_main_v1_summary.json`
- summary SHA-256:
  `d7b3b617ea30b7fd5ab144c1a5454fdc68d9d04c8de562c158a7bcfb88def080`
- locked protocol:
  `results/sota/phystwin_hierarchical_combined_main_v1_locked_protocol.json`
- locked protocol SHA-256:
  `fc7de3be05ea6a78ee32aada490f61b71ed06e1dda6eaedfefc70ad632b8467e`

Thirteen missing posterior-prediction shards were resumed under the exact
lock. One case was deliberately reproduced on both RTX 4090 and RTX 6000 Ada;
its baseline, posterior-mean, and matched trajectories were byte-identical.
The compressed NPZ container hash differed because of archive metadata, while
the scientific trajectory payloads did not.

## Next hypothesis

Do not add more global pooling. The next source-gated family keeps every
released spring as the teacher and fits continuously shrunk offsets for DINO
part-pair groups. Cross-part groups provide a low-dimensional topology proxy,
while an exact-teacher validation fallback prevents a failed physical update
from weakening the selected stack. Its frozen source protocol is
`configs/sota/phystwin_part_pair_source_v1.json`.

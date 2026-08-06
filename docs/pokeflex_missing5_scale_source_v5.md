# PokeFlex missing-five scale source protocol V5

## Purpose

Five official PokeFlex validation archives remain unavailable:
`3dPrintedCylinder_T7`, `3dPrintedHeart_T14`, `3dPrintedPizza_T13`,
`Pillow_T8`, and `Sponge_T10`. Their physical objects have 30 public takes in
the currently released corpus, and none is the unavailable official target.

This protocol uses those 30 already-open actions only to decide whether an
object-specific discrepancy magnitude is stable enough to freeze before the
five target archives arrive. It does not change the V4 official-18 protocol and
cannot itself support a prospective or state-of-the-art claim.

## Frozen method

For every source action, the existing causal PokeFlex runner evaluates the
released checkpoint and the `action_local_state_relative_0.4` correction at
effective scales `0.0625`, `0.125`, `0.1875`, `0.25`, `0.375`, and `0.5`.
Frame `f` uses only depth and robot history through `f-1`. A missing measured
`T_WE` in the required action history discards the correction and restores the
exact checkpoint score; pose sentinels are execution placeholders only.

For each object, the full source selector maximizes the worst per-action gain
over the global `0.125` scale and then the mean gain. Leave-one-action-out
selection repeats that decision without the held action. A non-global scale is
promoted only when:

- the full-action candidate does not regress any source action;
- no LOO-selected candidate regresses its held action; and
- at least half of the held actions improve strictly.

Failure returns the multiplier to exactly `1.0`, not to a learned interpolation.
The complete source gate additionally requires at least two of the five objects
to promote, zero deployed source/LOO regressions, and the frozen synthetic
controls to detect 12/12 positive mechanisms while admitting 0/12 placebos.

## Custody

The protocol binds all 30 ZIP hashes and sizes, the source projection and
evaluation wrappers, the unchanged legacy runner, the registration protocol,
the implementation commit, and the upstream PokeFlex commit. Projected payloads
contain only robot data, camera calibration, depth frames, and scored mesh files.
Every result binds the source ZIP through a checksummed projection manifest.

The source run must not read any unavailable official target archive, alter the
frozen V4 protocol, or access any held-v8 runtime, target, query, score, barrier,
or outcome artifact.

## Interpretation

A passing source gate authorizes one new pre-target V5 completion candidate for
the five unavailable actions. A failing gate preserves the global scale for
those objects. Only later execution on the exact author-provided targets can
evaluate transfer; those target outcomes may not modify the frozen selector.

# PokeFlex public-action transfer audit v6 result

## Outcome

The frozen object-specific action-robust scale passes the registered broad-transfer
interpretation gate on the 78 previously exposed, non-source public PokeFlex
actions. It improves object-balanced CD-UL1 by **2.40%** over the released
checkpoint and **1.32%** over the fixed global scale `0.125`.

| Retrospective 78 | Released checkpoint | Global `0.125` | Object-specific | Relative to checkpoint | Relative to global |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object-balanced | 5.3977 mm | 5.3387 mm | **5.2682 mm** | **+2.40%** | **+1.32%** |
| Action-balanced | 5.4575 mm | 5.3965 mm | **5.3198 mm** | +2.52% | +1.42% |
| Frame-balanced | 5.5080 mm | 5.4469 mm | **5.3670 mm** | +2.56% | +1.47% |

Against the checkpoint, object means improve for 17/18 objects and tie for one;
the paired object-cluster candidate-minus-reference interval is
`[-0.1724, -0.0880]` mm. Against the global scale, 15 objects improve, two tie,
and one regresses; the interval is `[-0.1035, -0.0404]` mm. The sole object-level
regression is Sponge at `-0.0578%`, within the locked 1% guard. At action level,
the candidate records 71/4/3 wins/ties/losses against the checkpoint and
61/8/9 against the global scale.

All four registered retrospective checks pass for both references: positive
object-balanced improvement, a negative 97.5% object-cluster upper bound, at
least 12 object wins, and no object regression larger than 1%.

## Evidence order

The earlier prospective two-action result remains unchanged and must be read
first when making an advancement claim:

| Prospective two | Released checkpoint | Global `0.125` | Object-specific | Relative to checkpoint | Relative to global |
| --- | ---: | ---: | ---: | ---: | ---: |
| Object-balanced | 8.0557 mm | 8.0332 mm | **7.9830 mm** | **+0.90%** | +0.62% |

The candidate improves both prospective actions over the checkpoint, but only
one of two over the global scale. Its upper object-bootstrap difference against
global is `+0.00356` mm, so the preregistered strict advancement gate failed.
Combining the retrospective and prospective blocks descriptively gives 5.2727 mm
for the candidate, 5.3995 mm for the checkpoint, and 5.3414 mm for global, but
that mixed-evidence summary does not replace either separate result.

## Technical accounting

The completed audit contains 78/78 retained case artifacts and no unsealable
case. There are 74 ordinary predictions and four exact technical fallbacks:
`PlushVolleyball_T1`, `T2`, `T4`, and `T5`. Their released robot records contain
no usable `T_WE` pose for any required source frame. Consequently, all 337 scored
frames from those four takes use the exact checkpoint value. The wrapper uses a
nonphysical in-memory sentinel only to execute the unchanged checkpoint path;
the sentinel never contributes to a prediction, and source `robot_data.json`
bytes remain unchanged.

The source projection and replay workers overlapped. Their bounded retry loop
recorded 102 transient HTTP 404 responses while projected packages were not yet
available. Every one of the 78 packages subsequently downloaded, matched its
registered source/projection checksums, produced a validated artifact, and was
removed from staging. Both workers exited normally.

## Interpretation

This result is strong evidence that the source-calibrated object-specific scale
has broad public-action headroom. It justifies a **new, genuinely fresh,
baseline-relative guarded evaluation**. It does not authorize retuning from these
78 outcomes, overturn the prospective global-scale failure, or establish direct
state of the art.

The public 78-action cohort is not PokeFlex's unavailable official 18-take test
split, so the published 6.498 mm number remains contextual rather than directly
comparable. The next candidate should retain exact fallback and admit the
object-specific correction only when source-calibrated evidence predicts lower
regret than the global baseline. That selector must be locked before selecting or
opening fresh actions or objects.

## Custody

The amended protocol has canonical digest
`4dc9d5d5c45100e09e251ac042fd5fbf7f85b6d45b8dc558d5ce1bce15e9ae23`.
The executed implementation revision is
`c4c80aed799ffa6f18ec96ecb0104f7a16e40d87`. The aggregate has canonical
digest `5cb382596d9e29fa1b1be20c8c40549c5e9c43769c3fab3c6168d5af8f7bcaaa`
and file SHA-256
`2e1e0ff91abd432461b198917d14059da38ec0254727b1ccddb2aa3d0a2a4340`.
The custody summary has canonical digest
`0e56c165d0bacf94ce950c9c636532cf045adea5992b363d7200bfa1e8f7d1f5`
and file SHA-256
`8d6d30afc7773ec49cdf58248396fad10aa47a71176f1f11a297dd62ae7135e1`.

The 376.9 GB original archives stayed on `gpuserver6000`. Compact source
projections moved directly over the server LAN to `gpuserver4090`; the jump
server was not in the payload path. The retained bundle includes all 78 per-take
artifacts, all 78 projection manifests, 80 execution logs, the aggregate, and the
custody summary.

The final clean installed environment passed 1,841 tests with 26 skips. The
current Prob4D dependency was synchronized to commit
`364f216c14f7770c1b360bb1b836b11ecf0c18b8`, and changed-file Ruff and Git
whitespace checks passed.

# DLO-Lab wrapping posterior chance-guard source result v4

## Status

**Terminal technical failure; no prospective task-value result.**

The sole registered attempt ran under frozen revision
`5d5150a794653c212d3a11f086d0ff845a427448`. Its CPU runtime preflight
passed, all eight prefix batches sealed, and the pre-future decision gate
passed before any task future was generated. The primary 0.975 posterior
chance guard made 97,847 nonfixed sensor decisions, differed from continuous
Bayes on 170,175 decisions, selected three distinct actions, and had minimum
registered nonfixed improvement probability `0.9750006376`.

Sixty-nine of the 72 registered all-action future worlds then completed
ordinary native QA. The native process for `future-69` exited with signal 11
before publishing a seal. The parent preserved this as terminal failure
`003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f`.
The failed world has a write-once claim and runtime log but no prediction
bundle or seal. Worlds 70 and 71 were never started.

The attempt has not been retried, the failed world has not been replaced, and
the 69-world subset has not been scored. There is no generation seal or result
artifact. Consequently, neither the primary source gate nor any guard-versus-
continuous-Bayes reward, downside, or harm claim is available from v4.

## Interpretation

The sealed prefix evidence shows that the proposed chance guard is executable
and materially changes the controller on a fresh roster. That is implementation
evidence only. It does not answer whether the guard retains reward or reduces
realized downside because the registered 72-world denominator is incomplete.

The failure is technical rather than a scientific negative result, but the
one-attempt rule is intentional: silently rerunning a near-complete study would
weaken the project's evidence discipline. No fresh successor is automatically
authorized. A future study would require a separately motivated and newly
frozen runtime-hardening protocol, not continuation or replacement of this
attempt.

This result makes no official benchmark, SOTA, real-robot, physical-safety, or
real-world-transfer claim. It uses only public native simulation and no new
recordings.

## Evidence identities

| Artifact | Identity |
| --- | --- |
| Frozen source commit | `5d5150a794653c212d3a11f086d0ff845a427448` |
| Runtime preflight | `095084fae2e256097acc0dafc6dde86ec2c065b36cfe376fcc2ef0e5dc1a60d2` |
| Study attempt | `17f4e1e6f140222b1d9c725c3921861aebae7dc9a59454c8938a3af94ddf43c8` |
| Study lock | `ddd6724de1b0dc3303df6c5aba6906f4c41f8f896f27ed96f70ca24e89da5f18` |
| Decision seal | `198041b3f9f480802e8a529a94a740650b311c9a6e7375bbcf714c9d47238ea8` |
| Decision barrier | `519407e4590ffd61b901683c1a3d0f824e3ce608b15633b2012fecd5b6d07a07` |
| Terminal failure | `003be585e995ad8e38818cbb341fe9d39c8344d2dd8bc59d4bd6ace61945443f` |
| Compact summary | `ef75f43b46654530ed8a788303feee13c36a3d448566041b42707fe898e07873` |

The verification script reconstructs all prefix decisions, the pre-future
barrier, and native QA for all 69 ordinary futures. It also proves that no
complete generation or score exists. This is a second implementation check,
not independent human review.

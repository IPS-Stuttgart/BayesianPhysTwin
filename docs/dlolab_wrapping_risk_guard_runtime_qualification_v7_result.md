# DLO-Lab native-Linux runtime qualification v7 result

## Result

The registered native-Linux qualification passed in its sole attempt:

| Stage | Required | Ordinary successes | Result |
| --- | ---: | ---: | --- |
| Fresh-process constructors | 24 | 24 | pass |
| Complete 2,200-step rollouts | 4 | 4 | pass |

Every constructor reached `init_cmaes_env`, retained finite initial state, and
kept material randomization deferred. Every complete rollout realized the exact
registered bending and stretching values and passed the unchanged v4 native
future QA. The parent process exited zero and published result ID
`a147939df81acd11580f00405ae96a7b198909d00705b79d6636f228af0b0ee7`.

The 114-file evidence tree has canonical ID
`d0b772d13b92d09699e1942128c68d237815a736c54580f7bb2041a8a30c585a`.
The local mirror is checksum-identical to the preserved `workstation2` tree.

## Interpretation

V6 failed after 22 constructor successes with a native `SIGSEGV` inside
Genesis `scene.build` under WSL. V7 kept the public simulator, Python 3.11.15
binary bytes, package versions, OSMesa closure, native adapter, worlds, actions,
and QA fixed while moving execution to native Linux. Passing 24 constructors
and four complete rollouts supports using this exact native-Linux runtime for a
separately frozen scientific study. It does not prove Genesis is universally
stable, nor does it erase the WSL failures.

## Claim boundary

This is runtime qualification, not task-value evidence. V7 defined no fresh
scientific worlds, read no protected data, opened no scientific outcome, and
computed no accuracy, uncertainty, safety, or SOTA metric. The pass authorizes
only a new prospective study with its own worlds, predictions, comparators, and
decision gate. No retry or replacement was used.

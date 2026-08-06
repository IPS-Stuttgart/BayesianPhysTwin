# TAPNext++ Material Transport Provider Source Lock

Date: 2026-08-06

Status: 14/14 fixed cases staged before tracker execution or withheld-prefix
scoring.

## Implementation

- pre-outcome method commit: `d74f6597`
- protocol SHA-256:
  `b4e65b8f073fdda855c3d35d153d8920e61a8ae32237f95621ed90f6ed0d9a08`
- source manifest canonical SHA-256:
  `a3443cc082891c978a740a41981e0b3febd1982043548cff9e27868155b28e8f`
- source manifest file SHA-256:
  `e36f57fae7c8c05eb67796088bc28839f420a9eca58c3fa1f940c0d88c71d84f`

Every case has a separately hashed prediction input, withheld 20-frame prefix,
tracker protocol, physical trajectory, and immutable frame-zero material-node
attachment. All windows end exactly at the released training boundary.

## Aborted Staging Attempt

The first operator invocation used a partial source mirror that lacked
`mask/processed_masks.pkl` for all 14 cases. It produced no prediction and
opened no withheld prefix or future outcome. Its all-failure manifest is
preserved rather than overwritten:

- file SHA-256:
  `8c915bb77dd88a18cf1901e562e5c9e3efb7beccb8aadf26183fa1f41e8dd055`
- canonical SHA-256:
  `408ec81f239ee3f4b47607945e69d82787bf7e73ddaab36e62bf0ba801a43e68`

The successful invocation changed only the raw source root to the complete
released PhysTwin extraction. The implementation commit, protocol, case list,
windows, identities, settings, and gates remained unchanged.

## Boundary

No tracker prediction or provider score existed when this lock was written.
No previous eight-case future outcome and no held-v8 runtime, target, query,
score, barrier, or outcome artifact was read or modified.

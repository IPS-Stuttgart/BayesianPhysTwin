# DLO-Lab Slingshot posterior-aware policy certificate source v2 result

## Decision

**Retained technical negative at the complete-denominator native-QA gate.** The
posterior-aware certificate passed its prefix-only admission gate, but two of
288 registered evaluation future tasks failed native QA. The frozen protocol
requires a complete 288-world denominator, so no partial score, retry,
replacement, policy-value estimate, coverage estimate, or matched-comparator
result is authorized.

## What completed

- Frozen source revision: `fe88fa40f875394a4881ce72b72fec098ac5f469`.
- Calibration: 128/128 prefixes and 128/128 all-action futures passed native QA.
- Evaluation prefixes: 288/288 worlds passed native QA.
- Policy calibration: rank 117/128, offset `0.039885809910921444`.
- Matched simultaneous-regret calibration: rank 117/128, offset
  `0.6982761874352208`.
- Prefix-only admission: 34 accepted, 254 exact fallbacks; the registered
  minimums of 24 each passed.
- Evaluation future claims: 288/288.
- Ordinary evaluation future seals: 286/288.
- Terminal native-QA failures: 2/288, at registered world indices 46 and 261.
- Replacements and retries: 0 and 0.
- `result.json` and partial scientific scoring: absent.

The frozen source was reloaded before repository edits. The lock, calibration,
candidate predictions, guarded decisions, and barrier reproduced; all 286
ordinary future seals revalidated; and the two write-once failure records were
bound to their claims. The compact evidence is
`results/source/dlolab_slingshot_policy_certificate_source_v2/summary.json`.

## Interpretation

There is one useful prospective prefix-only result. The richer posterior-aware
local certificate admitted 34/288 fresh worlds, clearing the locked capacity
gate. The v1 global-offset certificate admitted 12/288 on its separate fresh
panel. This supports the narrower statement that query-local posterior
diagnostics can preserve more selective decision capacity than the rejected v1
construction.

It does not establish that the 34 accepted decisions improved value. Two native
future tasks lacked valid seals, so the complete statistical estimand does not
exist. Scoring the 286 successful futures would silently condition on simulator
success, violate the locked denominator, and understate execution risk.

The failures also expose a protocol-design requirement for any successor:
native admissibility must be determined before outcome access, or a
prospectively specified technical-failure estimand must retain failed worlds.
Neither can be added retroactively to this run.

## Claim boundary

There is no prospective policy-value, paired-bootstrap, harmful-world,
coverage, oracle-headroom, matched-comparator, benchmark, SOTA, or physical
safety result from v2. The exact v2 roster is closed. Its 286 ordinary futures
must not be scored as a subset, and the two failed worlds must not be retried or
replaced.

The established public-simulator evidence remains unchanged: DLO-Lab Wrapping
is the positive query-competence certificate in the atlas; Slingshot remains
unpromoted. A successor needs a new disjoint roster and a newly frozen
technical-failure policy.

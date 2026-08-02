# PokeFlex official-18 public feasibility result

## Outcome

The exact official-18 protocol is **not executable from the public archive as
named**. It is closed before prediction and before any target deformation mesh
was opened. This is a source-contract failure, not a result for or against the
Bayesian-PhysTwin correction.

The upstream evaluator hard-codes one internal recording identifier for each of
18 objects. The paper describes these as a randomly selected validation
sequence per object. The complete public poking download contains 116 archives,
but five hard-coded identifiers have no matching public archive:

| Internal evaluator ID | Highest public poking take |
| --- | ---: |
| `Pillow_T8` | `T7` |
| `3dPrintedCylinder_T7` | `T6` |
| `3dPrintedHeart_T14` | `T6` |
| `Sponge_T10` | `T5` |
| `3dPrintedPizza_T13` | `T6` |

The download completed successfully on 14 July 2026, so this is not an
incomplete local transfer. The public supplementary folder contains pretrained
models and printable assets, but no processed validation set or internal-to-
public take mapping. Guessing a renumbering would destroy exact-split
comparability and is forbidden by the lock's no-replacement rule.

## Custody

- Target meshes opened: **no**
- Prediction seals created: **0/18**
- Prediction barrier created: **no**
- Scoring run: **no**
- Replacement or inferred take mapping: **none**

The locked protocol remains preserved at commit `a35c081` with canonical hash
`cb1a7270362fe102e3deb4b589f5f4dd67268ba2b7bbee13fa4dbd9c9ad3ab97`.
The upstream evaluator hash is
`ea1854ba5224b8aec2e8ba6b80fb762eba7314b925e87ca7775d810003615b60`.
The completed public downloader used for the availability audit has hash
`ae13a9acd2ba02d46fb340c15edba687929e6f84debc01c7792d6bf41859ed11`.

## Decision

Do not run or amend `official18-v1` using guessed public take identities. A
direct reproduction of the published 6.498 mm aggregate requires either:

1. an author-supplied mapping from the internal validation IDs to public
   archives; or
2. the authors' processed validation set.

Without one of those, the defensible public-data claim remains the prospective
paired result: the frozen correction improved the released checkpoint by
1.046% object-balanced CD on eight locked `T2` objects, with seven wins and one
exact fallback. A future public benchmark should compare candidate and released
checkpoint on the same preregistered public takes and should not call that split
identical to the paper's internal validation split.

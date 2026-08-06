# PokeFlex same-object paper artifacts

## Purpose

This workflow turns the already completed PokeFlex independent-depth regret-guard
experiment into one reviewable calibration figure and one clear MP4 without
changing the estimator, source certificate, prospective protocol, candidate
bank, acceptance threshold, or evaluation cohort.

The claim is deliberately narrow:

> A source-calibrated independent-depth regret guard improved the released
> PokeFlex checkpoint on three prospective new takes of two previously seen
> objects while returning the released checkpoint exactly when it abstained.

This is a **same-object temporal-transfer** result. It is not independent-object
generalization, a target-object result, general deployment calibration, or a
state-of-the-art claim.

## Frozen result

The preregistered prospective panel contains `FoamDice_T7`, `FoamDice_T8`, and
`PlushOctopus_T7`. It was evaluated once with the source-fitted certificate and
selector correction from
`pokeflex-independent-depth-regret-guard-prospective-v1`.

| Quantity | Result |
| --- | ---: |
| Released checkpoint, object-balanced CD_UL1 | 6.418 mm |
| Regret-guarded output, object-balanced CD_UL1 | 6.222 mm |
| Relative reduction | 3.06% |
| Object wins / losses | 2 / 0 |
| Accepted updates | 87 / 241 frames |
| Exact checkpoint fallbacks | 154 / 241 frames |
| Accepted-frame improvements / regressions | 77 / 10 |

All three take means improve, although the effect is heterogeneous: 0.30% on
`FoamDice_T7`, 7.81% on `FoamDice_T8`, and 2.19% on
`PlushOctopus_T7`.

The later four-object calibration evaluation failed: object-balanced CD_UL1
changed from 4.817 to 4.872 mm (+1.16%), only two of four objects improved, and
`3dPrintedPyramid` regressed by 19.36%. The same-object result therefore must not
be generalized to unseen objects.

## Calibration figure

`scripts/paper/make_pokeflex_same_object_figure.py` reconstructs the candidate
that the frozen certificate would rank first before fallback. It verifies every
stored bound, arm decision, and deployed error against the committed prospective
result. Only then does it open the candidate outcome for a post-outcome
visualization.

The left panel plots selector-adjusted upper regret against realized candidate
regret. The vertical zero line is the frozen acceptance threshold. The four
quadrants distinguish accepted improvements, harmful accepted updates,
conservative fallbacks, and harmful updates prevented by fallback. The right
panel reports the three prospective take means and the object-balanced result.

The plot is a diagnostic of an already frozen decision rule. It is not used to
retune the threshold or fit a new certificate.

## Video

`scripts/paper/render_pokeflex_same_object_video.py` chooses the prospective take
with the largest take-level improvement (`FoamDice_T8`) unless an explicit take
is supplied. This is an illustrative, post-outcome selection and not another
statistical test unit.

The renderer:

1. verifies the frozen candidate-runner SHA-256 from the prospective protocol;
2. reruns the unchanged candidate computation on the selected opened take;
3. verifies every reproduced baseline and deployed error against the committed
   prospective evaluation;
4. captures the volumetric target, released prediction, and exact guarded output
   without modifying the hash-locked runner;
5. uses a fixed deterministic projection and common target-distance color scale;
6. overlays the complete error trace, accepted-update markers, frozen upper
   regret, and exact-fallback status; and
7. encodes one 1920x1080 H.264 MP4 at 30 frames per second.

The video is explanatory evidence. The quantitative claim remains the complete
three-take, two-object result.

## GitHub Actions boundary

`.github/workflows/pokeflex-same-object-paper-artifacts.yml` has two jobs:

- a GitHub-hosted contract job runs Ruff, formatting, MyPy, focused tests, and
  committed-result validation;
- the `workstation2` job runs on `[self-hosted, Linux, X64, nvidia-smi]`, stages
  the original content-addressed candidate artifacts when available, reproduces
  any missing artifact without refitting, builds the figure, reruns the visual
  exemplar, and uploads the MP4, poster, PDF, PNG, compact JSON, and checksums.

The workflow has `contents: read`, uses `persist-credentials: false`, never
rewrites the pull-request branch, and leaves large data and rendered media in the
Actions artifact rather than the source tree.

Required repository variables are the existing `POKEFLEX_DATA_ROOT`,
`POKEFLEX_UPSTREAM_CHECKOUT`, and `POKEFLEX_CHECKPOINT_ROOT`. Optional variables
are `POKEFLEX_PYTHON` and `POKEFLEX_PROSPECTIVE_ARTIFACT_ROOT`. When the latter is
unset, the workflow checks the original retained result location and regenerates
missing candidate artifacts from the already opened prospective takes.

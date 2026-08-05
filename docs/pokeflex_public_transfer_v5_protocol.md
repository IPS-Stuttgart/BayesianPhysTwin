# PokeFlex public-transfer v5 protocol

## Purpose

This study tests whether the frozen action-robust Bayesian-PhysTwin update transfers
beyond its 36 source-calibration actions. It does not reconstruct the published
18-take validation split: five legacy take identifiers still have no authoritative
mapping into the 116-take public release.

## Evidence order

1. Freeze code, calibration, cohort, ZIP hashes, metrics, and gates.
2. Seal predictions for the final two previously unscored public takes:
   `Pillow_T4` and `PlushDice_T3`.
3. Open target meshes only after both prediction seals pass one registered barrier.
4. Score the two-take prospective panel without replacement or adaptation.
5. Only afterward, audit the other 78 non-source public takes retrospectively.
6. Report the 2 prospective and 78 retrospective takes separately before reporting
   the combined 80-take public-release transfer summary.

The 36 source actions are excluded from every transfer aggregate. A technical
failure remains in its registered cohort and uses the frozen exact-checkpoint
fallback where the prediction contract permits it.

## Frozen method

The physical prior is the released PokeFlex Kinect checkpoint. The candidate uses
the existing action-local graph state correction with a base scale of `0.125` and
the source-calibrated per-object multiplier. The final-two runner remains bound to
the v3 calibration (`Pillow=2`, `PlushDice=4`); the all-18 v4 extension is
corroborating source lineage and preserves these rows exactly. The global `0.125`
arm and byte-identical released checkpoint are retained as paired controls.

No target outcome, target mesh, or later observation may select a multiplier,
change support, or alter the update. Unsupported frames return the released
checkpoint bit for bit.

## Prospective gate

The final-two arm advances only if all of the following hold:

- both physical objects improve over the released checkpoint in mean CD-UL1;
- both improve over the global `0.125` correction;
- the object-balanced mean improvement is positive against both references;
- the 97.5% upper paired object-bootstrap difference is below zero against both;
- no take is replaced and no target-dependent retry or tuning occurs.

With two objects, this gate is deliberately strict and descriptive rather than a
claim of population-wide significance.

## Public-release transfer audit

After the prospective result is immutable, the 78 remaining non-source takes may
be scored with the all-18 v4 multiplier map. The primary retrospective summaries
are equal-weight physical-object CD-UL1, action-balanced CD-UL1, paired wins/ties/
losses, and a physical-object cluster bootstrap. Results must include the released
checkpoint and global-scale controls.

The combined 80-take result is considered evidence of broad public-release
superiority only when the prospective gate passes and the all-80 object-balanced
candidate beats both controls with a cluster-bootstrap upper difference below
zero. It remains a public-release transfer result, not a direct published-SOTA
comparison, until the five unavailable official records receive an authoritative
mapping or processed evaluation artifacts.

## Boundaries

- The two objects are not unseen; only their exact actions are prospectively held.
- Published `6.498 mm` Kinect CD-UL1 is contextual because the cohort differs.
- Jaccard is diagnostic because released meshes are not guaranteed volumetric.
- held-v8 artifacts, identities, outcomes, barriers, and processes remain outside
  this study's authority.

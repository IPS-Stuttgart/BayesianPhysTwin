# Deform360 `001-rope` public held-out pilot

This milestone evaluates one same-object held-out action without waiting for
gated PokeFlex access. The source/calibration/target split was fixed from
metadata before geometry, tactile values, or prediction errors were read.

## Protocol

- Source episodes: `0,2,3,4,5,7,8`.
- Calibration episodes: `1,9`.
- Target episode: `6`, action `move both edges`.
- Target prefix: frames `[103,109)`.
- Target future: frames `[109,237)`.
- Primary metric: symmetric 3D centerline Chamfer distance.
- Secondary metric: ordered normalized-arc-length pseudo-track error.

Episodes `2` and `7` failed source-only contact/geometry registration gates and
were excluded before forward fitting. The five accepted source actions selected
one inextensible forward model from a 200-candidate finite grid. It improved
pooled source Chamfer by 25.17% over persistence and improved four of five
leave-one-action-out episodes.

## Held-Out Result

| Method | Future CD (mm) | Future track (mm) | CD vs persistence |
| --- | ---: | ---: | ---: |
| Constant persistence | 71.84 | 78.87 | 0.00% |
| Visual-only contact | 47.58 | 60.16 | -33.77% |
| Six-frame tactile-conditioned `z` | 59.74 | 69.67 | -16.84% |
| Full-tactile oracle | 46.70 | 59.80 | -34.99% |

A source-pooled physical forward model transfers to the held-out action and
beats persistence. This comparison establishes forward-model competence, not
that pooling caused the gain: a matched single-source fitting control was not
run in this pilot. The backend is a reduced inextensible centerline simulator,
not PhysTwin/Warp.

The six-frame tactile posterior is worse than visual-only contact because
gripper 1 is inactive at prefix frame 108 but its oracle contact starts at frame
109. The frozen open-loop policy holds that inactive state until a visual
transition, so it misses the future onset. Visual-only contact is within 1.84%
CD and 0.60% track error of the full-tactile oracle.

This is a negative result for a static prefix-conditioned intervention state,
not for tactile sensing generally. Online tactile filtering or a learned
contact-transition model remains untested.

## Claim Boundary

- One object and one held-out target action; no population-level inference.
- Constant persistence is a competence baseline; the missing single-source
  control prevents attributing the target gain specifically to pooling.
- The fitted backend is not PhysTwin/Warp or Bayesian-PhysTwin.
- Public SAM2 fallback, not Deform360's gated SAM3 mask stage.
- Centerline nodes are silhouette-derived normalized-arc-length
  pseudo-correspondences, not verified material tracks.
- Released tactile is unitless normal response, not calibrated force or slip.
- Released robot motion is vision-recovered/measured action conditioning, not a
  separately logged command trajectory.
- The full target tactile oracle and suffix geometry were opened only after the
  deployable prediction seal `add9b28154159f71e9bd7d631d68cbfd73e0b63ba8a487cedace5bead48ec667`.

Compact, checksummed artifacts are under `artifacts/`. Large masks and the raw
dataset remain on `gpuserver6000` under
`/mnt/lexar4tb/datasets/deform360/results`.

## Verification

The captured implementation passed on 2026-07-14:

```text
ruff format --check: 54 files already formatted
ruff check: all checks passed
git diff --check: passed
pytest: 381 passed, 3 skipped in 383.01 s
artifact manifest: 12/12 file hashes matched
```

See `verification.json` for the machine-readable record. The code and paper
changes remain uncommitted in the local worktrees at capture time.

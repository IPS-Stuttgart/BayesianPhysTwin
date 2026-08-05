# TrackDeform3D public smoke v1 result

## Scope

This is an interface-and-capacity result on the public TrackDeform3D sample,
not a benchmark or state-of-the-art result. The upstream tracker trajectory is
a future-RGB-D-derived pseudo-observation rather than independent material-point
ground truth. The locked upstream revision has no license file, only one DLO
chunk is used, and the physical prior is the frozen inextensible graph smoke
model rather than Warp/PhysTwin.

The implementation and protocol were committed before validation scoring at
`78403de24f917d357c530d31ed0b3ec30c6e8dc6`. The upstream revision is
`9920060a76f7d750f98e429bd1e0f172150c9ffa`. Clip zero was explicitly open for
development. Clip one was prepared into disjoint prediction and evaluator
carriers; its prediction was sealed before the evaluator carrier was scored.

## Frozen method

The physical prior assigns the two chain endpoints to the two released robot
end effectors using frame-zero geometry, transfers known endpoint motion along
graph geodesics, and projects all edges to their frame-zero rest lengths. The
Bayesian arm represents residuals in four graph-Laplacian modes and regresses
their coefficients on the known left/right action displacement and frame
velocity.

Frames 0--39 fit the candidate and frames 40--59 gate it. Admission requires at
least 10% prefix-validation RMSE improvement and no last-five-frame regression.
Rejection is bit-exact physical fallback. The prediction function cannot accept
the hidden future carrier, and only identities disjoint from the four observed
prefix identities are scored.

## Results

All values below are millimetres; lower is better.

| Split | Arm | Hidden RMSE | Hidden FDE | Hidden Chamfer |
|---|---|---:|---:|---:|
| Development clip 0 | Frame-zero persistence | 288.987 | 361.944 | 226.452 |
| Development clip 0 | Sparse constant velocity | 103.177 | 164.359 | 74.169 |
| Development clip 0 | Known-action graph prior | 51.474 | 62.386 | 46.403 |
| Development clip 0 | Guarded Bayesian discrepancy | **45.626** | **60.414** | **39.584** |
| Validation clip 1 | Frame-zero persistence | 173.038 | 181.511 | 140.526 |
| Validation clip 1 | Sparse constant velocity | 108.718 | 158.577 | 85.352 |
| Validation clip 1 | Known-action graph prior | **36.926** | **38.245** | **35.713** |
| Validation clip 1 | Guarded Bayesian discrepancy | 53.024 | 82.734 | 37.786 |

On development clip zero, the guarded correction improves hidden RMSE by
11.36% over the physical prior. On untouched validation clip one, it regresses
hidden RMSE by 43.60% and FDE by 116.33%. Its early/middle/late validation RMSE
is 27.530/47.095/73.883 mm, compared with 36.353/36.854/37.561 mm for the
physical prior. The correction helps briefly and then diverges as the future
action-response regime leaves the prefix fit.

Both prefix gates admitted the candidate. Validation-prefix RMSE improved by
63.64% on clip zero and 59.37% on clip one, so within-prefix transfer was not a
sufficient future-safety gate. Uncertainty also fails as a useful calibrated
object: development coverage is 66.26% at nominal 90% with mean NEES 7.75;
validation coverage is 99.75% with mean NEES 0.34 after a 6.54-fold variance
inflation. The latter is conservative but far too broad.

## Decision

The action-conditioned Bayesian discrepancy arm is rejected. Do not tune it on
clip one and do not run a larger preregistered evaluation of this exact method.
The known-action graph prior itself transfers across the two clips and strongly
beats both kinematic controls, but that is only capacity evidence against an
upstream pseudo-observation.

The next credible update must learn a baseline-relative future-regret guard
across independent source episodes or objects, not from a temporal split inside
the same prefix. It must preserve the unchanged physical prior exactly when the
source-calibrated upper confidence bound cannot certify improvement. A
claim-bearing TrackDeform3D study additionally requires explicit reuse terms,
the full object cohort, and independent ground truth or a clearly scoped
observation-prediction claim.

## Evidence

- Admission manifest file SHA-256:
  `12eb82de428f3c2ba129e64be5ed9e3216c72d672c653568b33c7c52882a2c3c`
- Development prediction SHA-256:
  `ce72a9b7feec644f19b17d2a4562d0e08953b88dc9843ebd3cf7e287443a0e76`
- Development result file SHA-256:
  `cf6391fb8c2b57be905d531a3950e4949eaf4043633ade80a81727ac316f66b3`
- Validation prediction SHA-256:
  `7df7628cd8869b3aa3fb2fddd909b031de33006f327f7d6cd54c0e1715e08c38`
- Validation result file SHA-256:
  `dd1ca45bc5bbde87523be88fe475b0dcaf3ceb26aa62fcffee7a73e1337e441a`
- Validation canonical result SHA-256:
  `43ac125fa11a0c2248d6a4f296fabb43576706c44f16cd87d0bce689c9d206ae`

Compact carrier manifests, prediction manifests, and result JSON files are in
`results/sota/trackdeform3d_public_smoke_v1/`. Full NPZ carriers remain in the
durable source-only server archive.

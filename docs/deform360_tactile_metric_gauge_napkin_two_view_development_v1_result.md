# Napkin conservative two-view metric-gauge result

## Result

The frozen development-only two-view gauge failed with zero admitted cameras.
Both cameras had complete provider support for all 17 tactile contact rows over
all six source frames, so this is not a visibility or missing-data failure.

| Camera | Assignment | Median | p90 | Maximum | Decision |
|---|---:|---:|---:|---:|---|
| `019_cam1` | direct | 3.10 mm | 81.80 mm | 88.52 mm | fallback |
| `019_cam1` | swapped | 3.85 mm | 188.01 mm | 200.50 mm | fallback |
| `022_cam1` | direct | 7.64 mm | 81.31 mm | 84.63 mm | fallback |
| `022_cam1` | swapped | 5.90 mm | 46.89 mm | 48.88 mm | fallback |

The frozen limits were 5 mm median and 15 mm p90 for both assignments in both
cameras. Carrier admission was therefore not run, and no prediction score was
opened.

## Interpretation

Reducing the required panel from three cameras to two does not rescue this
MotionCrafter metric-gauge path. The failure occurs within each camera before
cross-view fusion: sparse median residuals can be small while complete held
frames show 47--188 mm tails. The conservative covariance union correctly
prevents correlated cameras from manufacturing confidence, but covariance
inflation cannot repair an inconsistent point-map mean.

This closes the frozen combination of:

- decoded-uniform MotionCrafter point maps;
- tactile taxel trajectories as sparse metric-gauge evidence;
- one robust similarity transform per camera and assignment;
- complete-frame held-prefix validation;
- two-view unknown-correlation covariance union.

It does not reject learned visual observations generally. It says this global
similarity-gauge interface is not stable enough to admit an object-state update
on the development case. A successor would need a different observation
interface or independent metric modality, not a weaker camera-count or error
gate.

## Boundary and provenance

- lock ID: `3faf5199e58f9f9d2b005887b4efd3dc215433eff23ff3f53cb1312cdc632be6`;
- result artifact ID: `ee86d827c4ecee90630ef013306e5e9b3a3c38899fc7912691056f6c17bcda11`;
- result SHA-256: `9274c5278d9e3b18945d342b14838393a14ca4f7e4986a23cf383fcafd25aad6`.

The napkin geometry was already open and motivated this method class, so this
is development evidence only. Calibration scores, future frames, confirmation
payloads, target outcomes, and all held-v8 artifacts remained unopened.

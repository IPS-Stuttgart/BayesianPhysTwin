# Deform360 tactile-prompted carrier napkin validation result

## Result

The independent calibration-object source gate selected exact fallback before
opening any MotionCrafter array. Robot recovery and tactile geometry passed,
but only two of the three preregistered carrier cameras satisfied the frozen
contact-visibility requirements.

| Stage | Result | Key evidence |
| --- | --- | --- |
| Robot source preflight v1 | Technical failure | Three carrier cameras did not equal the 32-camera robot panel; zero frames decoded |
| Robot prefix v2 | Pass | Both grippers supported; max step 2.30 mm / 0.65 degrees |
| Tactile geometry | Pass | 17 active taxels over 6/6 contact frames; assignment separation at least 123.80 mm |
| Fixed carrier-camera visibility | Fail | 2/3 cameras meet full assignment coverage and 64 px margin |

The failing fixed camera, `brics-odroid-010_cam0`, had minimum assignment
coverage `0.4706` and minimum target-image margin `19.76` px. The other two
cameras passed:

- `brics-odroid-019_cam1`: coverage `1.0`, margin `75.43` px;
- `brics-odroid-022_cam1`: coverage `1.0`, margin `72.19` px.

The protocol forbids camera substitution after seeing these diagnostics. It
therefore stopped before metric-gauge fitting, SAM2 inference, carrier
construction, state update, or prediction scoring.

## Meaning

The source-only result identifies another concrete hard-three-view failure
surface. The tactile evidence and robot state are valid, and two geometrically
distinct cameras see both assignment hypotheses, but requiring every member
of a preselected three-camera provider panel rejects the case.

This does not authorize weakening the gate on `036-napkin-cloth`. It motivates
a new protocol, frozen on a fresh calibration object, in which two-view gauge
evidence is admitted only with explicit covariance inflation and a shared-bias
nuisance. Such a protocol must retain exact fallback and cannot claim that two
views provide three independent measurements.

## Boundary

- calibration prediction scores remained unopened;
- MotionCrafter NPZ members for this case remained unopened;
- no future camera or tactile frames were used;
- no confirmation or target payload was opened;
- no held-v8 artifact was accessed;
- no state update or SOTA claim is authorized.

The admitted robot artifact ID is
`a2601582c5dd712d2adef4b599e3fd909dc39204e4b4fa7aa01be73c45b07215`.
The admitted tactile artifact ID is
`ebdbee93c38254758c3c53fe02de3b85256f554ed1d27d684d5f4ea346fc3e39`.

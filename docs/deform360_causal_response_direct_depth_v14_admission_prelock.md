# V14 Source Admission Pre-Lock

## Purpose

The source panel is selected only after a frame-zero physical carrier and an
adaptive multiview carrier exist. This child protocol binds the executable
path from those pre-lock artifacts to one hash-only source disposition.

## Count Boundary

The causal object-facing camera prefix has exactly 58 frames. Robot, tactile,
and physical prediction have a 76-frame predictive horizon. A camera is
complete when its exact prefix depth and mask streams and calibration are
present; projected physical-seed support is recorded separately and is judged
by the adaptive carrier rather than being folded into stream completeness.

This distinction prevents two invalid shortcuts:

- claiming 76 camera masks or depths that were never lawfully created; and
- treating absence of a projected seed as though the camera stream itself were
  missing.

## Sealed Disposition

For each immutable queue rank, the runner emits:

- the frozen adaptive carrier and its numeric archive;
- a typed V14 source preflight;
- one outer admission report binding the physical, geometry, carrier, source,
  and implementation hashes.

The output retains queue rank plus object and case hashes, but no plaintext
object or episode identity. An admitted strict or inflated carrier may enter
the source panel. An abstained carrier is a pre-lock rejection and may advance
to the next queue rank.

## Information Boundary

Admission uses object observation frame zero only. It does not use prefix
response, future RGB-D, tactile values beyond stream custody, identities,
Chamfer targets, manual trajectories, source outcomes, target artifacts, or
held-v8 artifacts and processes.

# Deform360 same-object rope/cable pilot v1

This pilot uses the existing raw Deform360 holdings on `gpuserver6000` and treats the physical object as the independent unit.

## Registered objects

- `001-rope`
- `002-rope-silk`
- `003-cable`
- `081-stripe-rope`

The source and target episodes must belong to the same object and be distinct. Pair selection prefers different released action labels, then stronger shared camera and tactile coverage. One primary ordered pair is selected per object.

## Stage-1 boundary

The current workflow reads directory entries, file names, exact MP4/TXT and NPY/TXT stem pairing, and allowlisted JSON metadata no larger than 1 MiB. It does not decode media, load numeric arrays, hash large payloads, open target futures, or score predictions.

A successful Stage 1 establishes only a deterministic four-object source-target roster. Before target scoring, Stage 2 must freeze:

1. the metric geometry and action adapter;
2. the physical hypothesis and parameter bank;
3. source-only fitting and uncertainty construction;
4. global-physics, deterministic-identification, residual-persistence, wrong-object, action-misalignment, phase-shift, and identity-permutation comparators;
5. the prediction-seal format; and
6. object-balanced proper scores and harm accounting.

Raw objects and the processed-only `004-rubber-band` object are not pooled under this protocol. Single-episode objects do not count as source-to-target test units.

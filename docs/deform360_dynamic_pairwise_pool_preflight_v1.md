# Dynamic Pairwise Pool: Target-Free Preflight V1

This operational preflight is subordinate to the frozen
`deform360-dynamic-pairwise-belief-open27-v1-development` source protocol. It
does not change the candidate, thresholds, source panel, or advancement gates.

Before AllTracker or source-outcome evaluation, every Open27 case must provide:

- the separately sealed frame-zero physical geometry;
- at least 64 material points supported by the frozen frame-zero mask/depth
  and triangulation-angle rules;
- exactly eight selectable calibrated cameras; and
- every camera/calibration file needed for later causal prefix tracking.

The preflight reads calibration plus HDF5 index zero only. It does not decode
RGB, read a future reconstruction slice, open `target_data.pkl`, or open
`outcome.json`. Failure in any case closes this exact 64-point source arm; the
pool size and camera count may not be relaxed after seeing the record.

On a pass, `pool_preflight.json` contains the deterministic center IDs, selected
cameras, frame-zero slice hashes, and a minimal relative-path staging list. The
camera payload may then be copied to the tracker host without moving unused
views. A pass establishes operational feasibility only, not tracker competence
or predictive improvement.

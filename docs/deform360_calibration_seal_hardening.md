# Deform360 calibration seal hardening

The Stage-1 Deform360 calibration sealer treats its provenance inputs as independently verifiable evidence rather than trusted declarations.

Before publishing a confirmation-opening token, the implementation now:

- recomputes the Stage-0 selection, content-selection, and complete selection-artifact identities;
- verifies the Stage-0 protocol identity from the exact protocol bytes;
- requires the supplied Stage-0 selection to be byte-identical to the committed reviewed lock;
- requires every selected calibration artifact to name the same implementation revision as the calibration bundle;
- compares the imported sealer modules byte-for-byte with the clean reviewed checkout; and
- persists visual-provider and visual-calibration locks atomically without overwriting an existing lock by default.

These checks prevent a structurally valid substitute cohort, stale calibration output, or substituted runtime package from receiving a token under an unrelated reviewed revision. They establish provenance and information-order guarantees only; they do not establish empirical accuracy, calibration quality, provider competence, or deployment safety.

# Deform360 causal tactile feature extraction

## Purpose

The tactile regret guard must consume the same causal raw-taxel representation
on source and prospective cases. `deform360_tactile_features.py` locates the raw
four-sensor segment and the preceding baseline, aligns samples to the common
camera timeline, clears the released invalid final taxel column, and computes
features only through update frames 19, 38, and 57.

The extractor never uses episode-wide peak normalization. It may hash the full
raw payload for custody, but no tactile value after the latest permitted update
enters a feature. Source-reproduction and prospective artifacts carry distinct
boundary flags.

## Source reproduction

The implementation was exercised against all 27 already-open Open27 cases on
`gpuserver4090`. Every one of the 27 cases was reconstructed from the raw
tactile arrays and aligned camera timestamps. Across all three updates and all
13 guard features, the maximum absolute difference from the locked source
artifact was exactly zero.

This is an implementation check, not new predictive evidence. No target outcome
or held-v8 artifact was read.

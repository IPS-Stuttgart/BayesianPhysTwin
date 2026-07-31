# DEFORM fresh DLO2 source protocol v2

This protocol keeps the v1 DLO2 split, training schedule, physical
initialization, and source gates unchanged. It adds the preregistered posterior
predictive median and closes a release-chain gap: fresh DLO2 access now requires
both the successful frozen DLO1 long-run result and the successful, checksummed
DLO1 posterior-transfer result.

The wrapper verifies that the posterior result did not fall back, passed its
1%/five-of-eight source-transfer gate, contains a source calibration record,
and agrees exactly with its immutable selection seal and v2 posterior protocol.
The generic source runner also refuses DLO2 without the wrapper's stage-
authorization artifact, and the resulting source record binds that artifact.
No DLO2 trajectory is read before those checks complete.

```bash
python scripts/remote/run_deform_dlo2_fresh.py \
  --protocol configs/sota/deform_dlo2_fresh_v2.json \
  --parent-longrun-result /path/to/longrun_result.json \
  --parent-posterior-result /path/to/posterior_result.json \
  --upstream-root /path/to/DEFORM \
  --output-root /new/empty/dlo2-source-output \
  --device cuda:0 \
  --mode run
```

The DLO2 source result remains a source-stage result. Official evaluation stays
closed until the independent DLO2 posterior gate and all-training refit both
authorize it.

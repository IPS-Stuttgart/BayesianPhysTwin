# Deform360 v6 runtime dependency-scope repair

Date: **2026-08-12**
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Retained failure

Protected-main source run `31529240660` installed the declared Deform360
`processing` extra, including Nerfstudio `1.1.5` and gsplat `1.4.0`, but the
unqualified whole-environment `pip check` stopped the runtime before prediction
generation. The sole reported conflict was:

```text
pyrecest 2.4.3 has requirement numpy<2.5,>=2.0, but you have numpy 1.26.4.
```

The compact receipt retained `0/10` physical manifests and `0/100` source
prediction seals. Every information-boundary flag remained false.

## Why this conflict is outside the source candidate

The resolved Nerfstudio closure contains `nuscenes-devkit==1.2.0`, whose
released metadata requires NumPy `<2.0.0`. The processing runtime therefore
uses NumPy `1.26.4`.
The self-hosted runner also exposes PyRecEst `2.4.3` through inherited system
site-packages, and that unrelated distribution requires NumPy `>=2.0`.

The frozen candidate installs BayesianPhysTwin with only the `graph` and
`vision` extras. It does not install the optional `pyrecest` extra, and this
source execution path does not execute PyRecEst.

This repair does not change the locked plan's use of precomputed Prob4D source
artifacts; it only scopes the unused PyRecEst distribution inherited from the
shared runner environment.

## Repair

The runtime still executes the complete `pip check`. It now requires that check
to return exactly one line and exactly the known nonzero status above. The
runtime also verifies the NumPy, PyRecEst, Nerfstudio, nuscenes-devkit, and
gsplat distribution versions. Any additional, missing, or changed conflict
remains a terminal technical failure. The exact exception and its inactive or
active status are recorded in every compact execution receipt.

This is narrower than dropping dependency validation: all non-allowlisted
dependency conflicts remain forbidden.

## Frozen scope

The repair changes no data, selected object, camera panel, RGB frame, selector,
SAM2 model, physical algorithm, reconstruction settings, candidate roster,
loss, gate, fallback, suffix policy, or target policy. It authorizes only one
new protected-main source execution after review. Ten physical manifests and
100 immutable source prediction seals remain mandatory before any development
suffix can be opened.

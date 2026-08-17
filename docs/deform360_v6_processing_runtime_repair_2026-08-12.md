# Deform360 v6 processing runtime repair

Date: **2026-08-12**
Status: **pre-source-prediction technical repair; all suffix, confirmation, and target data remain closed**

## Retained failure

Protected-main source run `31526029495` activated the frozen selector API
adapter and reached frame-zero reconstruction. The exact Deform360 processing
source then failed while importing `nerfstudio.configs.method_configs`. The
workflow had installed the base Deform360 package and a hand-selected subset
of processing dependencies, but not Deform360's declared `processing` extra.

The retained compact evidence is:

| Item | Value |
| --- | --- |
| Source revision | `5c8a121580ddff2d2c03ddbe6a75b1090b713dbf` |
| Workflow run | `31526029495`, attempt `1` |
| Artifact ID | `9115032023` |
| Artifact digest | `sha256:6d24e7d63b4b6b9ae977fc048a9aee21cfefcabb86ff78730777ffd4161620f7` |
| Receipt ID | `8c7f87d82915dc83a89db6ef2cfab365d83c6b4ef2a186dd5b60a56bccbc0f8d` |
| Physical manifests | `0/10` |
| Source prediction seals | `0/100` |

Every information-boundary flag remained false. This is technical readiness
evidence only.

## Repair

The workflow now installs the exact frozen Deform360 checkout with its declared
`processing` extra. It binds the checkout's `pyproject.toml` by SHA-256 and
checks the dependency surface expected by the frozen reconstruction code:

- Nerfstudio `1.1.5`;
- gsplat `1.4.0`;
- the `splatfacto` method configuration; and
- the Gaussian-splat exporter.

The separate `requirements-processing.txt` contains unrelated git-pinned model
providers and remains unused. The runtime dependency identity is added to every
source execution receipt produced by the launcher.

The amendment is
`protocols/amendments/deform360_official_hub_fresh_object_session_v6_processing_runtime.json`.

## Frozen scope

The repair changes no data, selected object, camera panel, RGB frame, selector,
SAM2 model, physical algorithm, reconstruction settings, candidate roster,
loss, gate, fallback, suffix policy, or target policy. It authorizes only one
new protected-main source execution after review. Ten physical manifests and
100 immutable source prediction seals remain mandatory before any development
suffix can be opened.

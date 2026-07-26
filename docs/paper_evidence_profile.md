# Paper-evidence profile for RunManifestV2

`RunManifestV2` remains unchanged. Paper-facing runs can add the reserved
`information_boundary.paper_evidence_bindings_v1` profile to make the semantics
of already content-addressed artifacts explicit.

The profile binds:

- the exact provider-manifest artifact ID;
- the resolved Prob4D causal-stream contract version and whether it was declared
  by the producer or inferred for a frozen transitional artifact;
- the exact observation-belief and `TwinBelief` artifact IDs;
- one wheel and one source-distribution digest for the primary executable
  project; and
- the named run-manifest artifact records carrying those bytes.

Because the complete profile is inside `information_boundary`, it is covered by
the existing `evidence_fingerprint`. The V2 schema and all previously written V2
manifests remain valid and byte-identical.

## Profile JSON

```json
{
  "schema_name": "bayesian_phystwin.paper_evidence_bindings",
  "schema_version": 1,
  "primary_distribution_project": "bayesian-phystwin",
  "provider_manifest": {
    "artifact_name": "provider_manifest",
    "artifact_id": "<sha256>",
    "role": "input"
  },
  "prob4d_stream_contract": {
    "version": 2,
    "resolution": "declared"
  },
  "observation_belief": {
    "artifact_name": "observation_belief",
    "artifact_id": "<sha256>",
    "role": "input"
  },
  "twin_belief": {
    "artifact_name": "twin_belief",
    "artifact_id": "<sha256>",
    "role": "output"
  },
  "distributions": [
    {
      "project": "bayesian-phystwin",
      "kind": "wheel",
      "artifact_name": "bayesian_phystwin_wheel",
      "artifact_id": "<sha256>"
    },
    {
      "project": "bayesian-phystwin",
      "kind": "sdist",
      "artifact_name": "bayesian_phystwin_sdist",
      "artifact_id": "<sha256>"
    }
  ]
}
```

Use `"resolution": "inferred"` only for a frozen artifact whose covariance
semantics determine the version unambiguously. Use
`{"version": null, "resolution": "not_applicable"}` when Prob4D is not part of
the run.

## CLI

Pass the profile when creating a manifest:

```bash
bpt run manifest create runs/example/manifest.json \
  --paper-evidence-json runs/example/paper-evidence.json \
  --input provider_manifest=provider-manifest.json \
  --input observation_belief=observation-belief.npz \
  --input bayesian_phystwin_wheel=dist/bayesian_phystwin.whl \
  --input bayesian_phystwin_sdist=dist/bayesian_phystwin.tar.gz \
  --output-artifact twin_belief=twin-belief.json \
  --claim-id bpt.example \
  --method-freeze-id method-v1 \
  --protocol-id protocol-v1 \
  --split-id split-v1 \
  --baseline-id baseline-v1
```

The command rejects digest, role, or artifact-name disagreement before writing
the manifest. Validation automatically checks an embedded profile; use
`--require-paper-evidence` to reject manifests without one:

```bash
bpt run manifest validate runs/example/manifest.json \
  --artifact-root runs/example \
  --require-paper-evidence
```

Paper-facing validation also rejects dirty participating repositories and
requires nonempty claim, method-freeze, protocol, split, and baseline
identifiers.

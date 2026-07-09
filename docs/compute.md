# Compute Notes

Heavy experiments should run on the configured GPU servers:

- `gpuserver6000`
- `gpuserver4090`

Both hosts are expected to be reachable through the jumpserver via SSH config:

```bash
ssh gpuserver6000
ssh gpuserver4090
```

## Run Conventions

- Keep raw run folders on the compute server under `runs/`.
- Keep datasets and PhysTwin checkpoints outside git under `data/` or
  `checkpoints/`.
- Copy only curated plots, tables, and compact result summaries into the paper
  repository.
- Record the git commit, host, config file, and command for each experiment.

## First Remote Smoke Test

```bash
bash scripts/remote/smoke_test.sh gpuserver4090
```


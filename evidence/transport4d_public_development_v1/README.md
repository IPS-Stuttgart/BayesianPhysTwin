# Public Transport4D development matrix

This directory binds the already-open public DEFORM/PyElastica outcomes used to
design the transport hierarchy. Reproduce the compact result with:

```bash
python scripts/science/run_transport4d_public_development_v1.py \
  --protocol protocols/transport4d_public_development_v1.json \
  --output evidence/transport4d_public_development_v1/result.json \
  --report evidence/transport4d_public_development_v1/report.md
```

The matrix contains a same-object cross-backend positive result, a cross-object
exact-coefficient negative result, and a matching-object procedure-replication
positive result. It is explicitly retrospective development evidence, not a
confirmation of the newly introduced selector.

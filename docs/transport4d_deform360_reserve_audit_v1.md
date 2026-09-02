# Transport4D Deform360 reserve audit

Run only against the registered public Deform360 root:

```bash
PYTHONPATH=src python \
  scripts/science/audit_transport4d_deform360_reserve_v1.py \
  --data-root /mnt/seagate10tb/florianpfaff/datasets/deform360 \
  --protocol protocols/transport4d_deform360_reserve_audit_v1.json \
  --action-kernel-protocol protocols/deform360_action_kernel_v3.json \
  --untouched-protocol protocols/deform360_untouched_confirmation_v5.json \
  --output reserve.json \
  --report reserve.md
```

The command reads object-directory names and `metadata.json` only. It excludes
every object used or protected by the two bound BayesianPhysTwin predecessor
protocols and an exact Causal4D holdings binding. The latter conservatively
protects `085-scarf-cloth`, `170-spider`, and `171-penguin`, which were reserved
or explored outside the two local rosters. All remaining namespaces are assigned
to calibration or confirmation. The audit cannot authorize numeric payload access.

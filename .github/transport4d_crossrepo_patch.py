#!/usr/bin/env python3
"""Close the cross-repository Deform360 reserve-custody gap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/transport4d_deform360_reserve_audit_v1.json"
MODULE = ROOT / "src/bayesian_phystwin_experiments/transport4d_public_reserve_v1.py"
TEST = ROOT / "tests/test_transport4d_public_reserve_v1.py"
DOC = ROOT / "docs/transport4d_deform360_reserve_audit_v1.md"


def canonical_id(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected one replacement in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_protocol() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol_id") != (
        "75b9fec8f3faab5f34b920e1a8f28eaab52a378655ab7db9c6b9377df0e8c599"
    ):
        raise SystemExit("unexpected reserve protocol identity")
    if value.get("additional_protected_object_ids") != []:
        raise SystemExit("reserve additional protection was already changed")
    additional = ["085-scarf-cloth", "170-spider", "171-penguin"]
    value["additional_protected_object_ids"] = additional
    upstream = value["upstream_bindings"]
    if "causal4d_deform360_holdings_v1" in upstream:
        raise SystemExit("Causal4D binding already exists")
    upstream["causal4d_deform360_holdings_v1"] = {
        "repository": "IPS-Stuttgart/Causal4D",
        "revision": "e071e3c14e3e318ccf20c2206c3640b8de496eff",
        "path": "configs/causal4d_public/deform360_gpuserver6000_holdings_v1.json",
        "git_blob_sha1": "5cc7832ad61d7933fb3860463e920d918e687eda",
        "additional_protected_object_ids": additional,
    }
    unsigned = {key: item for key, item in value.items() if key != "protocol_id"}
    value["protocol_id"] = canonical_id(unsigned)
    if value["protocol_id"] != (
        "a7d5f520ae3e4761452476af7af57fa023aea031114aa0239c2690bb531407c8"
    ):
        raise SystemExit("unexpected revised reserve protocol identity")
    PROTOCOL.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def patch_module() -> None:
    replace_once(
        MODULE,
        '''    action_binding = upstream.get("action_kernel_v3")\n    untouched_binding = upstream.get("untouched_confirmation_v5")\n    if not isinstance(action_binding, Mapping) or not isinstance(\n        untouched_binding, Mapping\n    ):\n        raise ValueError("reserve upstream protocol bindings are malformed")\n''',
        '''    action_binding = upstream.get("action_kernel_v3")\n    untouched_binding = upstream.get("untouched_confirmation_v5")\n    causal4d_binding = upstream.get("causal4d_deform360_holdings_v1")\n    if (\n        not isinstance(action_binding, Mapping)\n        or not isinstance(untouched_binding, Mapping)\n        or not isinstance(causal4d_binding, Mapping)\n    ):\n        raise ValueError("reserve upstream protocol bindings are malformed")\n''',
    )
    replace_once(
        MODULE,
        '''    eligible = _literal_strings(\n        untouched_protocol.get("eligible_object_ids"),\n        name="untouched eligible_object_ids",\n    )\n''',
        '''    if causal4d_binding.get("repository") != "IPS-Stuttgart/Causal4D":\n        raise ValueError("Causal4D reserve repository binding changed")\n    if causal4d_binding.get("path") != (\n        "configs/causal4d_public/deform360_gpuserver6000_holdings_v1.json"\n    ):\n        raise ValueError("Causal4D reserve path binding changed")\n    for field_name in ("revision", "git_blob_sha1"):\n        value = causal4d_binding.get(field_name)\n        if (\n            type(value) is not str\n            or len(value) != 40\n            or any(character not in "0123456789abcdef" for character in value)\n        ):\n            raise ValueError(f"invalid Causal4D reserve {field_name}")\n    eligible = _literal_strings(\n        untouched_protocol.get("eligible_object_ids"),\n        name="untouched eligible_object_ids",\n    )\n''',
    )
    replace_once(
        MODULE,
        '''    additional = reserve_protocol.get("additional_protected_object_ids", [])\n    protected.update(_literal_strings(additional, name="additional protected objects"))\n''',
        '''    additional = _literal_strings(\n        reserve_protocol.get("additional_protected_object_ids", []),\n        name="additional protected objects",\n    )\n    causal4d_additional = _literal_strings(\n        causal4d_binding.get("additional_protected_object_ids"),\n        name="Causal4D additional protected objects",\n    )\n    if tuple(sorted(additional)) != tuple(sorted(causal4d_additional)):\n        raise ValueError("cross-repository protected object roster differs")\n    protected.update(additional)\n''',
    )


def patch_test() -> None:
    replace_once(
        TEST,
        '''        "opened-confirmation",\n        "new-object-a",\n''',
        '''        "opened-confirmation",\n        "cross-repo-protected",\n        "new-object-a",\n''',
    )
    replace_once(
        TEST,
        '''            "untouched_confirmation_v5": {\n                "path": str(untouched_path),\n                "protocol_id": "untouched-test-v5",\n                "eligible_object_count": 1,\n            },\n        },\n        "additional_protected_object_ids": [],\n''',
        '''            "untouched_confirmation_v5": {\n                "path": str(untouched_path),\n                "protocol_id": "untouched-test-v5",\n                "eligible_object_count": 1,\n            },\n            "causal4d_deform360_holdings_v1": {\n                "repository": "IPS-Stuttgart/Causal4D",\n                "revision": "a" * 40,\n                "path": (\n                    "configs/causal4d_public/"\n                    "deform360_gpuserver6000_holdings_v1.json"\n                ),\n                "git_blob_sha1": "b" * 40,\n                "additional_protected_object_ids": ["cross-repo-protected"],\n            },\n        },\n        "additional_protected_object_ids": ["cross-repo-protected"],\n''',
    )
    replace_once(
        TEST,
        '''    assert result["reservation_ready"] is True\n    assert all(row["numeric_payload_opened"] is False for row in result["objects"])\n''',
        '''    assert result["reservation_ready"] is True\n    assert "cross-repo-protected" in result["protected_object_ids"]\n    assert all(row["numeric_payload_opened"] is False for row in result["objects"])\n''',
    )
    TEST.write_text(
        TEST.read_text(encoding="utf-8")
        + '''\n\ndef test_cross_repository_roster_mismatch_fails_closed(tmp_path: Path) -> None:\n    root, protocol_path, action, untouched = fixture(tmp_path)\n    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))\n    protocol["additional_protected_object_ids"] = []\n    unsigned = {\n        key: value for key, value in protocol.items() if key != "protocol_id"\n    }\n    protocol["protocol_id"] = canonical_id(unsigned)\n    write_json(protocol_path, protocol)\n\n    with pytest.raises(ValueError, match="cross-repository protected object roster"):\n        audit_deform360_transport_reserve(\n            data_root=root,\n            reserve_protocol_path=protocol_path,\n            action_kernel_protocol_path=action,\n            untouched_protocol_path=untouched,\n        )\n''',
        encoding="utf-8",
    )


def patch_doc() -> None:
    replace_once(
        DOC,
        '''The command reads object-directory names and `metadata.json` only. It excludes\nevery object used or protected by the two bound predecessor protocols and assigns\nall remaining namespaces to calibration or confirmation. It cannot authorize\nnumeric payload access.\n''',
        '''The command reads object-directory names and `metadata.json` only. It excludes\nevery object used or protected by the two bound BayesianPhysTwin predecessor\nprotocols and an exact Causal4D holdings binding. The latter conservatively\nprotects `085-scarf-cloth`, `170-spider`, and `171-penguin`, which were reserved\nor explored outside the two local rosters. All remaining namespaces are assigned\nto calibration or confirmation. The audit cannot authorize numeric payload access.\n''',
    )


def main() -> None:
    patch_protocol()
    patch_module()
    patch_test()
    patch_doc()


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from bayesian_phystwin import deform360_frame_zero_assets as frame_zero
from bayesian_phystwin import deform360_frame_zero_semantic_gate as semantic_gate
from bayesian_phystwin import deform360_held_online_prefix as online
from bayesian_phystwin import deform360_held_physical_prior as physical
from bayesian_phystwin import deform360_held_protocol as v7_protocol
from bayesian_phystwin import deform360_held_v8_builders as builders
from bayesian_phystwin import deform360_held_v8_protocol as v8_protocol


ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "scripts" / "held"
CALIBRATION_CASE = HELD / "run_deform360_v8_calibration_case.sh"
CALIBRATION_SHARD = HELD / "run_deform360_v8_calibration_shard.sh"
CONFIRMATION_CASE = HELD / "run_deform360_v8_confirmation_case.sh"
CONFIRMATION_SHARD = HELD / "run_deform360_v8_confirmation_shard.sh"
COMMON = HELD / "run_deform360_v8_case_common.sh"

CALIBRATION_SPECS = (
    "072-cotton-clohesline-ep0003:072-cotton-clohesline:0003",
    "002-rope-silk-ep0004:002-rope-silk:0004",
    "002-rope-silk-ep0008:002-rope-silk:0008",
    "083-blanket-cloth-ep0000:083-blanket-cloth:0000",
    "083-blanket-cloth-ep0003:083-blanket-cloth:0003",
    "083-blanket-cloth-ep0006:083-blanket-cloth:0006",
    "085-scarf-cloth-ep0000:085-scarf-cloth:0000",
    "085-scarf-cloth-ep0005:085-scarf-cloth:0005",
    "085-scarf-cloth-ep0007:085-scarf-cloth:0007",
    "092-squirrel-ep0002:092-squirrel:0002",
    "092-squirrel-ep0003:092-squirrel:0003",
    "092-squirrel-ep0006:092-squirrel:0006",
    "170-spider-ep0002:170-spider:0002",
    "170-spider-ep0004:170-spider:0004",
    "170-spider-ep0007:170-spider:0007",
)
CONFIRMATION_SPECS = (
    "002-rope-silk-ep0001:002-rope-silk:0001",
    "081-stripe-rope-ep0005:081-stripe-rope:0005",
    "085-scarf-cloth-ep0002:085-scarf-cloth:0002",
    "083-blanket-cloth-ep0007:083-blanket-cloth:0007",
    "092-squirrel-ep0001:092-squirrel:0001",
    "170-spider-ep0006:170-spider:0006",
)


def _array(source: str, name: str) -> tuple[str, ...]:
    match = re.search(
        rf"readonly -a {re.escape(name)}=\((.*?)\n\)", source, flags=re.DOTALL
    )
    assert match is not None, name
    return tuple(re.findall(r'"([^"\n]+)"', match.group(1)))


def _case_pattern_values(source: str) -> tuple[str, ...]:
    match = re.search(
        r'case "\$CASE_NAME:\$OBJECT:\$EPISODE" in\n(.*?)\n\s*\*\)',
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    values = re.findall(r"\s*([0-9][^|\\\n]+)(?:\|\\)?", match.group(1))
    return tuple(value.removesuffix(") ;;").rstrip() for value in values)


def test_exact_fresh_calibration_and_untouched_confirmation_tuples() -> None:
    calibration_case = CALIBRATION_CASE.read_text()
    confirmation_case = CONFIRMATION_CASE.read_text()
    assert _case_pattern_values(calibration_case) == CALIBRATION_SPECS
    assert _case_pattern_values(confirmation_case) == CONFIRMATION_SPECS
    assert "002-rope-silk-ep0003" not in calibration_case
    assert tuple(value.split(":", 1)[0] for value in CALIBRATION_SPECS) == (
        v8_protocol.CALIBRATION_CASE_NAMES
    )
    assert tuple(value.split(":", 1)[0] for value in CONFIRMATION_SPECS) == (
        v8_protocol.CONFIRMATION_CASE_NAMES
    )


@pytest.mark.parametrize(
    ("path", "all_specs", "left_count", "right_count"),
    (
        (CALIBRATION_SHARD, CALIBRATION_SPECS, 8, 7),
        (CONFIRMATION_SHARD, CONFIRMATION_SPECS, 3, 3),
    ),
)
def test_two_gpu_shards_are_disjoint_and_complete(
    path: Path, all_specs: tuple[str, ...], left_count: int, right_count: int
) -> None:
    source = path.read_text()
    observed_all = _array(source, "ALL_CASE_SPECS")
    left = _array(source, "SHARD_0_CASE_SPECS")
    right = _array(source, "SHARD_1_CASE_SPECS")
    assert observed_all == all_specs
    assert len(left) == left_count
    assert len(right) == right_count
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(observed_all)
    assert len((*left, *right)) == len(set((*left, *right)))
    assert 'case "$SHARD_INDEX:$CUDA_DEVICE" in 0:0|1:1)' in source
    assert '"$(hostname)" == "workstation2"' in source
    assert "gpuserver6000/workstation2" in source


def test_case_common_is_v8_only_and_freezes_field_before_any_barrier() -> None:
    source = COMMON.read_text()
    assert '/held-v82"' in source
    assert "held-v7" not in source
    assert "run_deform360_v7" not in source
    assert "bayesian_phystwin.cli.deform360_held_v8_frame_zero_assets" in source
    assert "bayesian_phystwin.cli.deform360_held_v8_physical_prior" in source
    assert "bayesian_phystwin.cli.deform360_held_v8_online_prefix" in source
    assert "validate_frame_zero_bundle_manifest" in source
    assert "validate_physical_prior_seal" in source
    assert "create_prefix_stage_authorization" in source
    assert "validate_online_prediction_seal" in source
    assert "write_preoutcome_frozen_field_manifest" in source
    assert "110b3c1831898ff6b333f35236401761222f85eafac1dcbcea7b7183d5b434bd" in source
    assert source.index("validate_online_prediction_seal") < source.index(
        "write_preoutcome_frozen_field_manifest"
    )
    assert "--target" not in source
    assert "--outcome" not in source
    assert "--tactile" not in source
    assert "authorize_target_reconstruction_capabilities" not in source
    assert "authorize_future_score_capabilities" not in source


def _frame_zero_sealing_program() -> str:
    source = COMMON.read_text(encoding="utf-8")
    match = re.search(
        r"<<'PY_FRAME_ZERO_SEAL'\n(?P<program>.*?)\nPY_FRAME_ZERO_SEAL",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("program")


def test_frame_zero_sealing_is_after_build_before_validation_and_exact() -> None:
    source = COMMON.read_text(encoding="utf-8")
    build = source.index('CURRENT_PHASE="frame-zero-build"')
    sealing = source.index('CURRENT_PHASE="frame-zero-sealing"')
    validation = source.index('CURRENT_PHASE="frame-zero-validation"')
    assert build < sealing < validation

    program = _frame_zero_sealing_program()
    assert "artifact_names = (" in program
    assert (
        program.index('"known_action_76.npz"')
        < program.index('"frame_zero_bundle.npz"')
        < program.index('"frame_zero_bundle.manifest.json"')
    )
    assert "observed_names == tuple(sorted(artifact_names))" in program
    assert "os.fchmod(descriptor, 0o400)" in program


def test_fresh_minimal_frame_zero_outputs_are_sealed_before_v8_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "fresh-frame-zero"
    output.mkdir()
    artifacts = tuple(
        output / name
        for name in (
            "known_action_76.npz",
            "frame_zero_bundle.npz",
            "frame_zero_bundle.manifest.json",
        )
    )
    for path in artifacts:
        path.write_bytes(f"complete:{path.name}\n".encode())
        path.chmod(0o600)

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-", str(output)],
        input=_frame_zero_sealing_program(),
        text=True,
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "mode=0400 count=3 manifest_last=true" in completed.stdout
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in artifacts)

    reached_lock_validation = False

    def stop_after_manifest_mode(_path: str | Path) -> dict[str, object]:
        nonlocal reached_lock_validation
        reached_lock_validation = True
        raise RuntimeError("v8 validator passed the frame-zero mode gate")

    monkeypatch.setattr(v8_protocol, "validate_protocol_lock", stop_after_manifest_mode)
    with pytest.raises(RuntimeError, match="passed the frame-zero mode gate"):
        v8_protocol.validate_frame_zero_bundle_manifest(
            artifacts[-1], tmp_path / "unused-lock.json"
        )
    assert reached_lock_validation is True


def test_frame_zero_sealing_rejects_any_extra_output_before_chmod(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fresh-frame-zero-with-extra"
    output.mkdir()
    expected = (
        output / "known_action_76.npz",
        output / "frame_zero_bundle.npz",
        output / "frame_zero_bundle.manifest.json",
    )
    for path in expected:
        path.write_bytes(b"complete\n")
        path.chmod(0o600)
    (output / "unexpected.bin").write_bytes(b"not allowlisted\n")

    rejected = subprocess.run(
        [sys.executable, "-I", "-B", "-", str(output)],
        input=_frame_zero_sealing_program(),
        text=True,
        check=False,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "exact three-artifact allowlist" in rejected.stderr
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in expected)


def test_replacement_source_is_explicit_validated_and_semantically_frozen() -> None:
    case_source = CALIBRATION_CASE.read_text()
    shard_source = CALIBRATION_SHARD.read_text()
    common = COMMON.read_text()
    assert "[REPLACEMENT_SOURCE_MANIFEST]" in case_source
    assert "REPLACEMENT_SOURCE_MANIFEST" in shard_source
    assert "validate_aligned_source_manifest" in common
    assert "replacement_source_permit_evidence" in common
    assert "expected_source_permit=" in common
    assert "replacement_source_permit_evidence" in shard_source
    assert "expected_source_permit=" in shard_source
    assert common.index("validate_aligned_source_manifest") < common.index(
        "deform360_held_v8_frame_zero_assets"
    )
    for literal in (
        '"semantic_label": "rope"',
        '"action": "drag"',
        '"action_location": "center"',
        '"bimanual": False',
        '"prehensile": True',
        'manifest["aligned_episode_dir"]',
    ):
        assert literal in common
    assert "non-replacement case rejects a source manifest" in case_source
    assert '"$(stat -c \'%a\' -- "$REPLACEMENT_SOURCE_MANIFEST")" == "400"' in common


def test_explicit_v8_adapter_changes_only_process_local_hooks_and_restores_v7() -> None:
    originals = {
        "frame_protocol": frame_zero.HELD_PROTOCOL_ID,
        "frame_loader": frame_zero.load_generic_held_lock,
        "frame_semantic": frame_zero.semantic_label_for_object_id,
        "semantic": semantic_gate.semantic_label_for_object_id,
        "physical_protocol": physical.PROTOCOL_ID,
        "physical_loader": physical.load_held_protocol_lock,
        "physical_sealer": physical.create_physical_prior_seal,
        "physical_pycache": physical.HELD_PYCACHE_PREFIX,
        "online_protocol": online.PROTOCOL_ID,
        "online_sealer": online.create_online_prediction_seal,
        "online_frame_validator": online.validate_frame_zero_asset_manifest,
    }
    assert originals["frame_protocol"] == v7_protocol.PROTOCOL_ID
    with pytest.raises(ValueError, match="outside the frozen"):
        semantic_gate.semantic_label_for_object_id("072-cotton-clohesline")

    with builders.explicit_v8_builder_context("frame-zero"):
        assert frame_zero.HELD_PROTOCOL_ID == v8_protocol.PROTOCOL_ID
        assert physical.PROTOCOL_ID == v7_protocol.PROTOCOL_ID
        assert online.PROTOCOL_ID == v7_protocol.PROTOCOL_ID
        assert frame_zero.load_generic_held_lock is v8_protocol.validate_protocol_lock
        assert (
            frame_zero.semantic_label_for_object_id("072-cotton-clohesline") == "rope"
        )
        assert (
            semantic_gate.semantic_label_for_object_id("072-cotton-clohesline")
            == "rope"
        )

    with builders.explicit_v8_builder_context("physical"):
        # The old frame-zero module stays at v7 so the v8 validator can replay
        # its deep audit on a transient legacy view.
        assert frame_zero.HELD_PROTOCOL_ID == v7_protocol.PROTOCOL_ID
        assert physical.PROTOCOL_ID == v8_protocol.PROTOCOL_ID
        assert physical.HELD_PYCACHE_PREFIX == "/nonexistent/bpt-held-v82-pycache"
        assert (
            physical.create_physical_prior_seal
            is v8_protocol.create_physical_prior_seal
        )
        assert online.PROTOCOL_ID == v7_protocol.PROTOCOL_ID

    with builders.explicit_v8_builder_context("online"):
        assert frame_zero.HELD_PROTOCOL_ID == v7_protocol.PROTOCOL_ID
        assert physical.PROTOCOL_ID == v8_protocol.PROTOCOL_ID
        assert online.PROTOCOL_ID == v8_protocol.PROTOCOL_ID
        assert (
            online.create_online_prediction_seal
            is v8_protocol.create_online_prediction_seal
        )
        assert (
            online.validate_frame_zero_asset_manifest
            is builders._validate_already_protocol_validated_v8_frame_zero
        )

    assert frame_zero.HELD_PROTOCOL_ID == originals["frame_protocol"]
    assert frame_zero.load_generic_held_lock is originals["frame_loader"]
    assert frame_zero.semantic_label_for_object_id is originals["frame_semantic"]
    assert semantic_gate.semantic_label_for_object_id is originals["semantic"]
    assert physical.PROTOCOL_ID == originals["physical_protocol"]
    assert physical.load_held_protocol_lock is originals["physical_loader"]
    assert physical.create_physical_prior_seal is originals["physical_sealer"]
    assert physical.HELD_PYCACHE_PREFIX == originals["physical_pycache"]
    assert online.PROTOCOL_ID == originals["online_protocol"]
    assert online.create_online_prediction_seal is originals["online_sealer"]
    assert (
        online.validate_frame_zero_asset_manifest is originals["online_frame_validator"]
    )


def test_v8_cli_wrappers_are_dedicated_and_allowlist_only_numerical_clis() -> None:
    wrapper_names = (
        "deform360_held_v8_frame_zero_assets.py",
        "deform360_held_v8_physical_prior.py",
        "deform360_held_v8_online_prefix.py",
    )
    for name in wrapper_names:
        source = (ROOT / "src" / "bayesian_phystwin" / "cli" / name).read_text()
        assert "deform360_held_v8_builders import main_for" in source
    with pytest.raises(ValueError, match="outside the v8 numerical CLI allowlist"):
        builders.run_v8_adapted_cli("bayesian_phystwin.deform360_held_protocol")


@pytest.mark.parametrize(
    "path",
    (CALIBRATION_CASE, CALIBRATION_SHARD, CONFIRMATION_CASE, CONFIRMATION_SHARD),
)
def test_runners_normalize_environment_and_validate_immutable_code_and_lock(
    path: Path,
) -> None:
    source = path.read_text()
    assert "exec env -i" in source
    assert "unset BASH_ENV ENV CDPATH" in source
    if "case.sh" in path.name:
        common = COMMON.read_text()
        assert 'find "$CODE" -xdev -perm /222' in common
        assert "stat -c '%a' -- \"$LOCK\"" in common
        assert "validate_protocol_lock" in common
        assert "code-([0-9a-f]{40}|[0-9a-f]{64})" in common
    else:
        assert 'find "$CODE" -xdev -perm /222' in source
        assert "stat -c '%a' -- \"$LOCK\"" in source
        assert "validate_protocol_lock" in source
        assert "code-([0-9a-f]{40}|[0-9a-f]{64})" in source


def test_confirmation_shard_requires_recursively_validated_go_lock() -> None:
    source = CONFIRMATION_SHARD.read_text()
    assert 'lock.get("stage") != "confirmation"' in source
    assert 'lock.get("confirmation_access_authorized") is not True' in source
    assert "validate_protocol_lock" in source
    assert "calibration GO" in source
    assert "/calibration/cases" not in source
    assert "calibration-gate-decision.json" not in source


@pytest.mark.parametrize(
    ("path", "normalized_name", "arguments", "message"),
    (
        (
            CALIBRATION_CASE,
            "BPT_HELD_V8_CALIBRATION_CASE_ENV_NORMALIZED",
            ("0", "bad", "bad", "0000"),
            "normalized calibration-case environment contains POISON",
        ),
        (
            CONFIRMATION_CASE,
            "BPT_HELD_V8_CONFIRMATION_CASE_ENV_NORMALIZED",
            ("0", "bad", "bad", "0000"),
            "normalized confirmation-case environment contains POISON",
        ),
    ),
)
def test_normalized_case_environment_rejects_unknown_variables(
    path: Path, normalized_name: str, arguments: tuple[str, ...], message: str
) -> None:
    environment = {
        "HOME": "/home/florianpfaff",
        "USER": "florianpfaff",
        "LOGNAME": "florianpfaff",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        normalized_name: "1",
        "BPT_HELD_V8_CODE": "",
        "BPT_HELD_V8_LOCK_VERIFIED_SHA256": "",
        "POISON": "must-not-survive",
    }
    completed = subprocess.run(
        ["/bin/bash", os.fspath(path), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_shell_sources_parse_and_are_executable() -> None:
    for path in (
        CALIBRATION_CASE,
        CALIBRATION_SHARD,
        CONFIRMATION_CASE,
        CONFIRMATION_SHARD,
        COMMON,
    ):
        assert path.stat().st_mode & 0o111
        subprocess.run(["/bin/bash", "-n", os.fspath(path)], check=True)

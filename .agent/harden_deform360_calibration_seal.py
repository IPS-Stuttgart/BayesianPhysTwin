from __future__ import annotations

from pathlib import Path


def replace_once(path: str, label: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{label}: expected one replacement anchor in {path}, found {count}"
        )
    source.write_text(text.replace(old, new), encoding="utf-8")
    print(f"applied: {label}")


def append_once(path: str, marker: str, addition: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if marker in text:
        print(f"already present: {marker}")
        return
    source.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")
    print(f"appended: {marker}")


def patch_stage0_execution() -> None:
    path = "src/bayesian_phystwin/deform360_calibration_execution.py"
    replace_once(
        path,
        "insert Stage-0 digest verifier",
        "def load_deform360_stage0_selection(\n"
        "    path: str | Path,\n"
        ") -> Deform360Stage0SelectionV1:\n",
        '''def _validate_stage0_content_ids(
    value: Mapping[str, Any],
    *,
    protocol_path: str | Path | None,
) -> None:
    """Recompute every Stage-0 identity instead of trusting declarations."""

    selection = value["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("Stage-0 selection must be an object")
    declared_selection = sha256_digest(
        value["selection_sha256"],
        name="Stage-0 selection_sha256",
    )
    observed_selection = content_id(selection)
    if observed_selection != declared_selection:
        raise ValueError("Stage-0 selection_sha256 does not match selection")

    content_payload = dict(value)
    content_payload.pop("content_selection_sha256")
    content_payload.pop("implementation_revision")
    content_payload.pop("selection_artifact_sha256")
    declared_content = sha256_digest(
        value["content_selection_sha256"],
        name="Stage-0 content_selection_sha256",
    )
    observed_content = content_id(content_payload)
    if observed_content != declared_content:
        raise ValueError("Stage-0 content_selection_sha256 does not match content")

    artifact_payload = dict(value)
    artifact_payload.pop("selection_artifact_sha256")
    declared_artifact = sha256_digest(
        value["selection_artifact_sha256"],
        name="Stage-0 selection_artifact_sha256",
    )
    observed_artifact = content_id(artifact_payload)
    if observed_artifact != declared_artifact:
        raise ValueError("Stage-0 selection_artifact_sha256 does not match content")

    if protocol_path is not None:
        protocol = load_strict_json_object(
            protocol_path,
            label="Deform360 Stage-0 protocol",
        )
        declared_protocol = sha256_digest(
            value["protocol_sha256"],
            name="Stage-0 protocol_sha256",
        )
        observed_protocol = content_id(protocol)
        if observed_protocol != declared_protocol:
            raise ValueError("Stage-0 protocol_sha256 does not match protocol content")


def load_deform360_stage0_selection(
    path: str | Path,
    *,
    protocol_path: str | Path | None = None,
) -> Deform360Stage0SelectionV1:
''',
    )
    replace_once(
        path,
        "retain Stage-0 object before digest validation",
        "    return Deform360Stage0SelectionV1(\n",
        "    result = Deform360Stage0SelectionV1(\n",
    )
    replace_once(
        path,
        "validate and return Stage-0 object",
        '''        confirmation_units=tuple(
            _stage0_unit(item, name=f"Stage-0 confirmation unit {index}")
            for index, item in enumerate(confirmation_raw)
        ),
    )


def load_deform360_calibration_artifact_ref(
''',
        '''        confirmation_units=tuple(
            _stage0_unit(item, name=f"Stage-0 confirmation unit {index}")
            for index, item in enumerate(confirmation_raw)
        ),
    )
    _validate_stage0_content_ids(value, protocol_path=protocol_path)
    return result


def load_deform360_calibration_artifact_ref(
''',
    )


def patch_bundle_revision_binding() -> None:
    path = "src/bayesian_phystwin/deform360_calibration_bundle.py"
    replace_once(
        path,
        "bind artifact revisions to bundle revision",
        '''        expected_groups = tuple(sorted(calibration_ids))
        for artifact in calibration_artifacts:
            _require(
                artifact.calibration_group_ids == expected_groups,
''',
        '''        expected_groups = tuple(sorted(calibration_ids))
        for artifact in calibration_artifacts:
            _require(
                artifact.implementation_revision == implementation_revision,
                f"calibration artifact {artifact.role} implementation revision "
                "differs from bundle",
            )
            _require(
                artifact.calibration_group_ids == expected_groups,
''',
    )


def patch_cli_runtime_and_selection_guards() -> None:
    path = "src/bayesian_phystwin/cli/deform360_calibration_execution.py"
    replace_once(
        path,
        "import runtime module verifier",
        "import argparse\nimport json\n",
        "import argparse\nimport importlib\nimport json\n",
    )
    replace_once(
        path,
        "register committed selection and runtime sources",
        '''_REPOSITORY_SOURCES = (
    ".github/workflows/deform360-calibration-seal.yml",
    "protocols/deform360_official_hub_visuotactile_v1.json",
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    ),
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    ),
    "src/bayesian_phystwin/deform360_calibration_execution.py",
    "src/bayesian_phystwin/deform360_calibration_bundle.py",
    "src/bayesian_phystwin/deform360_visual_provider_lock.py",
    "src/bayesian_phystwin/evidence_use_ledger.py",
    "src/bayesian_phystwin/cli/deform360_calibration_execution.py",
    "src/bayesian_phystwin/cli/experiments.py",
)
''',
        '''_COMMITTED_SELECTION = (
    "protocols/locks/deform360_official_hub_visuotactile_v1_selection.json"
)
_REPOSITORY_SOURCES = (
    ".github/workflows/deform360-calibration-seal.yml",
    "protocols/deform360_official_hub_visuotactile_v1.json",
    _COMMITTED_SELECTION,
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    ),
    (
        "protocols/amendments/"
        "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    ),
    "src/bayesian_phystwin/_canonical_contracts.py",
    "src/bayesian_phystwin/_portable_contracts.py",
    "src/bayesian_phystwin/deform360_calibration_execution.py",
    "src/bayesian_phystwin/deform360_calibration_bundle.py",
    "src/bayesian_phystwin/deform360_visual_provider_lock.py",
    "src/bayesian_phystwin/evidence_use_ledger.py",
    "src/bayesian_phystwin/cli/deform360_calibration_execution.py",
    "src/bayesian_phystwin/cli/experiments.py",
)
_RUNTIME_MODULE_SOURCES = {
    "bayesian_phystwin._canonical_contracts": (
        "src/bayesian_phystwin/_canonical_contracts.py"
    ),
    "bayesian_phystwin._portable_contracts": (
        "src/bayesian_phystwin/_portable_contracts.py"
    ),
    "bayesian_phystwin.deform360_calibration_execution": (
        "src/bayesian_phystwin/deform360_calibration_execution.py"
    ),
    "bayesian_phystwin.deform360_calibration_bundle": (
        "src/bayesian_phystwin/deform360_calibration_bundle.py"
    ),
    "bayesian_phystwin.deform360_visual_provider_lock": (
        "src/bayesian_phystwin/deform360_visual_provider_lock.py"
    ),
    "bayesian_phystwin.evidence_use_ledger": (
        "src/bayesian_phystwin/evidence_use_ledger.py"
    ),
    "bayesian_phystwin.cli.deform360_calibration_execution": (
        "src/bayesian_phystwin/cli/deform360_calibration_execution.py"
    ),
}
''',
    )
    replace_once(
        path,
        "add runtime and committed-selection verification helpers",
        '''    return observed


def _copy_source(source: Path, destination: Path) -> str:
''',
        '''    return observed


def _verify_runtime_sources(repository: Path) -> None:
    """Require the imported sealer code to match the reviewed checkout."""

    for module_name, relative_path in _RUNTIME_MODULE_SOURCES.items():
        module = importlib.import_module(module_name)
        runtime_name = getattr(module, "__file__", None)
        if type(runtime_name) is not str or not runtime_name:
            raise ValueError(f"cannot identify runtime source for module {module_name}")
        runtime = _ordinary_file(
            Path(runtime_name),
            name=f"runtime source for {module_name}",
        )
        reviewed = _ordinary_file(
            repository / relative_path,
            name=f"reviewed source for {module_name}",
        )
        if file_sha256(runtime) != file_sha256(reviewed):
            raise ValueError(
                "runtime source bytes differ from reviewed checkout: "
                f"{module_name}"
            )


def _verify_committed_selection_lock(
    repository: Path,
    supplied_selection: Path,
) -> None:
    """Reject a structurally valid but unreviewed Stage-0 cohort."""

    committed = _ordinary_file(
        repository / _COMMITTED_SELECTION,
        name="committed Stage-0 selection",
    )
    supplied = _ordinary_file(
        supplied_selection,
        name="supplied Stage-0 selection",
    )
    if file_sha256(committed) != file_sha256(supplied):
        raise ValueError("supplied Stage-0 selection bytes differ from committed lock")


def _copy_source(source: Path, destination: Path) -> str:
''',
    )
    replace_once(
        path,
        "enforce runtime and committed-selection guards",
        '''    revision = _verify_repository(
        repository_root,
        expected_revision=args.implementation_revision,
    )
    output = args.output_dir.resolve()
''',
        '''    revision = _verify_repository(
        repository_root,
        expected_revision=args.implementation_revision,
    )
    _verify_runtime_sources(repository_root)
    _verify_committed_selection_lock(repository_root, args.selection_lock)
    output = args.output_dir.resolve()
''',
    )
    replace_once(
        path,
        "validate Stage-0 protocol bytes in CLI",
        '''        stage0 = load_deform360_stage0_selection(
            temporary / "sources/stage0/selection.json"
        )
''',
        '''        stage0 = load_deform360_stage0_selection(
            temporary / "sources/stage0/selection.json",
            protocol_path=(
                temporary
                / "sources/repository/protocols/"
                "deform360_official_hub_visuotactile_v1.json"
            ),
        )
''',
    )


def patch_visual_lock_writers() -> None:
    path = "src/bayesian_phystwin/deform360_visual_provider_lock.py"
    replace_once(
        path,
        "import atomic writer",
        '''from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
''',
        '''from ._canonical_contracts import (
    canonical_string_tuple,
    frozen_finite_json_mapping,
    genuine_boolean,
    genuine_integer,
    literal_lower_hex,
    plain_json,
)
from ._portable_contracts import write_atomic_json
''',
    )
    replace_once(
        path,
        "make provider lock persistence atomic",
        '''def save_deform360_visual_provider_lock(
    path: str | Path,
    lock: Deform360VisualProviderLockV1,
) -> None:
    """Serialize a canonical human-readable visual-provider lock."""

    Path(path).write_text(
        json.dumps(lock.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\\n",
        encoding="utf-8",
    )
''',
        '''def save_deform360_visual_provider_lock(
    path: str | Path,
    lock: Deform360VisualProviderLockV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist a visual-provider lock without silent replacement."""

    if not isinstance(lock, Deform360VisualProviderLockV1):
        raise TypeError("lock must be a Deform360VisualProviderLockV1")
    write_atomic_json(lock.to_record(), path, overwrite=overwrite)
''',
    )
    replace_once(
        path,
        "make calibration lock persistence atomic",
        '''def save_deform360_visual_calibration_lock(
    path: str | Path,
    lock: Deform360VisualCalibrationLockV1,
) -> None:
    """Serialize a canonical human-readable Stage-1 calibration lock."""

    Path(path).write_text(
        json.dumps(lock.to_record(), indent=2, sort_keys=True, allow_nan=False) + "\\n",
        encoding="utf-8",
    )
''',
        '''def save_deform360_visual_calibration_lock(
    path: str | Path,
    lock: Deform360VisualCalibrationLockV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist a Stage-1 lock without silent replacement."""

    if not isinstance(lock, Deform360VisualCalibrationLockV1):
        raise TypeError("lock must be a Deform360VisualCalibrationLockV1")
    write_atomic_json(lock.to_record(), path, overwrite=overwrite)
''',
    )


def patch_tests() -> None:
    bundle_test = "tests/test_deform360_calibration_bundle.py"
    replace_once(
        bundle_test,
        "align bundle fixture artifact revisions",
        '            implementation_revision="a" * 40,\n',
        '            implementation_revision="5" * 40,\n',
    )
    append_once(
        bundle_test,
        "def test_bundle_rejects_artifact_revision_mismatch()",
        '''

def test_bundle_rejects_artifact_revision_mismatch() -> None:
    bundle = _bundle()
    changed = replace(
        bundle.calibration_artifacts[0],
        implementation_revision="9" * 40,
    )
    with pytest.raises(ValueError, match="implementation revision"):
        _bundle(
            calibration_artifacts=(
                changed,
                *bundle.calibration_artifacts[1:],
            )
        )
''',
    )

    execution_test = "tests/test_deform360_calibration_execution.py"
    replace_once(
        execution_test,
        "parameterize execution artifact revision fixture",
        '''def _artifacts(stage0) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
''',
        '''def _artifacts(
    stage0,
    *,
    implementation_revision: str = "c" * 40,
) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
''',
    )
    replace_once(
        execution_test,
        "use parameterized execution artifact revision",
        '            implementation_revision="a" * 40,\n',
        '            implementation_revision=implementation_revision,\n',
    )
    replace_once(
        execution_test,
        "parameterize CLI fixture artifact revision",
        '''def _write_cli_inputs(tmp_path: Path):
    stage0 = _stage0()
''',
        '''def _write_cli_inputs(
    tmp_path: Path,
    *,
    implementation_revision: str,
):
    stage0 = _stage0()
''',
    )
    replace_once(
        execution_test,
        "build CLI artifacts under current revision",
        "    for artifact in _artifacts(stage0):\n",
        "    for artifact in _artifacts(\n"
        "        stage0, implementation_revision=implementation_revision\n"
        "    ):\n",
    )
    replace_once(
        execution_test,
        "pass current revision to CLI fixtures",
        "    provider, ledger, artifact_args = _write_cli_inputs(tmp_path)\n",
        "    provider, ledger, artifact_args = _write_cli_inputs(\n"
        "        tmp_path, implementation_revision=revision\n"
        "    )\n",
    )
    append_once(
        execution_test,
        "def test_stage0_loader_recomputes_declared_identities(",
        '''

def test_stage0_loader_recomputes_declared_identities(tmp_path: Path) -> None:
    root = _repository_root()
    source = (
        root
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )
    protocol = root / "protocols/deform360_official_hub_visuotactile_v1.json"
    base = json.loads(source.read_text(encoding="utf-8"))

    selection_drift = json.loads(json.dumps(base))
    selection_drift["selection"]["calibration"][0]["episode_id"] += 1
    changed = tmp_path / "selection-drift.json"
    changed.write_text(json.dumps(selection_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_sha256"):
        load_deform360_stage0_selection(changed)

    content_drift = json.loads(json.dumps(base))
    content_drift["available_raw_object_count"] += 1
    changed = tmp_path / "content-drift.json"
    changed.write_text(json.dumps(content_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="content_selection_sha256"):
        load_deform360_stage0_selection(changed)

    artifact_drift = json.loads(json.dumps(base))
    artifact_drift["implementation_revision"] = "f" * 40
    changed = tmp_path / "artifact-drift.json"
    changed.write_text(json.dumps(artifact_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="selection_artifact_sha256"):
        load_deform360_stage0_selection(changed)

    protocol_drift = json.loads(json.dumps(base))
    protocol_drift["protocol_sha256"] = "0" * 64
    changed = tmp_path / "protocol-drift.json"
    changed.write_text(json.dumps(protocol_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        load_deform360_stage0_selection(changed, protocol_path=protocol)
''',
    )

    boundary_test = "tests/test_deform360_calibration_execution_boundaries.py"
    replace_once(
        boundary_test,
        "parameterize boundary artifact revision fixture",
        '''def _artifacts(
    stage0: execution.Deform360Stage0SelectionV1,
) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
''',
        '''def _artifacts(
    stage0: execution.Deform360Stage0SelectionV1,
    *,
    implementation_revision: str = "c" * 40,
) -> tuple[Deform360CalibrationArtifactRefV1, ...]:
    groups = tuple(unit.object_id for unit in stage0.calibration_units)
''',
    )
    replace_once(
        boundary_test,
        "use parameterized boundary artifact revision",
        '            implementation_revision="a" * 40,\n',
        '            implementation_revision=implementation_revision,\n',
    )
    append_once(
        boundary_test,
        "def test_cli_rejects_unreviewed_selection_and_runtime_code(",
        '''

def test_cli_rejects_unreviewed_selection_and_runtime_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _repository_root()
    committed = _stage0_path()
    substituted = tmp_path / "selection.json"
    substituted.write_bytes(committed.read_bytes() + b"\\n")
    with pytest.raises(ValueError, match="committed lock"):
        cli._verify_committed_selection_lock(  # noqa: SLF001
            root,
            substituted,
        )

    cli._verify_runtime_sources(root)  # noqa: SLF001
    module_name = next(iter(cli._RUNTIME_MODULE_SOURCES))  # noqa: SLF001
    monkeypatch.setitem(
        cli._RUNTIME_MODULE_SOURCES,  # noqa: SLF001
        module_name,
        "pyproject.toml",
    )
    with pytest.raises(ValueError, match="runtime source bytes"):
        cli._verify_runtime_sources(root)  # noqa: SLF001
''',
    )

    visual_test = "tests/test_deform360_visual_provider_lock.py"
    append_once(
        visual_test,
        "def test_visual_lock_writes_are_non_overwriting(",
        '''

def test_visual_lock_writes_are_non_overwriting(tmp_path: Path) -> None:
    provider_path = tmp_path / "provider.json"
    provider = _provider_lock()
    save_deform360_visual_provider_lock(provider_path, provider)
    with pytest.raises(FileExistsError):
        save_deform360_visual_provider_lock(provider_path, provider)
    save_deform360_visual_provider_lock(
        provider_path,
        provider,
        overwrite=True,
    )
    assert load_deform360_visual_provider_lock(provider_path) == provider

    calibration_path = tmp_path / "calibration.json"
    calibration = _calibration_lock()
    save_deform360_visual_calibration_lock(calibration_path, calibration)
    with pytest.raises(FileExistsError):
        save_deform360_visual_calibration_lock(calibration_path, calibration)
    save_deform360_visual_calibration_lock(
        calibration_path,
        calibration,
        overwrite=True,
    )
    assert load_deform360_visual_calibration_lock(calibration_path) == calibration

    with pytest.raises(TypeError, match="Deform360VisualProviderLockV1"):
        save_deform360_visual_provider_lock(
            tmp_path / "bad-provider.json",
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Deform360VisualCalibrationLockV1"):
        save_deform360_visual_calibration_lock(
            tmp_path / "bad-calibration.json",
            object(),  # type: ignore[arg-type]
        )
''',
    )


def main() -> None:
    patch_stage0_execution()
    patch_bundle_revision_binding()
    patch_cli_runtime_and_selection_guards()
    patch_visual_lock_writers()
    patch_tests()


if __name__ == "__main__":
    main()

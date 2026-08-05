from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one replacement in {path}, found {count}")
    source.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "tests/test_deform360_calibration_execution.py",
    '''    protocol_drift = json.loads(json.dumps(base))
    protocol_drift["protocol_sha256"] = "0" * 64
    changed = tmp_path / "protocol-drift.json"
    changed.write_text(json.dumps(protocol_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        load_deform360_stage0_selection(changed, protocol_path=protocol)
''',
    '''    protocol_payload = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_payload["status"] = "changed-after-selection"
    changed_protocol = tmp_path / "protocol-drift.json"
    changed_protocol.write_text(json.dumps(protocol_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol_sha256"):
        load_deform360_stage0_selection(source, protocol_path=changed_protocol)
''',
)

replace_once(
    "tests/test_deform360_calibration_execution_boundaries.py",
    '''    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    with pytest.raises(ValueError, match="outside the Git checkout"):
''',
    '''    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(cli, "_verify_runtime_sources", lambda _repository: None)
    monkeypatch.setattr(
        cli,
        "_verify_committed_selection_lock",
        lambda _repository, _selection: None,
    )
    with pytest.raises(ValueError, match="outside the Git checkout"):
''',
)

replace_once(
    "tests/test_deform360_calibration_execution_boundaries.py",
    '''    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(cli, "_artifact_mapping", lambda _values: {})
''',
    '''    monkeypatch.setattr(cli, "_verify_repository", lambda *_args, **_kwargs: "a" * 40)
    monkeypatch.setattr(cli, "_verify_runtime_sources", lambda _repository: None)
    monkeypatch.setattr(
        cli,
        "_verify_committed_selection_lock",
        lambda _repository, _selection: None,
    )
    monkeypatch.setattr(cli, "_artifact_mapping", lambda _values: {})
''',
)

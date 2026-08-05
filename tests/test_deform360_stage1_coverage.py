from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bayesian_phystwin.cli import deform360_stage1_control as stage1_cli
from bayesian_phystwin.deform360_stage1_control import (
    _cohort_units,
    _selection_units,
    _sha256_file,
    _unit_from_selection,
    _validate_amendment,
    _validate_calibration_design,
    _validate_stage0_selection,
)

_SELECTION_ARTIFACT_SHA256 = (
    "dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82"
)
_CANONICAL_SELECTION_SHA256 = (
    "b28daf8477e214cb74a4d250ef5eea8f9f1a014aec10487699ac0ce063961222"
)
_CONTENT_SELECTION_SHA256 = (
    "f3d3ac25020ec85cad3fadf097259930437baae2b50b4c7f21f61d4823fc649b"
)
_DATASET_REVISION = "f804696d7a133908c7497ffdab43819d879b5cbc"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _selection_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "locks"
        / "deform360_official_hub_visuotactile_v1_selection.json"
    )


def _amendment_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_visual_provider_lock.json"
    )


def _calibration_design_path() -> Path:
    return (
        _repository_root()
        / "protocols"
        / "amendments"
        / "deform360_official_hub_visuotactile_v1_calibration_separation.json"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_reidentified_selection(path: Path, value: dict[str, Any]) -> None:
    value["selection_sha256"] = _canonical_sha256(value["selection"])
    content = dict(value)
    for field_name in (
        "content_selection_sha256",
        "implementation_revision",
        "selection_artifact_sha256",
    ):
        content.pop(field_name, None)
    value["content_selection_sha256"] = _canonical_sha256(content)
    artifact = dict(value)
    artifact.pop("selection_artifact_sha256", None)
    value["selection_artifact_sha256"] = _canonical_sha256(artifact)
    _write_json(path, value)


def test_low_level_stage1_inputs_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        _sha256_file(tmp_path / "missing.json", label="missing input")

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        _sha256_file(link, label="linked input")

    with pytest.raises(ValueError, match="must be a sequence"):
        _cohort_units(
            "not-a-cohort",  # type: ignore[arg-type]
            name="cohort",
            expected_per_stratum=1,
        )
    with pytest.raises(ValueError, match="must contain"):
        _cohort_units((), name="cohort", expected_per_stratum=1)
    with pytest.raises(ValueError, match="JSON object"):
        _unit_from_selection([], name="unit")
    with pytest.raises(ValueError, match="JSON array"):
        _selection_units({"calibration": ()}, role="calibration")


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("selection", [], "selection cohorts"),
        ("dataset", [], "Stage-0 dataset"),
        ("official_processing", [], "official_processing"),
        ("information_boundary", [], "information_boundary"),
    ],
)
def test_stage0_selection_rejects_non_object_sections(
    tmp_path: Path,
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    selection = json.loads(_selection_path().read_text(encoding="utf-8"))
    selection[field_name] = replacement
    path = tmp_path / f"selection-{field_name}.json"
    _write_reidentified_selection(path, selection)

    with pytest.raises(ValueError, match=message):
        _validate_stage0_selection(path)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("parent_protocol", "parent_protocol"),
        ("selection_lock", "selection_lock"),
        ("finite_group_calibration_design", "calibration design"),
        ("information_boundary", "information_boundary"),
    ],
)
def test_visual_provider_amendment_rejects_non_object_sections(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    amendment = json.loads(_amendment_path().read_text(encoding="utf-8"))
    amendment[field_name] = []
    path = tmp_path / f"amendment-{field_name}.json"
    _write_json(path, amendment)

    with pytest.raises(ValueError, match=message):
        _validate_amendment(
            path,
            selection_artifact_sha256=_SELECTION_ARTIFACT_SHA256,
            canonical_selection_sha256=_CANONICAL_SELECTION_SHA256,
            content_selection_sha256=_CONTENT_SELECTION_SHA256,
            dataset_revision=_DATASET_REVISION,
        )


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("primary_interval", "primary_interval"),
        ("information_order", "information_order"),
        ("access_boundary", "access_boundary"),
    ],
)
def test_finite_group_design_rejects_non_object_sections(
    tmp_path: Path,
    field_name: str,
    message: str,
) -> None:
    design = json.loads(_calibration_design_path().read_text(encoding="utf-8"))
    design[field_name] = []
    path = tmp_path / f"design-{field_name}.json"
    _write_json(path, design)

    with pytest.raises(ValueError, match=message):
        _validate_calibration_design(
            path,
            canonical_selection_sha256=_CANONICAL_SELECTION_SHA256,
        )


def test_cli_scalar_and_metadata_parsers_cover_all_boundaries(
    tmp_path: Path,
) -> None:
    assert stage1_cli._optional_positive_integer("none") is None
    assert stage1_cli._optional_positive_integer("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        stage1_cli._optional_positive_integer("invalid")
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        stage1_cli._optional_positive_integer("0")

    metadata_path = tmp_path / "metadata.json"
    _write_json(metadata_path, {"operator": "test"})
    assert stage1_cli._metadata(metadata_path) == {"operator": "test"}


def test_cli_seal_without_summary_output_covers_optional_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = object()
    provider_lock = object()
    bundle = object()
    calibration_lock = object()
    saved: list[tuple[object, Path, bool]] = []
    printed: list[dict[str, object]] = []

    monkeypatch.setattr(stage1_cli, "load_deform360_stage1_plan", lambda _: plan)
    monkeypatch.setattr(stage1_cli, "_load_provider_lock", lambda _: provider_lock)
    monkeypatch.setattr(
        stage1_cli,
        "load_deform360_calibration_bundle",
        lambda _: bundle,
    )
    monkeypatch.setattr(
        stage1_cli,
        "derive_deform360_visual_calibration_lock",
        lambda **_: calibration_lock,
    )
    monkeypatch.setattr(
        stage1_cli,
        "save_deform360_visual_calibration_lock_atomic",
        lambda lock, path, *, overwrite: saved.append((lock, path, overwrite)),
    )
    monkeypatch.setattr(
        stage1_cli,
        "verify_deform360_stage1_seal",
        lambda **_: {"seal_id": "sealed"},
    )
    monkeypatch.setattr(stage1_cli, "_print_json", printed.append)

    output = tmp_path / "visual-calibration-lock.json"
    arguments = argparse.Namespace(
        plan=tmp_path / "plan.json",
        provider_lock=tmp_path / "provider.json",
        calibration_bundle=tmp_path / "bundle.json",
        output=output,
        summary_output=None,
        overwrite=False,
    )
    assert stage1_cli._seal(arguments) == 0
    assert saved == [(calibration_lock, output, False)]
    assert printed == [{"seal_id": "sealed", "output": str(output.resolve())}]

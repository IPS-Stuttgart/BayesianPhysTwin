from __future__ import annotations

import json

import numpy as np
import pytest

from bayesian_phystwin.phystwin_spring_overlay import (
    SPRING_OVERLAY_CONTRACT,
    build_spring_overlay_checkpoint,
)


def _load(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_spring_overlay_replaces_only_spring_field(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    field = tmp_path / "field.npy"
    output = tmp_path / "overlay.pt"
    summary = tmp_path / "overlay.json"
    source_values = torch.tensor([1000.0, 2000.0, 3000.0])
    torch.save({"spring_Y": source_values, "other": torch.tensor([7.0])}, source)
    candidate = np.array([900.0, 2200.0, 3100.0], dtype=np.float32)
    np.save(field, candidate, allow_pickle=False)

    result = build_spring_overlay_checkpoint(
        source, field, output, summary_path=summary
    )

    original = _load(torch, source)
    overlaid = _load(torch, output)
    torch.testing.assert_close(original["spring_Y"], source_values)
    torch.testing.assert_close(
        overlaid["spring_Y"], torch.as_tensor(candidate)
    )
    torch.testing.assert_close(overlaid["other"], original["other"])
    assert result["contract"] == SPRING_OVERLAY_CONTRACT
    assert result["replacement_scope"] == "spring_Y only"
    assert not result["identity_field"]
    assert json.loads(summary.read_text(encoding="utf-8"))["contract"] == (
        SPRING_OVERLAY_CONTRACT
    )
    assert summary.with_suffix(".json.sha256").is_file()


def test_spring_overlay_interpolates_in_log_space(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    field = tmp_path / "field.npy"
    torch.save({"spring_Y": torch.tensor([100.0, 400.0])}, source)
    np.save(field, np.array([400.0, 100.0]), allow_pickle=False)

    result = build_spring_overlay_checkpoint(
        source,
        field,
        tmp_path / "half.pt",
        strength=0.5,
    )

    applied = _load(torch, tmp_path / "half.pt")["spring_Y"]
    torch.testing.assert_close(applied, torch.tensor([200.0, 200.0]))
    assert result["proposal_strength"] == 0.5
    assert result["spring_ratio"]["minimum"] == pytest.approx(0.5)
    assert result["spring_ratio"]["maximum"] == pytest.approx(2.0)


def test_zero_strength_is_exact_checkpoint_identity(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    field = tmp_path / "field.npy"
    source_values = torch.tensor([100.0, 400.0])
    torch.save({"spring_Y": source_values}, source)
    np.save(field, np.array([400.0, 100.0]), allow_pickle=False)

    result = build_spring_overlay_checkpoint(
        source,
        field,
        tmp_path / "identity.pt",
        strength=0.0,
    )

    torch.testing.assert_close(
        _load(torch, tmp_path / "identity.pt")["spring_Y"], source_values
    )
    assert result["identity_field"] is True


def test_spring_overlay_rejects_invalid_strength(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    field = tmp_path / "field.npy"
    torch.save({"spring_Y": torch.ones(2)}, source)
    np.save(field, np.ones(2), allow_pickle=False)

    with pytest.raises(ValueError, match="strength"):
        build_spring_overlay_checkpoint(
            source, field, tmp_path / "output.pt", strength=1.1
        )


@pytest.mark.parametrize(
    "candidate, message",
    [
        (np.array([1.0, 2.0]), "disagree"),
        (np.array([1.0, 0.0, 3.0]), "positive"),
        (np.array([1.0, np.nan, 3.0]), "finite"),
    ],
)
def test_spring_overlay_rejects_invalid_fields(
    tmp_path, candidate, message
) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.pt"
    field = tmp_path / "field.npy"
    torch.save({"spring_Y": torch.ones(3)}, source)
    np.save(field, candidate, allow_pickle=False)
    with pytest.raises(ValueError, match=message):
        build_spring_overlay_checkpoint(source, field, tmp_path / "output.pt")

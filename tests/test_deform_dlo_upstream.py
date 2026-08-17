import hashlib
from pathlib import Path

import pytest

from bayesian_phystwin.deform_dlo_upstream import (
    load_deform_dlo_initialization,
)


def _branch(dlo_type: str, node_count: int, bend: float, twist: float) -> str:
    vertices = tuple(
        (float(index), float(index) + 0.25, float(index) + 0.5)
        for index in range(node_count)
    )
    keyword = "if" if dlo_type == "DLO1" else "elif"
    return f'''    {keyword} DLO_type == "{dlo_type}":
        rest_vert = torch.tensor({vertices!r}).unsqueeze(dim=0).to(device)
        rest_vert = torch.cat((rest_vert[:, :, 0], rest_vert[:, :, 2]), dim=-1)
        DEFORM_sim.rest_vert = nn.Parameter(rest_vert)
        DEFORM_sim.DEFORM_func.bend_stiffness = nn.Parameter(
            {bend!r} * torch.ones((1, n_edge), device=device)
        )
        DEFORM_sim.DEFORM_func.twist_stiffness = nn.Parameter(
            {twist!r} * torch.ones((1, n_edge), device=device)
        )
'''


def _write_fixture(path: Path, *, duplicate_dlo2: bool = False) -> bytes:
    dlo2 = _branch("DLO2", 12, 5e-4, 3e-5)
    if duplicate_dlo2:
        dlo2 += _branch("DLO2", 12, 5e-4, 3e-5).replace("elif", "if", 1)
    payload = ("def train(DLO_type):\n" + _branch("DLO1", 13, 5e-5, 2e-5) + dlo2).encode()
    path.write_bytes(payload)
    return payload


def test_extracts_official_dlo_specific_initialization(tmp_path: Path) -> None:
    source = tmp_path / "train_DEFORM.py"
    payload = _write_fixture(source)

    dlo1 = load_deform_dlo_initialization(source, "DLO1")
    dlo2 = load_deform_dlo_initialization(source, "DLO2")

    assert dlo1.node_count == 13
    assert dlo1.bend_stiffness == pytest.approx(5e-5)
    assert dlo1.twist_stiffness == pytest.approx(2e-5)
    assert dlo2.node_count == 12
    assert dlo2.bend_stiffness == pytest.approx(5e-4)
    assert dlo2.twist_stiffness == pytest.approx(3e-5)
    assert dlo2.rest_vertices_m[1] == pytest.approx((1.0, 1.5, -1.25))
    assert dlo2.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert dlo2.to_record()["contract"] == "official-deform-dlo-initialization-v1"


def test_rejects_unsupported_or_ambiguous_dlo_initialization(tmp_path: Path) -> None:
    source = tmp_path / "train_DEFORM.py"
    _write_fixture(source, duplicate_dlo2=True)

    with pytest.raises(ValueError, match="unsupported"):
        load_deform_dlo_initialization(source, "DLO9")
    with pytest.raises(ValueError, match="2 branches"):
        load_deform_dlo_initialization(source, "DLO2")


def test_rejects_wrong_registered_node_count(tmp_path: Path) -> None:
    source = tmp_path / "train_DEFORM.py"
    branch = _branch("DLO2", 11, 5e-4, 3e-5).replace("elif", "if", 1)
    payload = "def train(DLO_type):\n" + branch
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        load_deform_dlo_initialization(source, "DLO2")

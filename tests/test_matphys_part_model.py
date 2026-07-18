from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from bayesian_phystwin.matphys_part_model import (
    install_part_aware_simple_model,
    summarize_part_spring_ratios,
)


class _Codebook(nn.Module):
    def __init__(self, materials: int, width: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.zeros(materials, width))

    def forward(self, distribution):
        return distribution @ self.codebook


class _FakeSimpleModel(nn.Module):
    def __init__(self, d_mat: int = 2, **kwargs):
        super().__init__()
        del kwargs
        self.motion_encoder = lambda pixels: torch.zeros(1)
        self.material_codebook = _Codebook(3, d_mat)
        self.geo_stats_encoder = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
        self.global_context = nn.Identity()

    def _global_hidden(self, pixel_values, material_dist, geo_stats):
        z_motion = self.motion_encoder(pixel_values)
        z_mat_part = self.material_codebook(material_dist)
        z_geo = self.geo_stats_encoder(geo_stats.view(1, -1)).squeeze(0)
        return torch.cat((z_motion, z_mat_part.mean(0), z_geo)), z_mat_part

    def forward(
        self,
        pixel_values,
        z_geo,
        material_dist,
        edge_part_idx,
        geo_stats=None,
        ctrl_rest_length=None,
        ctrl_part_idx=None,
    ):
        del z_geo, edge_part_idx, ctrl_rest_length, ctrl_part_idx
        hidden, parts = self._global_hidden(pixel_values, material_dist, geo_stats)
        return {"hidden": hidden, "parts": parts}


def _training_namespace():
    return SimpleNamespace(
        SimpleVideoMaterialPhysicsModel=_FakeSimpleModel,
        forward_case=lambda *args, **kwargs: None,
        _unwrap_model=lambda model: model,
    )


def _inputs(features):
    return {
        "pixel_values": torch.zeros(1),
        "z_geo": torch.zeros(1),
        "material_dist": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "edge_part_idx": torch.tensor([0]),
        "part_features": torch.tensor(features, dtype=torch.float32),
        "geo_stats": torch.zeros(1),
    }


def test_zero_part_scale_is_upstream_embedding_identity() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(
        training,
        part_feature_dim=3,
        part_feature_scale=0.0,
    )
    model = model_type(d_mat=2)
    with torch.no_grad():
        model.material_codebook.codebook[0] = torch.tensor([0.25, -0.5])
        model.part_feature_encoder[-1].weight.fill_(1.0)

    first = model(**_inputs([[1, 0, 0], [0, 1, 0]]))
    second = model(**_inputs([[0, 0, 1], [1, 1, 0]]))

    torch.testing.assert_close(first["parts"], second["parts"])
    assert model._active_part_features is None


def test_part_descriptors_change_embeddings_when_enabled() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(
        training,
        part_feature_dim=3,
        part_feature_scale=1.0,
    )
    model = model_type(d_mat=2)
    with torch.no_grad():
        model.part_feature_encoder[-1].weight[0, 0] = 1.0
        model.part_feature_encoder[-1].weight[1, 1] = 1.0

    output = model(**_inputs([[1, 0, 0], [0, 1, 0]]))

    assert not torch.allclose(output["parts"][0], output["parts"][1])


def test_part_adapter_rejects_missing_features() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(training, part_feature_dim=3)
    model = model_type(d_mat=2)
    inputs = _inputs([[1, 0, 0], [0, 1, 0]])
    inputs.pop("part_features")

    with pytest.raises(ValueError, match="part_features"):
        model(**inputs)


def test_part_spring_summary_reports_identity_and_part_changes() -> None:
    summary = summarize_part_spring_ratios(
        np.log([2.0, 2.0, 6.0]),
        np.log([2.0, 2.0, 3.0]),
        [0, 0, 1],
    )

    assert summary["by_part"][0]["mean"] == pytest.approx(1.0)
    assert summary["by_part"][1]["mean"] == pytest.approx(2.0)

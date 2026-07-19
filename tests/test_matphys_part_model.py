from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from bayesian_phystwin.matphys_part_model import (  # noqa: E402
    install_part_aware_simple_model,
    summarize_part_spring_ratios,
)


class _Codebook(nn.Module):
    def __init__(self, materials: int, width: int):
        super().__init__()
        self.codebook = nn.Parameter(torch.zeros(materials, width))

    def forward(self, distribution):
        return distribution @ self.codebook


class _CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values, bool_masked_pos):
        del bool_masked_pos
        self.calls += 1
        pooled = pixel_values.mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
        return SimpleNamespace(last_hidden_state=pooled)


class _CachingMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _CountingBackbone()
        self.projector = nn.Linear(1, 1, bias=False)
        self.tubelet_size = 1
        self.patch_size = 1


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
        _unwrap_model=lambda model: getattr(model, "module", model),
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


def test_frozen_motion_backbone_is_cached_per_video_tensor() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(training, part_feature_dim=3)
    model = model_type(d_mat=2)
    model.motion_encoder = _CachingMotionEncoder()
    inputs = _inputs([[1, 0, 0], [0, 1, 0]])
    pixels = torch.ones(1, 1, 1, 1, 1)
    inputs["pixel_values"] = pixels

    first = model(**inputs)
    second = model(**inputs)
    third = model(**{**inputs, "pixel_values": pixels.clone()})

    torch.testing.assert_close(first["hidden"], second["hidden"])
    torch.testing.assert_close(second["hidden"], third["hidden"])
    assert model.motion_encoder.backbone.calls == 2


def test_public_trunk_initialization_excludes_residual_head_prefixes() -> None:
    seed_model = _FakeSimpleModel(d_mat=2)
    initialization = {
        name: torch.full_like(value, 3.0)
        for name, value in seed_model.state_dict().items()
    }
    training = _training_namespace()
    model_type = install_part_aware_simple_model(
        training,
        part_feature_dim=3,
        initialization_state_dict=initialization,
        initialization_excluded_prefixes=("material_codebook.",),
    )

    model = model_type(d_mat=2)

    torch.testing.assert_close(
        model.material_codebook.codebook,
        torch.zeros_like(model.material_codebook.codebook),
    )
    torch.testing.assert_close(
        model.geo_stats_encoder[0].weight,
        torch.full_like(model.geo_stats_encoder[0].weight, 3.0),
    )
    assert model.matphys_initialization_summary == {
        "provided": True,
        "loaded_tensor_count": len(initialization) - 1,
        "excluded_tensor_count": 1,
        "incompatible_tensor_count": 0,
        "excluded_prefixes": ["material_codebook."],
    }


def test_part_adapter_rejects_missing_features() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(training, part_feature_dim=3)
    model = model_type(d_mat=2)
    inputs = _inputs([[1, 0, 0], [0, 1, 0]])
    inputs.pop("part_features")

    with pytest.raises(ValueError, match="part_features"):
        model(**inputs)


def test_part_forward_case_preserves_distributed_wrapper() -> None:
    training = _training_namespace()
    model_type = install_part_aware_simple_model(training, part_feature_dim=3)
    model = model_type(d_mat=2)

    class Wrapper:
        def __init__(self, module):
            self.module = module
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.module(*args, **kwargs)

    wrapper = Wrapper(model)
    batch = {
        "z_geo": [torch.zeros(1)],
        "material_dist": [
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ],
        "edge_part_idx": [torch.tensor([0])],
        "part_features": [torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])],
        "geo_stats": [torch.zeros(1)],
        "ctrl_rest_length": [torch.empty(0)],
    }

    output = training.forward_case(wrapper, batch, 0, torch.device("cpu"), torch.zeros(1))

    assert wrapper.calls == 1
    assert output["parts"].shape == (2, 2)


def test_part_spring_summary_reports_identity_and_part_changes() -> None:
    summary = summarize_part_spring_ratios(
        np.log([2.0, 2.0, 6.0]),
        np.log([2.0, 2.0, 3.0]),
        [0, 0, 1],
    )

    assert summary["by_part"][0]["mean"] == pytest.approx(1.0)
    assert summary["by_part"][1]["mean"] == pytest.approx(2.0)

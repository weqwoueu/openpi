import numpy as np
import pytest

from openpi.training import weight_loaders


def test_convert_gemma_checkpoint_strips_transformer_prefix(monkeypatch):
    monkeypatch.setattr(weight_loaders, "gm_ckpt", None)
    checkpoint = {
        "transformer/embedder": {"input_embedding": np.zeros((4, 2), dtype=np.float32)},
        "transformer/final_norm": {"scale": np.ones((2,), dtype=np.float32)},
        "transformer/layer_0": {
            "mlp": {
                "gating_einsum": {"w": np.zeros((2, 4, 2), dtype=np.float32)},
                "linear": {"w": np.zeros((4, 2), dtype=np.float32)},
            }
        },
    }

    converted = weight_loaders._maybe_convert_gemma_ckpt_tree(checkpoint)  # noqa: SLF001

    assert set(converted) == {"embedder", "final_norm", "layer_0"}
    np.testing.assert_array_equal(
        converted["embedder"]["input_embedding"],
        checkpoint["transformer/embedder"]["input_embedding"],
    )
    assert converted["layer_0"]["mlp"]["gating_einsum"].shape == (2, 4, 2)
    assert converted["layer_0"]["mlp"]["linear"].shape == (4, 2)


def test_validate_param_coverage_rejects_missing_backbone_weights():
    loaded = {"embedder": {"input_embedding": np.zeros((4, 2), dtype=np.float32)}}
    reference = {
        "embedder": {"input_embedding": np.zeros((4, 2), dtype=np.float32)},
        "final_norm": {"scale": np.ones((2,), dtype=np.float32)},
    }

    with pytest.raises(ValueError, match=r"matched 1/2, missing=1"):
        weight_loaders._validate_param_coverage("Gemma", loaded, reference)  # noqa: SLF001


def test_convert_gemma_checkpoint_rejects_key_collision(monkeypatch):
    monkeypatch.setattr(weight_loaders, "gm_ckpt", None)
    checkpoint = {
        "transformer/layer_0/mlp/linear": np.zeros((4, 2), dtype=np.float32),
        "transformer/layer_0/mlp/linear/w": np.ones((4, 2), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="key collision"):
        weight_loaders._maybe_convert_gemma_ckpt_tree(checkpoint)  # noqa: SLF001


def test_validate_param_coverage_rejects_non_array_weight():
    loaded = {"final_norm": {"scale": [1.0, 1.0]}}
    reference = {"final_norm": {"scale": np.ones((2,), dtype=np.float32)}}

    with pytest.raises(ValueError, match=r"shape_mismatch=1"):
        weight_loaders._validate_param_coverage("Gemma", loaded, reference)  # noqa: SLF001

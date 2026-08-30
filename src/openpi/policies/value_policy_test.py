import numpy as np
import pytest

from openpi.policies import value_policy


def test_value_inputs_squeezes_scalar_like_value():
    transform = value_policy.ValueInputs()
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the block",
        "value": np.array([-0.5], dtype=np.float32),
    }

    transformed = transform(sample)

    assert np.asarray(transformed["value"]).shape == ()
    assert float(transformed["value"]) == pytest.approx(-0.5)


def test_value_inputs_rejects_non_scalar_value():
    transform = value_policy.ValueInputs()
    sample = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "prompt": "pick up the block",
        "value": np.array([-0.5, -0.25], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="value must be scalar-like"):
        transform(sample)


def test_value_inputs_accepts_piperx_camera_keys():
    transform = value_policy.ValueInputs()
    sample = {
        "observation.images.cam_head": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.cam_wrist": np.ones((4, 4, 3), dtype=np.uint8),
        "value": np.float32(-0.25),
    }

    transformed = transform(sample)

    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "wrist_0_rgb": np.True_,
    }
    assert transformed["image"]["base_0_rgb"].shape == (4, 4, 3)
    assert transformed["image"]["wrist_0_rgb"].shape == (4, 4, 3)

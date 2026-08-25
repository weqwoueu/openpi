import numpy as np
import pytest

from openpi.policies import piper_policy


def _valid_data() -> dict:
    actions = np.arange(35, dtype=np.float32).reshape(5, 7) / 10
    return {
        "state": np.arange(7, dtype=np.float32),
        "actions": actions,
        "images": {
            "cam_high": np.full((480, 640, 3), 0.5, dtype=np.float32),
            "cam_wrist": np.full((3, 480, 640), 127, dtype=np.uint8),
            "ignored_debug_camera": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "prompt": "insert the black plug",
    }


def test_piper_inputs_preserve_absolute_actions_and_map_cameras():
    data = _valid_data()
    result = piper_policy.PiperInputs()(data)

    np.testing.assert_array_equal(result["state"], data["state"])
    np.testing.assert_array_equal(result["actions"], data["actions"])
    assert result["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["left_wrist_0_rgb"].shape == (480, 640, 3)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert not result["image"]["right_wrist_0_rgb"].any()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("state", np.zeros(6, dtype=np.float32)),
        ("state", np.array([0, 0, 0, 0, 0, 0, np.nan], dtype=np.float32)),
        ("actions", np.zeros(7, dtype=np.float32)),
        ("actions", np.zeros((5, 6), dtype=np.float32)),
        ("actions", np.full((5, 7), np.inf, dtype=np.float32)),
    ],
)
def test_piper_inputs_reject_invalid_state_or_actions(key, value):
    data = _valid_data()
    data[key] = value
    with pytest.raises(ValueError, match="shape|finite"):
        piper_policy.PiperInputs()(data)


def test_piper_inputs_require_both_cameras():
    data = _valid_data()
    del data["images"]["cam_wrist"]
    with pytest.raises(ValueError, match="cam_wrist"):
        piper_policy.PiperInputs()(data)


def test_piper_outputs_extract_first_seven_absolute_dimensions():
    model_actions = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    result = piper_policy.PiperOutputs()({"actions": model_actions})
    np.testing.assert_array_equal(result["actions"], model_actions[:, :7])


def test_piper_outputs_reject_nonfinite_actions():
    model_actions = np.zeros((5, 32), dtype=np.float32)
    model_actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        piper_policy.PiperOutputs()({"actions": model_actions})

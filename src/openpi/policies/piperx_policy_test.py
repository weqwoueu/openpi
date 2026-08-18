import numpy as np
import pytest

from openpi.policies import piperx_policy


def test_piperx_inputs_maps_three_cameras_and_actions() -> None:
    data = piperx_policy.make_piperx_example()
    data["actions"] = np.ones((30, piperx_policy.PIPERX_ACTION_DIM), dtype=np.float32)

    result = piperx_policy.PiperXInputs()(data)

    assert result["state"].shape == (piperx_policy.PIPERX_ACTION_DIM,)
    assert result["actions"].shape == (30, piperx_policy.PIPERX_ACTION_DIM)
    assert set(result["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (224, 224, 3) for image in result["image"].values())
    assert all(result["image_mask"].values())


def test_piperx_inputs_accepts_rtc_actions() -> None:
    data = piperx_policy.make_piperx_example()
    data["rtc_actions"] = np.ones((30, piperx_policy.PIPERX_ACTION_DIM), dtype=np.float32)

    result = piperx_policy.PiperXInputs()(data)

    assert "rtc_actions" not in result
    assert result["actions"].shape == (30, piperx_policy.PIPERX_ACTION_DIM)


def test_piperx_inputs_masks_missing_wrist_camera() -> None:
    data = piperx_policy.make_piperx_example()
    del data["images"]["cam_right_wrist"]

    result = piperx_policy.PiperXInputs()(data)

    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert np.count_nonzero(result["image"]["right_wrist_0_rgb"]) == 0


def test_piperx_inputs_rejects_wrong_action_dim() -> None:
    data = piperx_policy.make_piperx_example()
    data["actions"] = np.zeros((30, 13), dtype=np.float32)

    with pytest.raises(ValueError, match="last dimension 14"):
        piperx_policy.PiperXInputs()(data)


def test_piperx_inputs_rejects_ambiguous_actions() -> None:
    data = piperx_policy.make_piperx_example()
    data["actions"] = np.zeros((30, 14), dtype=np.float32)
    data["rtc_actions"] = np.zeros((30, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="both"):
        piperx_policy.PiperXInputs()(data)


def test_piperx_outputs_extracts_14_dims() -> None:
    result = piperx_policy.PiperXOutputs()({"actions": np.ones((30, 32), dtype=np.float32)})

    assert result["actions"].shape == (30, piperx_policy.PIPERX_ACTION_DIM)

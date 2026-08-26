import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "example" / "deploy" / "piper_single_on_PI0_websocket.py"
SPEC = importlib.util.spec_from_file_location("piper_single_on_pi0_websocket", SCRIPT_PATH)
CLIENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CLIENT)


def _robot_data():
    return [
        {
            "left_arm": {
                "joint": np.arange(6, dtype=np.float32) / 10,
                "gripper": np.float32(0.75),
            }
        },
        {
            "cam_head": {"color": np.full((480, 640, 3), 10, dtype=np.uint8)},
            "cam_wrist": {"color": np.full((480, 640, 3), 20, dtype=np.uint8)},
        },
    ]


def test_input_transform_matches_root_piper_policy_contract():
    observation = CLIENT.input_transform(_robot_data(), "insert the plug", adv_ind="positive")

    assert set(observation) == {"state", "images", "prompt", "adv_ind"}
    assert set(observation["images"]) == {"cam_high", "cam_wrist"}
    assert observation["state"].shape == (7,)
    assert observation["state"].dtype == np.float32
    assert observation["images"]["cam_high"].shape == (224, 224, 3)
    assert observation["images"]["cam_wrist"].shape == (224, 224, 3)
    assert observation["adv_ind"] == "positive"


def test_input_transform_omits_advantage_for_sft():
    observation = CLIENT.input_transform(_robot_data(), "insert the plug")

    assert "adv_ind" not in observation


def test_output_transform_uses_first_seven_absolute_values_without_clipping():
    action = np.array([-4.0, 1.2, -1.3, 2.4, 3.5, -6.0, 1.25, 99.0], dtype=np.float32)
    move_data = CLIENT.output_transform(action)
    command = move_data["arm"]["left_arm"]

    np.testing.assert_array_equal(command["joint"], action[:6])
    assert command["gripper"] == pytest.approx(1.25)


@pytest.mark.parametrize("action", [np.zeros(6), np.array([0, 0, 0, 0, 0, 0, np.nan])])
def test_output_transform_rejects_invalid_action(action):
    with pytest.raises(ValueError, match="at least|finite"):
        CLIENT.output_transform(action)


def test_negative_one_episode_count_is_unbounded():
    assert list(itertools.islice(CLIENT._episode_indices(-1), 4)) == [0, 1, 2, 3]  # noqa: SLF001
    assert list(CLIENT._episode_indices(3)) == [0, 1, 2]  # noqa: SLF001


def test_select_action_chunk_requires_nonempty_action_matrix():
    actions = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    selected = CLIENT._select_action_chunk(actions, 3)  # noqa: SLF001
    np.testing.assert_array_equal(selected, actions[:3])

    with pytest.raises(ValueError, match="horizon"):
        CLIENT._select_action_chunk(np.empty((0, 32), dtype=np.float32), 3)  # noqa: SLF001

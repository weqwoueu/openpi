import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "piperx" / "replay_lerobot_episode.py"
SPEC = importlib.util.spec_from_file_location("replay_lerobot_episode", SCRIPT_PATH)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def write_dataset(root: Path, *, feature_key: str = "action") -> np.ndarray:
    actions = np.arange(21, dtype=np.float32).reshape(3, 7) / 20.0
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "robot_type": "piperx",
        "fps": 10,
        "features": {feature_key: {"dtype": "float32", "shape": [7]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    table = pa.table({feature_key: actions.tolist()})
    pq.write_table(table, root / "data" / "chunk-000" / "episode_000000.parquet")
    return actions


def test_load_episode_actions_reads_standard_action(tmp_path):
    expected = write_dataset(tmp_path)

    actual = REPLAY.load_episode_actions(tmp_path, episode_index=0, expected_fps=10)

    np.testing.assert_allclose(actual, expected)
    assert actual.dtype == np.float32


def test_load_episode_actions_rejects_legacy_actions_key(tmp_path):
    write_dataset(tmp_path, feature_key="actions")

    with pytest.raises(ValueError, match="standard 'action'"):
        REPLAY.load_episode_actions(tmp_path, episode_index=0, expected_fps=10)


def test_replay_actions_sends_joint_then_gripper(monkeypatch):
    calls = []

    class FakeController:
        def set_joint(self, joint, speed_percent):
            calls.append(("joint", np.asarray(joint).copy(), speed_percent))

        def set_gripper(self, gripper):
            calls.append(("gripper", gripper))

    actions = np.arange(14, dtype=np.float32).reshape(2, 7) / 10.0
    monkeypatch.setattr(REPLAY.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(REPLAY.time, "sleep", lambda _: None)

    REPLAY.replay_actions(FakeController(), actions, fps=10, speed_percent=100)

    assert [call[0] for call in calls] == ["joint", "gripper", "joint", "gripper"]
    np.testing.assert_allclose(calls[0][1], actions[0, :6])
    assert calls[0][2] == 100
    assert calls[1][1] == pytest.approx(float(actions[0, 6]))

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from robot.data.collect_lerobot_rl import ACTION_KEY, STATE_KEY, CollectLeRobotRL


def test_real_dataset_writes_standard_fields_and_two_videos(tmp_path):
    collector = CollectLeRobotRL(
        repo_id="piperx/video_smoke",
        output_dir=str(tmp_path),
        task_name="video smoke",
        fps=10,
        robot_type="piperx",
        state_dim=7,
        action_dim=7,
        image_size=(64, 96),
        camera_keys={
            "cam_head": "observation.images.cam_head",
            "cam_wrist": "observation.images.cam_wrist",
        },
        move_check=False,
    )

    for step in range(2):
        controllers = {
            "left_arm": {
                "joint": np.linspace(0.0, 0.5, 6, dtype=np.float32) + step * 0.01,
                "gripper": np.float32(0.3 + step * 0.1),
            }
        }
        sensors = {
            "cam_head": {"color": np.full((64, 96, 3), 40 + step, dtype=np.uint8)},
            "cam_wrist": {"color": np.full((64, 96, 3), 120 + step, dtype=np.uint8)},
        }
        action = {
            "joint": np.linspace(-0.5, 0.0, 6, dtype=np.float32) - step * 0.02,
            "gripper": np.float32(0.6 - step * 0.1),
        }
        collector.collect(
            controllers,
            sensors,
            action_data=action,
            is_intervention=True,
        )

    collector.save_episode(success=True, adv_ind_value="positive")
    collector.dataset.stop_image_writer()

    root = tmp_path / "piperx" / "video_smoke"
    info = json.loads((root / "meta" / "info.json").read_text())
    video_files = sorted(root.glob("videos/**/*.mp4"))
    parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(parquet_path, columns=[STATE_KEY, ACTION_KEY])

    assert info["robot_type"] == "piperx"
    assert info["total_frames"] == 2
    assert info["total_videos"] == 2
    assert len(video_files) == 2
    assert all(path.stat().st_size > 0 for path in video_files)
    assert not (root / "images").exists()
    assert table.column_names == [STATE_KEY, ACTION_KEY]

    actions = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
    states = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
    assert not np.allclose(actions[0], states[1])

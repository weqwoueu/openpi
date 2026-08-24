#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq


ACTION_KEY = "action"
EXPECTED_ACTION_DIM = 7


def load_episode_actions(dataset_dir: Path, episode_index: int, expected_fps: float) -> np.ndarray:
    dataset_dir = dataset_dir.expanduser().resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("robot_type") != "piperx":
        raise ValueError(f"Expected robot_type=piperx, got {info.get('robot_type')!r}")

    dataset_fps = float(info.get("fps", 0))
    if dataset_fps != float(expected_fps):
        raise ValueError(f"Replay FPS {expected_fps:g} does not match dataset FPS {dataset_fps:g}")

    features = info.get("features", {})
    action_feature = features.get(ACTION_KEY)
    if action_feature is None:
        raise ValueError("Dataset does not contain the standard 'action' feature")
    if action_feature.get("shape") != [EXPECTED_ACTION_DIM]:
        raise ValueError(f"Expected action shape [7], got {action_feature.get('shape')!r}")

    matches = sorted((dataset_dir / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for episode {episode_index}, found {len(matches)}"
        )

    table = pq.read_table(matches[0], columns=[ACTION_KEY])
    actions = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != EXPECTED_ACTION_DIM or actions.shape[0] == 0:
        raise ValueError(f"Expected non-empty action array [T, 7], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Dataset action contains non-finite values")
    return actions


def replay_actions(controller, actions: np.ndarray, fps: float, speed_percent: int) -> None:
    period = 1.0 / float(fps)
    started_at = time.monotonic()
    for frame_index, action in enumerate(actions):
        deadline = started_at + frame_index * period
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        controller.set_joint(action[:6], speed_percent=speed_percent)
        controller.set_gripper(float(action[6]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a PiperX LeRobot episode in MIT mode")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--can-name", default="can_left_slave")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument(
        "--reset-joint-position",
        type=float,
        nargs=6,
        default=[0.0, 1.0, -1.0, 1.0, 0.0, 0.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_index < 0:
        raise ValueError("episode-index must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    actions = load_episode_actions(args.dataset_dir, args.episode_index, args.fps)
    print(f"Dataset: {args.dataset_dir.expanduser().resolve()}")
    print(f"Episode: {args.episode_index}, frames: {len(actions)}, fps: {args.fps:g}")
    print(f"Action span: {np.array2string(np.ptp(actions, axis=0), precision=4)}")
    print(f"Follower CAN: {args.can_name}, control mode: MIT (0xAD)")

    from robot.controller.Piper_controller import PiperController

    controller = PiperController("replay_follower", use_mit_mode=True)
    controller.set_up(args.can_name)
    controller.set_gripper_effort(args.gripper_effort)
    controller.reset(np.asarray(args.reset_joint_position, dtype=float), speed_percent=100)
    time.sleep(1.0)

    input("Press Enter to replay the episode...")
    try:
        replay_actions(controller, actions, fps=args.fps, speed_percent=100)
    except KeyboardInterrupt:
        print("\nReplay interrupted")
        return
    print("Replay complete")


if __name__ == "__main__":
    main()

from pathlib import Path
import sys
import threading
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from robot.utils.teleop_filter import EmaSlewFilter, FixedRateControlLoop


def test_piperx_control_loop_keeps_running_while_capture_thread_is_blocked():
    fourth_command_sent = threading.Event()
    command_count = 0

    def send_command():
        nonlocal command_count
        command_count += 1
        if command_count == 4:
            fourth_command_sent.set()

    control_loop = FixedRateControlLoop(control_fps=30, step=send_command)
    control_loop.start()
    try:
        # This wait stands in for the worker blocking in RealSense wait_for_frames().
        assert fourth_command_sent.wait(timeout=0.5)
    finally:
        control_loop.stop()

    assert command_count >= 4


def test_control_loop_stop_keeps_live_thread_until_callback_finishes():
    callback_started = threading.Event()
    release_callback = threading.Event()

    def blocking_command():
        callback_started.set()
        release_callback.wait(timeout=1.0)

    control_loop = FixedRateControlLoop(control_fps=30, step=blocking_command)
    control_loop.start()
    assert callback_started.wait(timeout=0.5)

    assert control_loop.stop(timeout=0.01) is False
    release_callback.set()
    assert control_loop.stop(timeout=0.5) is True


def test_control_loop_exposes_an_isolated_latest_action_snapshot():
    action_ready = threading.Event()
    sent_action = {"joint": [0.1] * 6, "gripper": 0.25}

    def send_command():
        action_ready.set()
        return sent_action

    control_loop = FixedRateControlLoop(control_fps=30, step=send_command)
    control_loop.start()
    try:
        assert action_ready.wait(timeout=0.5)
        latest = None
        deadline = time.monotonic() + 0.5
        while latest is None and time.monotonic() < deadline:
            latest = control_loop.get_latest()
            time.sleep(0.005)
        assert latest == sent_action

        latest["joint"][0] = 99.0
        assert control_loop.get_latest()["joint"][0] == 0.1
    finally:
        control_loop.stop()


def test_ema_then_slew_matches_robocoin_piperx_settings():
    action_filter = EmaSlewFilter(
        ema_alpha=0.8,
        max_joint_step=0.04,
        max_gripper_step=0.025 / 0.07,
    )
    action_filter.seed(np.zeros(6), 0.0)

    output = action_filter.process(
        {
            "joint": [0.1, -0.1, 0.02, -0.02, 0.0, 0.2],
            "gripper": 1.0,
        }
    )

    np.testing.assert_allclose(
        output["joint"],
        [0.04, -0.04, 0.016, -0.016, 0.0, 0.04],
    )
    assert np.isclose(output["gripper"], 0.025 / 0.07)


def test_ema_state_is_independent_from_previous_sent_state():
    action_filter = EmaSlewFilter(
        ema_alpha=0.8,
        max_joint_step=0.04,
        max_gripper_step=1.0,
    )
    action_filter.seed(np.zeros(6), 0.0)

    first = action_filter.process({"joint": [0.2] * 6, "gripper": 0.0})
    second = action_filter.process({"joint": [0.0] * 6, "gripper": 0.0})

    np.testing.assert_allclose(first["joint"], [0.04] * 6)
    np.testing.assert_allclose(second["joint"], [0.032] * 6)


def test_follower_seed_limits_first_command_from_aligned_pose():
    action_filter = EmaSlewFilter(
        ema_alpha=0.8,
        max_joint_step=0.04,
        max_gripper_step=1.0,
    )
    action_filter.seed([0.5] * 6, 0.25)

    output = action_filter.process({"joint": [0.6] * 6, "gripper": 0.5})

    np.testing.assert_allclose(output["joint"], [0.54] * 6)
    assert np.isclose(output["gripper"], 0.45)

import copy
import threading
import time

import numpy as np


class FixedRateControlLoop:
    """Run a control callback independently from blocking data capture."""

    def __init__(self, control_fps, step):
        self.control_fps = float(control_fps)
        if self.control_fps <= 0:
            raise ValueError("control_fps must be positive")
        self.step = step
        self._stop_event = threading.Event()
        self._thread = None
        self._latest_lock = threading.Lock()
        self._latest = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="piperx-teleop", daemon=True)
        self._thread.start()

    def stop(self, timeout=None):
        self._stop_event.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return False
        self._thread = None
        return True

    def _run(self):
        period = 1.0 / self.control_fps
        while not self._stop_event.is_set():
            started_at = time.monotonic()
            result = self.step()
            if result is not None:
                with self._latest_lock:
                    self._latest = copy.deepcopy(result)
            remaining = period - (time.monotonic() - started_at)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def get_latest(self):
        with self._latest_lock:
            return copy.deepcopy(self._latest)


class EmaSlewFilter:
    """Apply the EMA-then-slew pipeline used by robocoin PiperX DAgger teleop."""

    def __init__(
        self,
        ema_enabled=True,
        ema_alpha=0.8,
        slew_enabled=True,
        max_joint_step=0.04,
        max_gripper_step=0.025 / 0.07,
    ):
        self.ema_enabled = bool(ema_enabled)
        self.ema_alpha = float(ema_alpha)
        self.slew_enabled = bool(slew_enabled)
        self.max_joint_step = float(max_joint_step)
        self.max_gripper_step = float(max_gripper_step)
        self._ema_joint = None
        self._ema_gripper = None
        self._previous_joint = None
        self._previous_gripper = None

    def seed(self, joint, gripper):
        joint = np.asarray(joint, dtype=float).copy()
        gripper = float(gripper)
        self._ema_joint = joint.copy()
        self._ema_gripper = gripper
        self._previous_joint = joint.copy()
        self._previous_gripper = gripper

    def process(self, move_data):
        joint = np.asarray(move_data["joint"], dtype=float)
        gripper = float(move_data["gripper"])

        if self.ema_enabled:
            if self._ema_joint is None:
                self._ema_joint = joint.copy()
                self._ema_gripper = gripper
            else:
                alpha = self.ema_alpha
                self._ema_joint = alpha * joint + (1.0 - alpha) * self._ema_joint
                self._ema_gripper = alpha * gripper + (1.0 - alpha) * self._ema_gripper
            joint = self._ema_joint.copy()
            gripper = float(self._ema_gripper)

        if self.slew_enabled:
            if self._previous_joint is None:
                self._previous_joint = joint.copy()
                self._previous_gripper = gripper
            else:
                joint_delta = np.clip(
                    joint - self._previous_joint,
                    -self.max_joint_step,
                    self.max_joint_step,
                )
                gripper_delta = float(
                    np.clip(
                        gripper - self._previous_gripper,
                        -self.max_gripper_step,
                        self.max_gripper_step,
                    )
                )
                self._previous_joint = self._previous_joint + joint_delta
                self._previous_gripper = self._previous_gripper + gripper_delta
            joint = self._previous_joint.copy()
            gripper = float(self._previous_gripper)

        return {"joint": joint.tolist(), "gripper": gripper}

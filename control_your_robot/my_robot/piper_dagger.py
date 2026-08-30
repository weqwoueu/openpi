from pathlib import Path
import math
import subprocess
import sys
import time

sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot
from my_robot.camera_config import get_piper_camera_serials
from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

# Master-slave linkage config (0x470).
MASTER_ROLE = 0xFA  # teaching input arm
FOLLOWER_ROLE = 0xFC  # motion output arm
FEEDBACK_OFFSET = 0x00
CTRL_OFFSET = 0x00
LINKAGE_OFFSET = 0x00

condition = {
    "robot": "piper_dagger",
    "save_path": "./save/",
    "task_name": "dagger",
    "save_format": "hdf5",
    "save_freq": 10,
}


class PiperDAgger(Robot):
    def __init__(
        self,
        condition=condition,
        move_check=True,
        start_episode=0,
        master_can="can_left_mas",
        follower_can="can_left_slave",
        follower_use_mit_mode=True,
    ):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.master_can = master_can
        self.follower_can = follower_can
        self.follower_use_mit_mode = bool(follower_use_mit_mode)
        self.camera_serials = get_piper_camera_serials("dagger")
        self.controllers = {
            "arm": {
                "left_arm": PiperController(
                    "left_arm", use_mit_mode=self.follower_use_mit_mode
                ),  # follower
                "right_arm": PiperController("right_arm"),  # master
            },
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }
        self._policy_enabled = True
        # Use a stronger follower gripper effort during rollout so grasping force
        # stays consistent while the arm is moving.
        self._follower_gripper_effort = 5000
        self._master_gripper_effort = 1000

    @staticmethod
    def _joint_to_cmd(joint):
        joint = np.array(joint, dtype=float)
        return (joint * 57295.7795).astype(int)  # 1000*180/pi

    @staticmethod
    def _gripper_to_cmd(gripper):
        return int(float(gripper) * 70 * 1000)

    @staticmethod
    def _raise_on_can_send_failure(controller, arm_name):
        """Turn piper-sdk's logged send failures into exceptions for data collection."""
        can_bus = controller.GetCanBus()
        if getattr(can_bus, "_pistar_checked_send", False):
            return
        original_send = can_bus.SendCanMessage

        def checked_send(*args, **kwargs):
            status = original_send(*args, **kwargs)
            if status != can_bus.CAN_STATUS.SEND_MESSAGE_SUCCESS:
                raise RuntimeError(f"{arm_name} CAN send failed: {status}")
            return status

        can_bus.SendCanMessage = checked_send
        can_bus._pistar_checked_send = True

    def _send_arm_target(
        self,
        controller,
        move_data,
        *,
        speed_percent: int | None = None,
        gripper_effort: int,
        mit_mode: bool = False,
    ):
        if controller is None or move_data is None:
            return None

        joint = np.asarray(move_data.get("joint"), dtype=float).reshape(-1)
        if joint.shape != (6,):
            raise ValueError(f"joint command must have shape (6,), got {joint.shape}")
        gripper = float(move_data["gripper"])
        action = np.concatenate([joint, np.asarray([gripper], dtype=float)])
        if not np.all(np.isfinite(action)):
            raise ValueError("arm command must contain only finite values")

        if speed_percent is not None:
            controller.MotionCtrl_2(
                0x01,
                0x01,
                int(speed_percent),
                0xAD if mit_mode else 0x00,
            )

        joint_cmd = self._joint_to_cmd(joint)
        controller.JointCtrl(
            int(joint_cmd[0]),
            int(joint_cmd[1]),
            int(joint_cmd[2]),
            int(joint_cmd[3]),
            int(joint_cmd[4]),
            int(joint_cmd[5]),
        )

        gripper_cmd = self._gripper_to_cmd(gripper)
        controller.GripperCtrl(gripper_cmd, int(gripper_effort), 0x01, 0)
        return action.astype(np.float32)

    def reset(self):
        reset_script = Path(__file__).resolve().parents[1] / "scripts/piperx/2_arm_go_init.sh"
        subprocess.run(["bash", str(reset_script)], check=True)

    def set_up(self):
        super().set_up()

        self.controllers["arm"]["right_arm"].set_up(self.master_can)
        self.controllers["arm"]["left_arm"].set_up(self.follower_can)
        self._raise_on_can_send_failure(
            self.controllers["arm"]["right_arm"].controller, "master"
        )
        self._raise_on_can_send_failure(
            self.controllers["arm"]["left_arm"].controller, "follower"
        )

        # 等待 CAN 总线稳定（刚上电时需要更长时间）
        print("[setup] Waiting for CAN bus to stabilize...")
        time.sleep(3.0)

        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller

        print("[setup] Configuring both arms as followers (0xFC)...")
        try:
            master.MasterSlaveConfig(0xFC, 0, 0, 0)
            follower.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            raise RuntimeError("[setup] MasterSlaveConfig failed") from e

        try:
            master.MotionCtrl_2(0x01, 0x01, 15, 0x00)
            follower.MotionCtrl_2(
                0x01,
                0x01,
                15,
                0xAD if self.follower_use_mit_mode else 0x00,
            )
            time.sleep(0.15)
            master.EnableArm(7)
            follower.EnableArm(7)
            time.sleep(0.15)
        except Exception as e:
            raise RuntimeError("[setup] MotionCtrl failed") from e

        print("[setup] Both arms configured as followers")

        self.sensors["image"]["cam_head"].set_up(self.camera_serials["head"])
        self.sensors["image"]["cam_wrist"].set_up(self.camera_serials["wrist"])

        self.set_collect_type({
            "arm": ["joint", "qpos", "gripper"],
            "image": ["color"],
        })

        self.reassert_follower_hold()
        follower_mode = "MIT (0xAD)" if self.follower_use_mit_mode else "position (0x00)"
        print(
            "piper_dagger set up success - both arms in follower role, "
            f"follower control={follower_mode}"
        )

    def move_follower(self, move_data, bypass_policy: bool = False):
        # Allow teleop/intervention to bypass policy gating.
        if not self._policy_enabled and not bypass_policy:
            return None
        follower_controller = self.controllers["arm"]["left_arm"].controller
        sent_action = self._send_arm_target(
            follower_controller,
            move_data,
            speed_percent=100,
            gripper_effort=self._follower_gripper_effort,
            mit_mode=self.follower_use_mit_mode,
        )
        return sent_action

    def set_policy_enabled(self, enabled: bool):
        self._policy_enabled = enabled

    def get_master_state(self):
        return self.controllers["arm"]["right_arm"].get_state()

    def get_master_input_state(self):
        """Read the live master feedback while it is being dragged."""
        return self.get_master_state()

    def get_follower_state(self):
        return self.controllers["arm"]["left_arm"].get_state()

    def hold_follower_position(self):
        """Stop follower movement by commanding its current feedback state."""
        state = self.get_follower_state()
        follower_controller = self.controllers["arm"]["left_arm"].controller

        # Lock the current position and keep the same stronger gripper effort used
        # during follower rollout.
        self._send_arm_target(
            follower_controller,
            {
                "joint": state["joint"],
                "gripper": state["gripper"],
            },
            gripper_effort=self._follower_gripper_effort,
            mit_mode=self.follower_use_mit_mode,
        )

    def reassert_follower_hold(self):
        """Re-apply follower hold after role/mode switches so gripper force stays on."""
        try:
            time.sleep(0.05)
            self.hold_follower_position()
        except Exception as exc:
            print(f"[warn] failed to reassert follower hold: {exc}")

    def align_master_to_follower(self, follower_state, *, speed=60):
        """Robocoin sequence: 0xFC, then ramp the master to follower feedback."""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        target_joint = np.asarray(follower_state["joint"], dtype=float).reshape(-1)
        target_gripper = float(follower_state["gripper"])
        if target_joint.shape != (6,) or not np.all(
            np.isfinite(np.concatenate([target_joint, [target_gripper]]))
        ):
            raise RuntimeError("takeover alignment requires finite 6D follower feedback")

        master.MasterSlaveConfig(
            FOLLOWER_ROLE,
            FEEDBACK_OFFSET,
            CTRL_OFFSET,
            LINKAGE_OFFSET,
        )
        time.sleep(0.5)
        master.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        time.sleep(0.15)

        target_sdk = np.concatenate(
            [self._joint_to_cmd(target_joint), [self._gripper_to_cmd(target_gripper)]]
        ).astype(int)
        for attempt in range(3):
            start_state = self.get_master_state()
            start_joint = np.asarray(start_state["joint"], dtype=float).reshape(-1)
            start_gripper = float(start_state["gripper"])
            if start_joint.shape != (6,) or not np.all(
                np.isfinite(np.concatenate([start_joint, [start_gripper]]))
            ):
                raise RuntimeError("takeover alignment received invalid master feedback")

            start_sdk = np.concatenate(
                [self._joint_to_cmd(start_joint), [self._gripper_to_cmd(start_gripper)]]
            ).astype(int)
            max_delta = int(np.max(np.abs(target_sdk - start_sdk)))
            steps = 1 if max_delta <= 1500 else max(6, min(48, math.ceil(max_delta / 2500)))
            print(
                "[align] moving master from init to follower feedback: "
                f"attempt={attempt + 1}/3, steps={steps}"
            )
            for step in range(1, steps + 1):
                ratio = step / steps
                command_sdk = np.rint(
                    start_sdk + (target_sdk - start_sdk) * ratio
                ).astype(int)
                self._send_arm_target(
                    master,
                    {
                        "joint": command_sdk[:6] / 57295.7795,
                        "gripper": command_sdk[6] / (70 * 1000),
                    },
                    speed_percent=int(speed) if step == 1 else None,
                    gripper_effort=self._master_gripper_effort,
                )
                time.sleep(0.045)

            time.sleep(0.5)
            actual = self.get_master_state()
            actual_sdk = np.concatenate(
                [
                    self._joint_to_cmd(actual["joint"]),
                    [self._gripper_to_cmd(actual["gripper"])],
                ]
            ).astype(int)
            joint_error = int(np.max(np.abs(actual_sdk[:6] - target_sdk[:6])))
            gripper_error = abs(int(actual_sdk[6] - target_sdk[6]))
            if joint_error <= 1500 and gripper_error <= 10000:
                print(
                    "[align] master aligned to follower: "
                    f"joint_error_sdk={joint_error}, gripper_error_sdk={gripper_error}"
                )
                return

        raise TimeoutError("master did not reach follower pose after 3 ramp attempts")

    def enable_master_drag_mode(self):
        """Switch the aligned master to the native 0xFA draggable role."""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        master.MasterSlaveConfig(
            MASTER_ROLE,
            FEEDBACK_OFFSET,
            CTRL_OFFSET,
            LINKAGE_OFFSET,
        )
        time.sleep(0.5)

        print("[mode] master arm is draggable; expert controls follower")

    def disable_master_drag_mode(self):
        """Return the master to 0xFC position control for the next reset."""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        self.reassert_follower_hold()
        master_state = self.get_master_input_state()
        master.MasterSlaveConfig(
            FOLLOWER_ROLE,
            FEEDBACK_OFFSET,
            CTRL_OFFSET,
            LINKAGE_OFFSET,
        )
        time.sleep(0.5)
        master.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        time.sleep(0.15)
        self._send_arm_target(
            master,
            master_state,
            speed_percent=15,
            gripper_effort=self._master_gripper_effort,
        )
        master.EnableArm(7)
        self.reassert_follower_hold()
        print("[mode] master arm returned to 0xFC position control")

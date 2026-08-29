from pathlib import Path
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
        self._mirror_last_joint_cmd = None
        self._mirror_last_gripper_cmd = None
        # Deadband thresholds to reduce jitter during intervention
        # Joint: 10000 ≈ 0.175 rad ≈ 10 degrees
        # Gripper: 2000 ≈ 0.029 (2.9% of range)
        self._mirror_joint_deadband_cmd = 10000
        self._mirror_gripper_deadband_cmd = 2000
        # Rate limiting & optional low-pass filter to reduce jitter
        self._mirror_min_send_interval = 0.02  # seconds
        self._mirror_filter_alpha = None  # 0~1, None disables smoothing
        self._mirror_last_send_time = 0.0
        self._intervention_anchor_joint_cmd = None
        self._intervention_anchor_gripper_cmd = None
        self._intervention_active = False
        self._intervention_start_deadband_cmd = 15000
        self._intervention_start_gripper_deadband_cmd = 3000
        self._intervention_move_count = 0
        self._intervention_move_required = 3
        self._policy_enabled = True
        self._mirror_master_speed_percent = 100
        self._sync_master_with_policy_commands = True
        self._mirror_master_joint_baseline = None
        self._mirror_master_gripper_baseline = None
        self._mirror_follower_joint_baseline = None
        self._mirror_follower_gripper_baseline = None
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

        import time

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
        time.sleep(3.0)  # 增加到 3 秒

        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller

        print("[setup] Configuring both arms as followers (0xFC)...")

        # Exit drag teaching mode first
        try:
            # Master arm: forcefully exit all special modes
            master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
            time.sleep(0.2)
            master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
            time.sleep(0.2)

            # Follower arm: clear all modes
            follower.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching (just in case)
            time.sleep(0.2)
            follower.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
            time.sleep(0.3)
        except Exception as e:
            raise RuntimeError("[setup] failed to exit previous drag mode") from e

        # CRITICAL: Force reset from MASTER role to FOLLOWER role
        # If master arm was in 0xFA (MASTER) mode, need explicit transition
        try:
            print("[setup] Force resetting master arm from any previous role...")
            # First set to FOLLOWER (this may fail if already in a weird state)
            master.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)
            # Set again to ensure it takes effect
            master.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)

            print("[setup] Configuring follower arm...")
            follower.MasterSlaveConfig(0xFC, 0, 0, 0)
            time.sleep(0.3)
        except Exception as e:
            raise RuntimeError("[setup] MasterSlaveConfig failed") from e

        # Enable joint control mode with lower speed to prevent jitter
        try:
            master.MotionCtrl_1(0x00, 0x00, 0x00)
            time.sleep(0.1)
            follower.MotionCtrl_1(0x00, 0x00, 0x00)
            time.sleep(0.1)
            # Use slow speed (15%) during setup to prevent sudden movements
            master.MotionCtrl_2(0x01, 0x01, 15, 0x00)
            time.sleep(0.1)
            follower.MotionCtrl_2(
                0x01,
                0x01,
                15,
                0xAD if self.follower_use_mit_mode else 0x00,
            )
            time.sleep(0.2)

            # Enable arms to stabilize
            master.EnableArm(7)
            time.sleep(0.1)
            follower.EnableArm(7)
            time.sleep(0.2)
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

    def _configure_both_as_followers(self):
        """Configure both arms as followers (software controllable)"""
        master = self.controllers["arm"]["right_arm"].controller
        follower = self.controllers["arm"]["left_arm"].controller
        if master is None or follower is None:
            raise RuntimeError("Controllers are not initialized")

        # Step 1: Exit drag teaching mode first (if in that mode)
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.1)
        master.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        follower.MotionCtrl_1(0x00, 0x00, 0x00)  # Clear all modes
        time.sleep(0.1)

        # Step 2: Configure both as follower role
        master.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        follower.MasterSlaveConfig(FOLLOWER_ROLE, FEEDBACK_OFFSET, CTRL_OFFSET, LINKAGE_OFFSET)
        time.sleep(0.2)

        # Step 3: Enable joint control mode for both arms
        master.MotionCtrl_2(0x01, 0x01, 100, 0x00)  # Enable joint control, max speed
        follower.MotionCtrl_2(
            0x01,
            0x01,
            100,
            0xAD if self.follower_use_mit_mode else 0x00,
        )
        time.sleep(0.1)
        master.EnableArm(7)
        follower.EnableArm(7)

        self.reassert_follower_hold()
        print("[reset] Both arms configured as followers")

    def move_follower(self, move_data, bypass_policy: bool = False):
        # Allow teleop/intervention to bypass policy gating.
        if not self._policy_enabled and not bypass_policy:
            return None
        if self._sync_master_with_policy_commands and self._policy_enabled and not bypass_policy:
            # During autonomous rollout, send the same target to the master arm immediately
            # instead of waiting for feedback-based mirroring, which lags behind the follower.
            master_controller = self.controllers["arm"]["right_arm"].controller
            self._send_arm_target(
                master_controller,
                move_data,
                speed_percent=100,
                gripper_effort=self._master_gripper_effort,
            )
        follower_controller = self.controllers["arm"]["left_arm"].controller
        sent_action = self._send_arm_target(
            follower_controller,
            move_data,
            speed_percent=100,
            gripper_effort=self._follower_gripper_effort,
            mit_mode=self.follower_use_mit_mode,
        )
        return sent_action

    def move_master(self, move_data):
        master_controller = self.controllers["arm"]["right_arm"].controller
        self._send_arm_target(
            master_controller,
            move_data,
            speed_percent=100,
            gripper_effort=self._master_gripper_effort,
        )

    def set_policy_enabled(self, enabled: bool):
        self._policy_enabled = enabled

    def get_master_state(self):
        return self.controllers["arm"]["right_arm"].get_state()

    def get_master_input_state(self):
        """Read the live master pose, preferring its teaching control frames."""
        master = self.controllers["arm"]["right_arm"].controller
        try:
            ctrl = master.GetArmJointCtrl()
            if getattr(ctrl, "Hz", 0) > 0:
                joints = ctrl.joint_ctrl
                gripper = master.GetArmGripperCtrl().gripper_ctrl.grippers_angle
                return {
                    "joint": np.array(
                        [
                            joints.joint_1,
                            joints.joint_2,
                            joints.joint_3,
                            joints.joint_4,
                            joints.joint_5,
                            joints.joint_6,
                        ],
                        dtype=float,
                    )
                    / 57295.7795,
                    "gripper": float(gripper) / (70 * 1000),
                }
        except Exception:
            pass
        return self.get_master_state()

    def get_follower_state(self):
        return self.controllers["arm"]["left_arm"].get_state()

    def reset_intervention_tracking(self):
        """Clear intervention tracking state when switching modes."""
        self._intervention_anchor_joint_cmd = None
        self._intervention_anchor_gripper_cmd = None
        self._intervention_active = False
        self._intervention_move_count = 0
        # CRITICAL: Also reset mirror tracking to prevent stale data
        self._mirror_last_joint_cmd = None
        self._mirror_last_gripper_cmd = None
        self._mirror_last_send_time = 0.0
        self._mirror_master_joint_baseline = None
        self._mirror_master_gripper_baseline = None
        self._mirror_follower_joint_baseline = None
        self._mirror_follower_gripper_baseline = None

    def set_mirror_sync_baseline(self):
        """Capture the current master/follower poses as the relative-sync baseline."""
        master_state = self.get_master_state()
        follower_state = self.get_follower_state()
        self._mirror_master_joint_baseline = np.array(master_state["joint"], dtype=float)
        self._mirror_master_gripper_baseline = float(master_state["gripper"])
        self._mirror_follower_joint_baseline = np.array(follower_state["joint"], dtype=float)
        self._mirror_follower_gripper_baseline = float(follower_state["gripper"])

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


    def mirror_master_to_follower(self):
        """Mirror master arm position to follower arm during intervention"""
        master_controller = self.controllers["arm"]["right_arm"].controller
        follower_controller = self.controllers["arm"]["left_arm"].controller

        # CRITICAL: Use control frames (GetArmJointCtrl) in drag mode, not feedback frames
        ctrl = master_controller.GetArmJointCtrl()
        if ctrl.Hz > 0:
            joint_ctrl = ctrl.joint_ctrl
            joint_cmd = np.array([
                joint_ctrl.joint_1,
                joint_ctrl.joint_2,
                joint_ctrl.joint_3,
                joint_ctrl.joint_4,
                joint_ctrl.joint_5,
                joint_ctrl.joint_6,
            ], dtype=int)
            # FIXED: Use correct gripper control field
            gripper_ctrl = master_controller.GetArmGripperCtrl()
            gripper_cmd = gripper_ctrl.gripper_ctrl.grippers_angle
        else:
            state = self.get_master_state()
            joint_cmd = (state["joint"] * 57295.7795).astype(int)  # 1000*180/3.1415926
            gripper_cmd = int(state["gripper"] * 70 * 1000)

        if not self._intervention_active:
            if self._intervention_anchor_joint_cmd is None:
                self._intervention_anchor_joint_cmd = joint_cmd.copy()
                self._intervention_anchor_gripper_cmd = gripper_cmd
                return

            anchor_delta = np.max(np.abs(joint_cmd - self._intervention_anchor_joint_cmd))
            anchor_gripper_delta = abs(gripper_cmd - self._intervention_anchor_gripper_cmd)
            if (anchor_delta < self._intervention_start_deadband_cmd and
                    anchor_gripper_delta < self._intervention_start_gripper_deadband_cmd):
                self._intervention_move_count = 0
                self.hold_follower_position()
                return

            self._intervention_move_count += 1
            if self._intervention_move_count < self._intervention_move_required:
                return

            self._intervention_active = True

        if self._mirror_last_joint_cmd is not None:
            joint_delta = np.max(np.abs(joint_cmd - self._mirror_last_joint_cmd))
            gripper_delta = abs(gripper_cmd - self._mirror_last_gripper_cmd)
            if (joint_delta < self._mirror_joint_deadband_cmd and
                    gripper_delta < self._mirror_gripper_deadband_cmd):
                return

        now = time.time()
        if now - self._mirror_last_send_time < self._mirror_min_send_interval:
            return

        if self._mirror_filter_alpha is not None and self._mirror_last_joint_cmd is not None:
            alpha = float(self._mirror_filter_alpha)
            joint_cmd = (alpha * joint_cmd + (1.0 - alpha) * self._mirror_last_joint_cmd).astype(int)
            gripper_cmd = int(alpha * gripper_cmd + (1.0 - alpha) * self._mirror_last_gripper_cmd)

        self._mirror_last_joint_cmd = joint_cmd.copy()
        self._mirror_last_gripper_cmd = gripper_cmd
        self._mirror_last_send_time = now

        self._send_arm_target(
            follower_controller,
            {
                "joint": joint_cmd / 57295.7795,
                "gripper": gripper_cmd / (70 * 1000),
            },
            gripper_effort=self._follower_gripper_effort,
            mit_mode=self.follower_use_mit_mode,
        )

    def mirror_follower_to_master(self):
        """Mirror follower arm position to master arm for observation.

        The master arm keeps its current pose as the synchronization baseline, then
        tracks follower deltas relative to that baseline instead of jumping to the
        follower's absolute reset pose.
        """
        if self._sync_master_with_policy_commands and self._policy_enabled:
            return

        follower_state = self.get_follower_state()
        master_state = self.get_master_state()
        master_controller = self.controllers["arm"]["right_arm"].controller

        if self._mirror_master_joint_baseline is None:
            self.set_mirror_sync_baseline()
            return

        joint_delta = np.array(follower_state["joint"], dtype=float) - self._mirror_follower_joint_baseline
        target_joint = self._mirror_master_joint_baseline + joint_delta
        target_gripper = (
            self._mirror_master_gripper_baseline
            + float(follower_state["gripper"])
            - self._mirror_follower_gripper_baseline
        )

        # Convert joint angles to controller format
        j1, j2, j3, j4, j5, j6 = target_joint * 57295.7795  # 1000*180/3.1415926
        j1, j2, j3, j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        master_controller.MotionCtrl_2(0x01, 0x01, self._mirror_master_speed_percent, 0x00)
        master_controller.JointCtrl(j1, j2, j3, j4, j5, j6)

        # Mirror gripper
        gripper = int(target_gripper * 70 * 1000)
        master_controller.GripperCtrl(gripper, self._master_gripper_effort, 0x01, 0)

    def enable_master_drag_mode(self):
        """Enable intervention mode: master arm becomes draggable teaching arm"""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        status = master.GetArmStatus().arm_status
        if int(status.ctrl_mode) == 0x01 and int(status.teach_status) == 0x01:
            # Cancel a previously accepted but incomplete teach-mode request.
            master.MotionCtrl_1(0x00, 0x00, 0x02)
            time.sleep(0.1)

        # Stop the master's last autonomous MOVE_J at its live feedback pose.
        # A still-pending policy target can keep ctrl_mode in CAN control even
        # after the drag-teach request has been accepted.
        master_state = self.get_master_state()
        self._send_arm_target(
            master,
            master_state,
            speed_percent=15,
            gripper_effort=self._master_gripper_effort,
        )
        time.sleep(0.1)

        settle_deadline = time.monotonic() + 2.0
        mode = teach = arm_state = motion = mode_feed = None
        while time.monotonic() < settle_deadline:
            status = master.GetArmStatus().arm_status
            mode = int(status.ctrl_mode)
            teach = int(status.teach_status)
            arm_state = int(status.arm_status)
            motion = int(status.motion_status)
            mode_feed = int(status.mode_feed)
            if (
                mode == 0x01
                and teach in (0x00, 0x02)
                and arm_state == 0x00
                and motion == 0x00
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "master did not settle before drag teaching: "
                f"ctrl_mode=0x{mode:02X}, mode_feed=0x{mode_feed:02X}, "
                f"teach_status=0x{teach:02X}, motion_status=0x{motion:02X}, "
                f"arm_status=0x{arm_state:02X}"
            )

        # Snapshot after the hold command so its TX echo cannot be mistaken for
        # a native master-input frame after the role switch.
        ctrl_before_switch = master.GetArmJointCtrl()
        ctrl_timestamp_before_switch = float(
            getattr(ctrl_before_switch, "time_stamp", 0.0)
        )

        # This PiperX teaching arm uses its native input-arm role for manual
        # motion. Switch only after the last autonomous MOVE_J is confirmed still.
        master.MasterSlaveConfig(
            MASTER_ROLE,
            FEEDBACK_OFFSET,
            CTRL_OFFSET,
            LINKAGE_OFFSET,
        )
        time.sleep(0.5)

        try:
            if hasattr(master, "GripperTeachingPendantParamConfig"):
                master.GripperTeachingPendantParamConfig(
                    teaching_range_per=100,
                    max_range_config=70,
                    teaching_friction=1,  # Very light friction
                )
                time.sleep(0.1)
        except Exception as e:
            print(f"[warn] failed to set teaching friction: {e}")

        master.MotionCtrl_1(0x00, 0x00, 0x01)

        self.reset_intervention_tracking()
        print("[mode] master input role requested; waiting for a new control frame")
        return ctrl_timestamp_before_switch

    def disable_master_drag_mode(self):
        """Disable intervention mode: master arm becomes software-controllable follower"""
        master = self.controllers["arm"]["right_arm"].controller
        if master is None:
            raise RuntimeError("Master controller is not initialized")

        print("[mode] disabling master drag mode...")
        # The follower never leaves follower/joint-control mode during intervention.
        # Reconfiguring it here causes a brief release of the gripper hold, so keep
        # the follower untouched and simply re-assert its current target.
        self.reassert_follower_hold()

        master_state = self.get_master_input_state()

        # Exit drag teaching, then use the live pose as the first CAN target so
        # the master does not return to a stale pre-takeover command.
        master.MotionCtrl_1(0x00, 0x00, 0x02)  # Exit drag teaching
        time.sleep(0.15)
        master.MasterSlaveConfig(
            FOLLOWER_ROLE,
            FEEDBACK_OFFSET,
            CTRL_OFFSET,
            LINKAGE_OFFSET,
        )
        time.sleep(0.25)

        self.reset_intervention_tracking()
        self._send_arm_target(
            master,
            master_state,
            speed_percent=15,
            gripper_effort=self._master_gripper_effort,
        )
        master.EnableArm(7)
        time.sleep(0.15)

        self.reassert_follower_hold()
        print("[mode] master arm back to CAN joint control - software control ready")

    def reset_to_follower_mode(self):
        """Reset both arms to follower mode (called at episode end)"""
        self._configure_both_as_followers()

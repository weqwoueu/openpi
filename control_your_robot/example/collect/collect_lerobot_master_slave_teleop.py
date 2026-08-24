"""
软件主从遥操作 + LeRobot 直接采集脚本
基于 example/collect/collect_lerobot_direct.py

说明:
- 适用于两根 CAN 线分别连接（无硬件主从链路）场景
- 脚本将读取主臂关节状态，并通过 CAN 控制从臂

使用方法:
1) 确认主臂处于可拖动（拖动示教）模式
2) 运行本脚本开始采集:
   python example/collect/collect_lerobot_master_slave_teleop.py
"""
import sys
sys.path.append("./")
import time
import tty
import termios
from multiprocessing import Process, Event, Barrier

import numpy as np

from my_robot.piper_single_lerobot import PiperSingleLeRobot

from robot.controller.Piper_controller import PiperController
from robot.data.collect_lerobot_rl import CollectLeRobotRL
from robot.utils.worker.time_scheduler import TimeScheduler
from robot.utils.base.data_handler import debug_print
from robot.utils.teleop_filter import EmaSlewFilter, FixedRateControlLoop


def _read_key_nonblocking(timeout: float = 0.1):
    """Read a single keypress if available, otherwise return None."""
    import select
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


class _StdinCbreak:
    """Temporarily switch stdin to cbreak mode so single keypresses are captured."""
    def __init__(self):
        self._old_settings = None
        self._enabled = sys.stdin.isatty()

    def __enter__(self):
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


def RobotWorkerLeRobot(
    robot_class,
    robot_kwargs: dict,
    teleop_kwargs: dict,
    time_lock: Barrier,
    start_event: Event,
    finish_event: Event,
    exit_event: Event,
    saved_event: Event,
    discard_event: Event,
    process_name: str,
):
    """
    LeRobot 版本的机器人 Worker (支持多 Episode 复用进程)

    Args:
        robot_class: 机器人类（如 PiperSingleLeRobot）
        robot_kwargs: 机器人初始化参数
        time_lock: 时间同步锁
        start_event: 开始事件
        finish_event: 结束事件
        exit_event: 退出进程事件
        saved_event: 保存完成事件
        discard_event: 放弃当前 episode 事件
        process_name: 进程名称
    """
    # 创建机器人实例
    robot = robot_class(**robot_kwargs)
    base_collection = robot.collection
    robot.collection = CollectLeRobotRL(
        repo_id=base_collection.repo_id,
        output_dir=str(base_collection.output_dir),
        task_name=base_collection.task_name,
        fps=base_collection.fps,
        robot_type=base_collection.robot_type,
        state_dim=base_collection.state_dim,
        action_dim=base_collection.action_dim,
        image_size=base_collection.image_size,
        camera_keys=base_collection.camera_keys,
        move_check=base_collection.move_check,
        tolerance=base_collection.tolerance,
    )
    robot.set_up()
    try:
        robot.reset()
        debug_print(process_name, f"从臂已复位到初始化位置: {robot.reset_joint_position.round(6).tolist()}", "INFO")
        time.sleep(1.0)
    except Exception as e:
        debug_print(process_name, f"从臂初始化位置设置失败: {e}", "WARNING")

    # ===== 软件主从遥操作: 初始化主臂控制器 =====
    teleop_enabled = teleop_kwargs.get("enabled", True)
    master = None
    control_loop = None
    control_fps = float(teleop_kwargs.get("control_fps", 30))
    action_filter = EmaSlewFilter(
        ema_enabled=teleop_kwargs.get("teacher_action_ema_enabled", True),
        ema_alpha=teleop_kwargs.get("teacher_action_ema_alpha", 0.8),
        slew_enabled=teleop_kwargs.get("teacher_action_slew_enabled", True),
        max_joint_step=teleop_kwargs.get("teacher_action_max_joint_step", 0.04),
        max_gripper_step=teleop_kwargs.get("teacher_action_max_gripper_step", 0.025 / 0.07),
    )

    if teleop_enabled:
        master_can = teleop_kwargs.get("master_can", "can_left_mas")
        master = PiperController("master_arm")
        master.set_up(master_can)
        master.set_collect_info(["joint", "gripper"])
        teleop_kwargs["alignment_ready"] = False

        # 先重置主臂状态（防止上次运行残留的配置）
        debug_print(process_name, "重置主臂状态...", "INFO")
        try:
            # 退出任何特殊模式
            master.controller.MotionCtrl_1(0x00, 0x00, 0x02)  # 退出拖动示教
            time.sleep(0.1)
            master.controller.MotionCtrl_1(0x00, 0x00, 0x00)  # 清除所有模式
            time.sleep(0.1)

            # 先配置为 FOLLOWER，再切换到 MASTER（确保干净的状态转换）
            FOLLOWER_ROLE = 0xFC
            master.controller.MasterSlaveConfig(FOLLOWER_ROLE, 0x00, 0x00, 0x00)
            time.sleep(0.2)
        except Exception as e:
            debug_print(process_name, f"主臂重置警告: {e}", "WARNING")

        # 进入拖动示教模式（主臂可自由拖动）
        if teleop_kwargs.get("enable_drag_teach", True):
            try:
                # 关键步骤1: 配置为 MASTER 角色 (0xFA)，这样才能真正无引导力
                MASTER_ROLE = 0xFA
                master.controller.MasterSlaveConfig(MASTER_ROLE, 0x00, 0x00, 0x00)
                time.sleep(0.2)

                # 关键步骤2: 降低拖动示教摩擦（数值越小越轻）
                teaching_friction = teleop_kwargs.get("teaching_friction", None)
                if teaching_friction is not None and hasattr(master.controller, "GripperTeachingPendantParamConfig"):
                    master.controller.GripperTeachingPendantParamConfig(
                        teaching_range_per=100,
                        max_range_config=70,
                        teaching_friction=int(teaching_friction),
                    )

                # 关键步骤3: 启用拖动示教模式
                master.controller.MotionCtrl_1(0x00, 0x00, 0x01)
                time.sleep(0.2)
                debug_print(process_name, "主臂拖动示教模式已启用 - 可自由拖动", "INFO")
            except Exception as e:
                debug_print(process_name, f"主臂拖动示教启用失败: {e}", "WARNING")

        # 使用从臂配置的 MIT 或位置/速度关节控制模式
        try:
            slave_controller = robot.controllers["arm"]["left_arm"]
            slave_ctrl = slave_controller.controller
            slave_ctrl.MotionCtrl_1(0x00, 0x00, 0x00)
            slave_ctrl.MotionCtrl_2(0x01, 0x01, 100, slave_controller._mit_mode_flag())
            try:
                slave_ctrl.EnableArm(7)
            except Exception:
                pass
        except Exception as e:
            debug_print(process_name, f"从臂控制模式设置失败: {e}", "WARNING")

        aligned_state = _align_master_to_follower_pose(robot, master, teleop_kwargs, process_name)
        if aligned_state is not None:
            teleop_kwargs["alignment_ready"] = True
            action_filter.seed(aligned_state["joint"], aligned_state["gripper"])

        def teleop_control_step():
            try:
                if not teleop_kwargs.get("alignment_ready", True):
                    retry_state = _align_master_to_follower_pose(
                        robot, master, teleop_kwargs, process_name, timeout=0.1, log_failure=False
                    )
                    if retry_state is None:
                        return
                    teleop_kwargs["alignment_ready"] = True
                    action_filter.seed(retry_state["joint"], retry_state["gripper"])

                move_data = _read_master_action(master, teleop_kwargs)
                if move_data is not None:
                    robot.move({"arm": {"left_arm": action_filter.process(move_data)}})
            except Exception as e:
                debug_print(process_name, f"镜像控制错误: {e}", "DEBUG")

        control_loop = FixedRateControlLoop(control_fps=control_fps, step=teleop_control_step)
        control_loop.start()

    debug_print(process_name, "初始化完成，等待主进程指令...", "INFO")

    try:
        while not exit_event.is_set():
            # 1. 等待开始信号（同时让从臂跟随主臂）
            if not start_event.is_set():
                time.sleep(0.01)
                continue

            # 收到开始信号，进入采集循环
            debug_print(process_name, "收到开始信号，开始采集...", "INFO")

            while not finish_event.is_set():
                if exit_event.is_set():
                    break

                try:
                    time_lock.wait()
                except Exception as e:
                    debug_print(process_name, f"警告: Barrier 被中止 ({e})", "WARNING")
                    break

                if finish_event.is_set():
                    break

                try:
                    # 相机/数据采集保持 10 Hz，不阻塞独立的 30 Hz PiperX 控制线程。
                    data = robot.get()
                    robot.collection.collect(data[0], data[1], is_intervention=True)
                except Exception as e:
                    debug_print(process_name, f"错误: {e}", "ERROR")

            # 2. 收到结束信号，检查是否需要放弃
            if discard_event.is_set():
                debug_print(process_name, "⚠ 放弃当前 episode，不保存数据", "WARNING")
                # 清空缓存但不保存
                robot.collection.clear_current_episode()
            else:
                debug_print(process_name, "收到结束信号，正在保存...", "INFO")
                robot.collection.save_episode(success=True, adv_ind_value="positive")
                debug_print(process_name, "✓ 保存成功！", "INFO")

            # 通知主进程处理完毕
            saved_event.set()

            # 等待主进程重置信号
            while start_event.is_set() and not exit_event.is_set():
                time.sleep(0.01)

    except KeyboardInterrupt:
        debug_print(process_name, "用户中断", "WARNING")
    except Exception as e:
        debug_print(process_name, f"发生未捕获异常: {e}", "ERROR")
        import traceback
        debug_print(process_name, traceback.format_exc(), "ERROR")
    finally:
        if control_loop is not None:
            control_loop.stop()

        # 清理：将主臂恢复到 FOLLOWER 模式
        if teleop_enabled and master is not None:
            try:
                debug_print(process_name, "恢复主臂到 FOLLOWER 模式...", "INFO")
                # 退出拖动示教模式
                master.controller.MotionCtrl_1(0x00, 0x00, 0x02)
                time.sleep(0.1)
                master.controller.MotionCtrl_1(0x00, 0x00, 0x00)
                time.sleep(0.1)

                # 恢复为 FOLLOWER 角色
                FOLLOWER_ROLE = 0xFC
                master.controller.MasterSlaveConfig(FOLLOWER_ROLE, 0x00, 0x00, 0x00)
                time.sleep(0.2)

                # 启用正常控制模式
                master.controller.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                try:
                    master.controller.EnableArm(7)
                except Exception:
                    pass
                debug_print(process_name, "✓ 主臂已恢复到 FOLLOWER 模式", "INFO")
            except Exception as e:
                debug_print(process_name, f"主臂恢复失败: {e}", "WARNING")

        debug_print(process_name, "Worker 退出", "INFO")


def _read_master_ctrl_action(master: PiperController):
    try:
        ctrl = master.controller.GetArmJointCtrl()
        if getattr(ctrl, "Hz", 0) <= 0:
            return None
        joint_ctrl = ctrl.joint_ctrl
        joint_cmd = np.array([
            joint_ctrl.joint_1,
            joint_ctrl.joint_2,
            joint_ctrl.joint_3,
            joint_ctrl.joint_4,
            joint_ctrl.joint_5,
            joint_ctrl.joint_6,
        ], dtype=float)
        joint = joint_cmd / 57295.7795

        gripper_ctrl = master.controller.GetArmGripperCtrl()
        gripper_hz = getattr(gripper_ctrl, "Hz", 0)
        gripper_cmd = getattr(gripper_ctrl.gripper_ctrl, "grippers_angle", 0) if gripper_hz > 0 else 0
        gripper = float(gripper_cmd) / (70 * 1000)
        return joint, gripper, "ctrl"
    except Exception:
        return None


def _read_master_feedback_action(master: PiperController):
    try:
        joint_msg = master.controller.GetArmJointMsgs()
        if getattr(joint_msg, "Hz", 0) <= 0:
            return None
        joint = np.array([
            joint_msg.joint_state.joint_1,
            joint_msg.joint_state.joint_2,
            joint_msg.joint_state.joint_3,
            joint_msg.joint_state.joint_4,
            joint_msg.joint_state.joint_5,
            joint_msg.joint_state.joint_6,
        ], dtype=float) * 0.001 / 180 * np.pi

        gripper_msg = master.controller.GetArmGripperMsgs()
        gripper = 0.0
        if getattr(gripper_msg, "Hz", 0) > 0:
            gripper = float(gripper_msg.gripper_state.grippers_angle) * 0.001 / 70
        return joint, gripper, "feedback"
    except Exception:
        return None


def _read_master_raw_action(master: PiperController, teleop_kwargs, allow_feedback_fallback=None):
    """
    读取主臂动作。
    - 优先使用对齐时锁定的数据源
    - 若未锁定，按配置优先读取控制帧
    - 必要时回退到反馈状态
    """
    preferred_source = teleop_kwargs.get("master_action_source")
    use_ctrl_frame = teleop_kwargs.get("use_ctrl_frame", True)
    if allow_feedback_fallback is None:
        allow_feedback_fallback = teleop_kwargs.get("fallback_to_feedback", False)

    if preferred_source == "feedback":
        return _read_master_feedback_action(master)
    if preferred_source == "ctrl":
        ctrl_action = _read_master_ctrl_action(master)
        if ctrl_action is not None:
            return ctrl_action
        if allow_feedback_fallback:
            return _read_master_feedback_action(master)
        return None

    if use_ctrl_frame:
        ctrl_action = _read_master_ctrl_action(master)
        if ctrl_action is not None:
            return ctrl_action
        if not allow_feedback_fallback:
            return None
        return _read_master_feedback_action(master)

    feedback_action = _read_master_feedback_action(master)
    if feedback_action is not None:
        return feedback_action
    return _read_master_ctrl_action(master)


def _read_master_action(master: PiperController, teleop_kwargs):
    raw_action = _read_master_raw_action(master, teleop_kwargs)
    if raw_action is None:
        return None
    joint, gripper, _ = raw_action
    return _teleop_transform(joint, gripper, teleop_kwargs)


def _align_master_to_follower_pose(
    robot,
    master: PiperController,
    teleop_kwargs,
    process_name: str,
    timeout: float | None = None,
    log_failure: bool = True,
):
    """将当前主臂姿态对齐到从臂初始化姿态，避免初始化后立即跳回绝对映射位置。"""
    if not teleop_kwargs.get("align_to_follower_on_init", True):
        return None

    timeout = float(timeout if timeout is not None else teleop_kwargs.get("alignment_timeout", 2.0))
    deadline = time.time() + timeout
    raw_action = None
    while time.time() < deadline:
        raw_action = _read_master_raw_action(master, teleop_kwargs, allow_feedback_fallback=True)
        if raw_action is not None:
            break
        time.sleep(0.02)

    if raw_action is None:
        if log_failure:
            debug_print(process_name, "初始化对齐失败: 未读取到主臂有效状态，暂不发送从臂跟随命令", "WARNING")
        return None

    follower_state = robot.controllers["arm"]["left_arm"].get_state()
    follower_joint = np.array(follower_state["joint"], dtype=float)
    follower_gripper = float(follower_state["gripper"])
    master_joint, master_gripper, source = raw_action
    master_joint = np.array(master_joint, dtype=float)
    master_gripper = float(master_gripper)

    joint_sign = np.array(teleop_kwargs.get("joint_sign", [1, 1, 1, 1, 1, 1]), dtype=float)
    joint_scale = float(teleop_kwargs.get("joint_scale", 1.0))
    base_joint_offset = np.array(teleop_kwargs.get("joint_offset", [0, 0, 0, 0, 0, 0]), dtype=float)
    gripper_scale = float(teleop_kwargs.get("gripper_scale", 1.0))
    base_gripper_offset = float(teleop_kwargs.get("gripper_offset", 0.0))

    runtime_joint_offset = follower_joint - (master_joint * joint_sign * joint_scale + base_joint_offset)
    runtime_gripper_offset = follower_gripper - (master_gripper * gripper_scale + base_gripper_offset)

    teleop_kwargs["runtime_joint_offset"] = runtime_joint_offset.tolist()
    teleop_kwargs["runtime_gripper_offset"] = runtime_gripper_offset
    teleop_kwargs["master_action_source"] = source

    debug_print(
        process_name,
        f"已建立主从对齐偏置，数据源={source}，当前主臂姿态将映射到从臂初始化位置: {follower_joint.round(6).tolist()}",
        "INFO",
    )
    return {
        "joint": follower_joint.tolist(),
        "gripper": follower_gripper,
    }


def _teleop_transform(joint_in, gripper_in, teleop_kwargs):
    # 基础映射（可在配置中做缩放/反向/偏置）
    joint = np.array(joint_in, dtype=float)
    gripper = float(gripper_in)

    joint_sign = np.array(teleop_kwargs.get("joint_sign", [1, 1, 1, 1, 1, 1]), dtype=float)
    joint_offset = (
        np.array(teleop_kwargs.get("joint_offset", [0, 0, 0, 0, 0, 0]), dtype=float)
        + np.array(teleop_kwargs.get("runtime_joint_offset", [0, 0, 0, 0, 0, 0]), dtype=float)
    )
    joint_scale = float(teleop_kwargs.get("joint_scale", 1.0))
    gripper_scale = float(teleop_kwargs.get("gripper_scale", 1.0))
    gripper_offset = float(teleop_kwargs.get("gripper_offset", 0.0)) + float(
        teleop_kwargs.get("runtime_gripper_offset", 0.0)
    )

    joint = joint * joint_sign * joint_scale + joint_offset
    gripper = max(0.0, min(1.0, gripper * gripper_scale + gripper_offset))

    return {
        "joint": joint.tolist(),
        "gripper": gripper,
    }


if __name__ == "__main__":
    import os
    os.environ["INFO_LEVEL"] = "INFO"  # DEBUG, INFO, ERROR

    # ==================== 配置参数 ====================
    REPO_ID = "piperx/piperx_black_plug_demo_v1"
    OUTPUT_DIR = "/home/standard/workspace/test/pistar/.cache/huggingface/lerobot"
    TASK_NAME = "put the white plug into the two-hole socket"
    FPS = 10
    NUM_EPISODES = 100

    # 主从设置：两根 CAN 线分别连接
    MASTER_CAN = "can_left_mas"
    SLAVE_CAN = "can_left_slave"
    MOVE_CHECK = True
    ENABLE_SOFT_TELEOP = True
    ENABLE_DRAG_TEACH = True
    USE_MIT_MODE = True
    USE_CTRL_FRAME = True
    FALLBACK_TO_FEEDBACK = False

    # 与 robocoin scripts/piperx/run_dagger.sh 一致的遥操控制链
    CONTROL_FPS = 30
    TEACHER_ACTION_EMA_ENABLED = True
    TEACHER_ACTION_EMA_ALPHA = 0.80
    TEACHER_ACTION_SLEW_ENABLED = True
    TEACHER_ACTION_MAX_JOINT_STEP = 0.040  # rad / control tick
    PIPERX_GRIPPER_RANGE_M = 0.07  # 与 PiperController 的 0..1 -> 0..70mm 映射一致
    TEACHER_ACTION_MAX_GRIPPER_STEP = 0.025 / PIPERX_GRIPPER_RANGE_M

    # 关节映射/缩放（如有方向相反可设为 -1）
    JOINT_SIGN = [1, 1, 1, 1, 1, 1]
    JOINT_OFFSET = [0, 0, 0, 0, 0, 0]
    JOINT_SCALE = 1.0
    GRIPPER_SCALE = 1.0
    GRIPPER_OFFSET = 0.0
    TEACHING_FRICTION = 1
    ALIGN_TO_FOLLOWER_ON_INIT = True
    ALIGNMENT_TIMEOUT = 2.0
    RESET_JOINT_POSITION = [
        0,
        1.0,
        -1.0,
        1.0,
        0,
        0,
    ]
    # ================================================

    print("=" * 60)
    print("LeRobot 软件主从遥操作采集模式 (从臂数据)")
    print("=" * 60)
    print(f"数据集 ID: {REPO_ID}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"任务名称: {TASK_NAME}")
    print(f"采集频率: {FPS} Hz")
    print(f"遥操控制频率: {CONTROL_FPS} Hz")
    print(f"计划收集: {NUM_EPISODES} 个 episodes")
    print(f"主臂 CAN: {MASTER_CAN}")
    print(f"从臂 CAN: {SLAVE_CAN}")
    print(f"从臂关节控制模式: {'MIT (0xAD)' if USE_MIT_MODE else '位置/速度 (0x00)'}")
    print(f"从臂初始化关节(rad): {RESET_JOINT_POSITION}")
    print("=" * 60)
    print("提示: 这是软件主从模式（无需硬件主从配置）")
    print("      请确保主臂处于拖动示教模式")
    print("=" * 60)

    # 创建多进程同步工具 (只创建一次)
    time_lock = Barrier(1 + 1)  # 1个robot进程 + 1个time_scheduler
    start_event = Event()
    finish_event = Event()
    exit_event = Event()
    saved_event = Event()
    discard_event = Event()  # 新增：放弃当前 episode 的事件

    # 机器人参数
    robot_kwargs = {
        "repo_id": REPO_ID,
        "output_dir": OUTPUT_DIR,
        "task_name": TASK_NAME,
        "fps": FPS,
        "move_check": MOVE_CHECK,
        "arm_can": SLAVE_CAN,
        "reset_joint_position": RESET_JOINT_POSITION,
        "use_mit_mode": USE_MIT_MODE,
    }
    teleop_kwargs = {
        "enabled": ENABLE_SOFT_TELEOP,
        "master_can": MASTER_CAN,
        "enable_drag_teach": ENABLE_DRAG_TEACH,
        "teaching_friction": TEACHING_FRICTION,
        "use_ctrl_frame": USE_CTRL_FRAME,
        "fallback_to_feedback": FALLBACK_TO_FEEDBACK,
        "control_fps": CONTROL_FPS,
        "teacher_action_ema_enabled": TEACHER_ACTION_EMA_ENABLED,
        "teacher_action_ema_alpha": TEACHER_ACTION_EMA_ALPHA,
        "teacher_action_slew_enabled": TEACHER_ACTION_SLEW_ENABLED,
        "teacher_action_max_joint_step": TEACHER_ACTION_MAX_JOINT_STEP,
        "teacher_action_max_gripper_step": TEACHER_ACTION_MAX_GRIPPER_STEP,
        "joint_sign": JOINT_SIGN,
        "joint_offset": JOINT_OFFSET,
        "joint_scale": JOINT_SCALE,
        "gripper_scale": GRIPPER_SCALE,
        "gripper_offset": GRIPPER_OFFSET,
        "align_to_follower_on_init": ALIGN_TO_FOLLOWER_ON_INIT,
        "alignment_timeout": ALIGNMENT_TIMEOUT,
    }

    # 启动机器人进程 (只启动一次)
    robot_process = Process(
        target=RobotWorkerLeRobot,
        args=(
            PiperSingleLeRobot,
            robot_kwargs,
            teleop_kwargs,
            time_lock,
            start_event,
            finish_event,
            exit_event,
            saved_event,
            discard_event,  # 新增参数
            "robot_worker_lerobot_slave",
        ),
    )
    robot_process.start()

    # 启动时间调度器 (只启动一次)
    time_scheduler = TimeScheduler(work_barrier=time_lock, time_freq=FPS)

    try:
        collected_episodes = 0  # 成功收集的 episode 数量

        while collected_episodes < NUM_EPISODES:
            print(f"\n{'='*60}")
            print(f"Episode {collected_episodes + 1}/{NUM_EPISODES}")
            print(f"{'='*60}")

            # 重置状态
            start_event.clear()
            finish_event.clear()
            saved_event.clear()
            discard_event.clear()  # 重置放弃标志

            # 等待用户按 Enter 开始
            input("按 Enter 键开始收集...")
            print("开始收集! (按 Enter 结束 | 按空格键放弃)")
            start_event.set()

            # 启动计时器
            time_scheduler.start()

            # 等待用户按 Enter 结束或按空格键放弃
            with _StdinCbreak():
                while True:
                    key = _read_key_nonblocking(timeout=0.1)
                    if key == ' ':
                        print("\n⚠ 检测到空格键 - 放弃当前 episode")
                        discard_event.set()
                        finish_event.set()
                        break
                    if key in ('\n', '\r'):
                        print("\n✓ 检测到 Enter 键 - 保存当前 episode")
                        finish_event.set()
                        break

            # 停止计时器 (暂停)
            time_scheduler.stop()

            # 强制打破屏障，唤醒正卡在 wait() 的 worker 进程
            try:
                time_lock.abort()
            except Exception:
                pass

            print("等待数据处理...", end="", flush=True)
            # 等待保存/放弃完成
            while not saved_event.is_set():
                time.sleep(0.1)
            print(" 完成!")

            # 重置 Barrier，防止因 scheduler 强制退出导致的 BrokenBarrierError
            try:
                time_lock.reset()
            except Exception:
                pass

            if discard_event.is_set():
                print(f"⚠ Episode {collected_episodes + 1} 已放弃（不计入总数）")
                print("继续采集下一个 episode...")
            else:
                print(f"✓ Episode {collected_episodes + 1} 完成！")
                print(f"  平均采集时间间隔: {time_scheduler.real_time_average_time_interval:.4f}s")
                collected_episodes += 1  # 只有保存成功才增加计数

            # 手动重置一下 start_event，让子进程进入下一个循环等待
            start_event.clear()

        print("\n" + "=" * 60)
        print("✅ 所有 episodes 收集完成！")
        print(f"数据集保存在: {OUTPUT_DIR}/{REPO_ID}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n程序中断，正在退出...")
    finally:
        # 清理
        exit_event.set()
        # 释放可能卡住的锁
        try:
            time_lock.reset()
        except Exception:
            pass

        if robot_process.is_alive():
            robot_process.join(timeout=2)
            if robot_process.is_alive():
                robot_process.terminate()
            robot_process.close()

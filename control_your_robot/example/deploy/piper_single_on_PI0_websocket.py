import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROL_ROOT = PROJECT_ROOT / "control_your_robot"
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(CONTROL_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

PIPER_ACTION_DIM = 7


def _load_task_instructions(task_name):
    root_dir = Path(__file__).resolve().parents[2]
    possible_paths = [
        root_dir / "task_instructions" / f"{task_name}.json",
        root_dir / "datasets" / "instructions" / f"{task_name}.json",
        Path("task_instructions") / f"{task_name}.json",
    ]

    for path in possible_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file_obj:
            instruction_dict = json.load(file_obj)
        instructions = instruction_dict.get("instructions", [])
        if instructions:
            return instructions

    print(f"Warning: instruction file not found for task '{task_name}', using task name as prompt.")
    return [task_name]


def _choose_instruction(task_name, fixed_instruction=None):
    if fixed_instruction:
        return fixed_instruction
    instructions = _load_task_instructions(task_name)
    return str(np.random.choice(instructions))


def _get_color_image(sensors, cam_keys):
    for cam_key in cam_keys:
        cam_data = sensors.get(cam_key)
        if cam_data is not None and "color" in cam_data:
            return cam_data["color"]
    return None


def _preprocess_image(image):
    resized = image_tools.resize_with_pad(np.asarray(image), 224, 224)
    return image_tools.convert_to_uint8(np.asarray(resized))


def input_transform(data, instruction, adv_ind=None):
    state_7d = np.concatenate(
        [
            np.asarray(data[0]["left_arm"]["joint"]).reshape(-1),
            np.asarray(data[0]["left_arm"]["gripper"]).reshape(-1),
        ]
    ).astype(np.float32)

    sensors = data[1]
    img_wrist = _get_color_image(sensors, ["cam_wrist", "wrist_image"])
    if img_wrist is None:
        raise KeyError(
            f"未找到 wrist 相机图像，当前可用键: {list(sensors.keys())}。"
            "期望其中之一: ['cam_wrist', 'wrist_image']"
        )

    img_head = _get_color_image(sensors, ["cam_head", "image"])
    if img_head is None:
        img_head = np.zeros_like(img_wrist)

    observation = {
        "state": state_7d,
        "images": {
            "cam_high": _preprocess_image(img_head),
            "cam_wrist": _preprocess_image(img_wrist),
        },
        "prompt": instruction,
    }
    if adv_ind is not None:
        observation["adv_ind"] = adv_ind
    return observation


def output_transform(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < PIPER_ACTION_DIM:
        raise ValueError(f"policy action must have at least {PIPER_ACTION_DIM} values, got {action.shape}")
    action_7d = action[:PIPER_ACTION_DIM]
    if not np.all(np.isfinite(action_7d)):
        raise ValueError("policy action must contain only finite values")

    return {
        "arm": {
            "left_arm": {
                "joint": action_7d[:6],
                "gripper": float(action_7d[6]),
            }
        }
    }


def _episode_indices(num_episode):
    if num_episode == -1:
        return itertools.count()
    return range(num_episode)


def _select_action_chunk(actions, chunk_size):
    action_chunk = np.asarray(actions, dtype=np.float32)
    if action_chunk.ndim != 2 or action_chunk.shape[0] == 0 or action_chunk.shape[1] < PIPER_ACTION_DIM:
        raise ValueError(
            f"policy actions must have shape (horizon, >={PIPER_ACTION_DIM}), got {action_chunk.shape}"
        )
    return action_chunk[:chunk_size]


def _cleanup_robot(robot):
    for sensor_group in getattr(robot, "sensors", {}).values():
        for sensor in sensor_group.values():
            cleanup = getattr(sensor, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    print(f"Warning: camera cleanup failed: {error}")
    for controller_group in getattr(robot, "controllers", {}).values():
        for controller in controller_group.values():
            sdk = getattr(controller, "controller", None)
            disconnect = getattr(sdk, "DisconnectPort", None)
            if disconnect is not None:
                try:
                    disconnect()
                except Exception as error:
                    print(f"Warning: CAN disconnect failed: {error}")


def _cleanup_policy_client(policy_client):
    websocket = getattr(policy_client, "_ws", None)
    close = getattr(websocket, "close", None)
    if close is not None:
        try:
            close()
        except Exception as error:
            print(f"Warning: websocket cleanup failed: {error}")


def _create_robot(arm_can):
    from my_robot.agilex_piper_single_base import PiperSingle

    return PiperSingle(arm_can=arm_can, use_mit_mode=True)


def main():
    from robot.utils.base.data_handler import is_enter_pressed

    parser = argparse.ArgumentParser(
        description="PI0 单臂 websocket 远程部署脚本（默认 pi05，传 adv_ind 时兼容 PiStar）"
    )
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="远端 websocket 推理服务器地址")
    parser.add_argument("--server-port", type=int, default=8000, help="远端 websocket 推理服务器端口")
    parser.add_argument("--arm-can", type=str, default="can_left_slave", help="PiperX 从臂 CAN 接口")
    parser.add_argument(
        "--task-name",
        type=str,
        default="Pick up the black plug and insert it into the white two-hole socket.",
        help="任务名称",
    )
    parser.add_argument("--instruction", type=str, default=None, help="显式指定 prompt；不传则从任务文件中随机采样")
    parser.add_argument(
        "--adv-ind",
        type=str,
        default=None,
        help="可选。传入时会把请求按 PiStar 方式附带 adv_ind，例如 positive/negative",
    )
    parser.add_argument("--chunk-size", type=int, default=10, help="每次仅执行动作块前多少步")
    parser.add_argument("--control-freq", type=float, default=30.0, help="本地 CAN 控制频率")
    parser.add_argument("--max-step", type=int, default=600, help="单个 episode 最大步数")
    parser.add_argument("--num-episode", type=int, default=1, help="episode 数量；-1 表示无限次")
    args = parser.parse_args()
    if args.num_episode == 0 or args.num_episode < -1:
        parser.error("--num-episode must be -1 or a positive integer")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.control_freq <= 0:
        parser.error("--control-freq must be positive")
    if args.max_step <= 0:
        parser.error("--max-step must be positive")

    robot = None
    policy_client = None
    try:
        print("=" * 50)
        print("PI0 单臂 websocket 远程部署脚本")
        print("=" * 50)
        print(f"服务器地址: ws://{args.server_host}:{args.server_port}")
        print(f"从臂 CAN: {args.arm_can}")
        print(f"任务名称: {args.task_name}")
        print(f"adv_ind: {args.adv_ind if args.adv_ind is not None else 'None (pi05 default)'}")
        print("=" * 50)

        print("\n[1/3] 初始化机器人...")
        robot = _create_robot(args.arm_can)
        robot.set_up()
        print("✓ 机器人初始化完成")

        print("\n[2/3] 连接 websocket 推理服务器...")
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host,
            port=args.server_port,
        )
        server_metadata = policy_client.get_server_metadata()
        print(f"✓ 服务器连接成功，metadata: {server_metadata}")

        deploy_mode = server_metadata.get("deploy_mode", "unknown")
        requires_field = server_metadata.get("requires_adv_ind")
        requires_adv_ind = bool(requires_field) or deploy_mode == "pi05star"
        if requires_adv_ind and not args.adv_ind:
            parser.error(
                "connected server requires adv_ind (PiStar), but --adv-ind was not provided."
            )
        metadata_declares_sft = requires_field is not None and not bool(requires_field)
        effective_adv_ind = None if metadata_declares_sft else args.adv_ind
        if metadata_declares_sft and args.adv_ind:
            print("Warning: 当前服务端为普通 pi0.5，已忽略 adv_ind。")
        print(f"服务端模式: {deploy_mode}")

        episode_total = "无限" if args.num_episode == -1 else str(args.num_episode)
        print(f"\n[3/3] 准备执行 {episode_total} 个 episode")
        print("-" * 50)

        for episode_idx in _episode_indices(args.num_episode):
            step = 0
            instruction = _choose_instruction(args.task_name, args.instruction)

            episode_label = str(episode_idx + 1)
            if args.num_episode != -1:
                episode_label += f"/{args.num_episode}"
            print(f"\n{'=' * 20} Episode {episode_label} {'=' * 20}")
            print(f"Prompt: {instruction}")

            robot.reset()

            print("\n按 Enter 键开始推理...")
            is_start = False
            while not is_start:
                if is_enter_pressed():
                    is_start = True
                    print("✓ 开始执行任务...")
                else:
                    time.sleep(0.1)

            while step < args.max_step:
                observation = input_transform(robot.get(), instruction, effective_adv_ind)
                response = policy_client.infer(observation)
                action_chunk = _select_action_chunk(response["actions"], args.chunk_size)

                next_send_time = time.monotonic()
                for action in action_chunk:
                    sleep_seconds = next_send_time - time.monotonic()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    robot.move(output_transform(action))
                    step += 1
                    next_send_time += 1.0 / args.control_freq

                    if is_enter_pressed():
                        print("\n用户结束当前 episode")
                        is_start = False
                        break

                    if step >= args.max_step:
                        break

                if not is_start:
                    break

            print(f"✓ Episode {episode_idx + 1} 完成 (总步数: {step})")
    finally:
        if policy_client is not None:
            _cleanup_policy_client(policy_client)
        if robot is not None:
            _cleanup_robot(robot)

    if args.num_episode != -1:
        print("\n" + "=" * 50)
        print("全部 episode 执行完成！")
        print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C received, client stopped normally.")

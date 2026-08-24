#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
关节控制脚本

功能：
    控制指定CAN端口对应的机械臂的6个关节角度和夹爪
    脚本会自动检查使能状态，如未使能则自动使能
    支持同步控制关节和夹爪，实现协调运动

使用方法：
    python3 ctrl_joint.py [can_name] [j1] [j2] [j3] [j4] [j5] [j6] [gripper] [speed]

参数说明：
    can_name: CAN端口名称，默认为 'can_left_slave'
              本任务使用：'can_left_slave', 'can_left_mas'
    j1-j6:   6个关节的目标角度（单位：度），可选参数，默认为0
             参数顺序对应J1到J6关节
    gripper: 夹爪行程（单位：m），可选参数，默认为0（闭合）
             0表示完全闭合，正值表示打开到指定行程（例如：0.05表示50mm）
    speed:   运动速度百分比（0-100），默认为10，可选参数
             0表示最慢，100表示最快

使用示例：
    # 使用默认参数（所有关节0度，夹爪闭合，速度10%）
    python3 ctrl_joint.py

    # 指定CAN端口和关节角度（夹爪保持闭合）
    python3 ctrl_joint.py can_left_slave 0 30 -45 0 0 0

    # 指定关节角度和夹爪（打开到0.05m，即50mm）
    python3 ctrl_joint.py can_left_slave 0 30 -45 0 0 0 0.05

    # 指定所有参数（关节、夹爪和速度）
    python3 ctrl_joint.py can_left_mas 0 30 -45 0 0 0 0.05 80

前置条件：
    1. 需要先安装piper_sdk: pip3 install piper_sdk
    2. 需要先激活CAN设备（使用can_activate.sh或相关工具）
    3. 机械臂需要已连接并上电
    4. CAN端口名称需要与实际设备匹配

工作流程：
    1. 解析命令行参数
    2. 检查关节角度限制
    3. 连接CAN端口
    4. 检查并自动使能机械臂
    5. 初始化夹爪（清除错误并使能）
    6. 设置关节控制模式
    7. 发送关节和夹爪控制指令

关节角度限制（度）：
    J1: [-150.0, 150.0]  J2: [0, 180.0]      J3: [-170, 0]
    J4: [-100.0, 100.0]  J5: [-70.0, 70.0]   J6: [-120.0, 120.0]

夹爪参数说明：
    单位：m（米）
    范围：0表示完全闭合，最大行程根据实际夹爪型号而定（通常0-0.07m，即0-70mm）
    力矩：固定为1N·m（1000 * 0.001N·m）
    注意：夹爪会在关节控制前自动初始化（清除错误并使能）
"""

import time
import sys
from piper_sdk import *

# 关节角度限制（度）
# 格式：关节编号: (最小角度, 最大角度)
JOINT_LIMITS = {
    1: (-150.0, 150.0),  # J1关节：左右旋转范围
    2: (0, 180.0),       # J2关节：向上抬起范围
    3: (-170, 0),        # J3关节：向下弯曲范围
    4: (-100.0, 100.0),  # J4关节：旋转范围
    5: (-70.0, 70.0),    # J5关节：俯仰范围
    6: (-120.0, 120.0)   # J6关节：末端旋转范围
}

def check_joint_limits(joint_angles_deg):
    """
    检查关节角度是否在限制范围内

    Args:
        joint_angles_deg: 6个关节的角度列表（单位：度）

    Returns:
        bool: 如果所有角度都在限制范围内返回True，否则返回False
    """
    for i, angle in enumerate(joint_angles_deg, start=1):
        min_angle, max_angle = JOINT_LIMITS[i]
        if angle < min_angle or angle > max_angle:
            print(f"⚠ 警告: 关节{i}角度 {angle}° 超出限制范围 [{min_angle}, {max_angle}]°")
            return False
    return True

def deg_to_sdk_unit(angle_deg):
    """
    将角度（度）转换为SDK单位（0.001度）

    Args:
        angle_deg: 角度值（度）

    Returns:
        int: SDK单位的角度值（0.001度）
    """
    return int(round(angle_deg * 1000))

def m_to_sdk_unit(m):
    """
    将夹爪行程（m）转换为SDK单位（0.001mm）

    Args:
        m: 夹爪行程（米）

    Returns:
        int: SDK单位的夹爪行程（0.001mm）
        转换公式：1m = 1000mm = 1000000 * 0.001mm
    """
    return int(round(m * 1000000))

def main():
    """
    主函数：解析命令行参数，连接机械臂，控制关节和夹爪
    """
    # 解析CAN端口名称（第1个参数，索引1）
    can_name = sys.argv[1] if len(sys.argv) > 1 else "can_left_slave"

    # 解析关节角度参数（第2-7个参数，索引2-7，单位：度）
    joint_angles_deg = [0.0] * 6
    if len(sys.argv) > 2:
        try:
            for i in range(6):
                if i + 2 < len(sys.argv):
                    joint_angles_deg[i] = float(sys.argv[i + 2])
        except ValueError as e:
            print(f"错误: 关节角度参数无效 - {e}")
            print("请使用数字格式，例如: 0 30 -45 0 0 0")
            sys.exit(1)

    # 解析夹爪参数（第8个参数，索引8，单位：m，0表示闭合）
    gripper_m = 0.0
    if len(sys.argv) > 8:
        try:
            gripper_m = float(sys.argv[8])
            if gripper_m < 0:
                print(f"警告: 夹爪行程不能为负数，将使用默认值0（闭合）")
                gripper_m = 0.0
        except ValueError:
            print(f"警告: 夹爪参数无效，将使用默认值0（闭合）")

    # 解析速度参数（第9个参数，索引9，范围：0-100，默认10%）
    speed = 10
    if len(sys.argv) > 9:
        try:
            speed = int(sys.argv[9])
            if speed < 0 or speed > 100:
                print(f"警告: 速度参数 {speed} 超出范围 [0, 100]，将使用默认值10")
                speed = 10
        except ValueError:
            print("警告: 速度参数无效，将使用默认值10")

    # 检查关节角度是否在限制范围内
    if not check_joint_limits(joint_angles_deg):
        print("\n是否继续执行？(y/n): ", end='')
        try:
            response = input().strip().lower()
            if response != 'y':
                print("已取消操作")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n已取消操作")
            sys.exit(0)

    try:
        # 创建接口实例并连接CAN端口
        piper = C_PiperInterface_V2(can_name=can_name)
        piper.ConnectPort()
        time.sleep(0.1)  # 等待连接稳定

        # 检查使能状态，如未使能则自动使能
        enable_list = piper.GetArmEnableStatus()
        if not all(enable_list):
            # 尝试使能所有电机（参数7表示所有关节电机）
            piper.EnableArm(7)
            time.sleep(0.1)

            # 等待使能完成（最多等待5秒）
            max_wait_time = 5.0
            wait_start = time.time()
            enable_success = False

            while time.time() - wait_start < max_wait_time:
                enable_list = piper.GetArmEnableStatus()
                if all(enable_list):
                    enable_success = True
                    break
                time.sleep(0.1)

            # 如果使能失败，提示用户选择是否继续
            if not enable_success:
                print("⚠ 警告: 部分电机未能使能")
                print("请检查：")
                print("1. 机械臂是否已上电")
                print("2. CAN总线连接是否正常")
                print("3. 机械臂是否有错误状态")
                print("\n是否继续执行关节控制？(y/n): ", end='')
                try:
                    response = input().strip().lower()
                    if response != 'y':
                        print("已取消操作")
                        sys.exit(0)
                except KeyboardInterrupt:
                    print("\n已取消操作")
                    sys.exit(0)

        # 初始化夹爪：先清除错误，再使能
        # 这是必要的步骤，确保夹爪处于正常工作状态
        # GripperCtrl参数说明：
        #   gripper_angle: 夹爪行程（0.001mm单位），0表示当前位置
        #   gripper_effort: 夹爪力矩（0.001N·m单位），1000表示1N·m
        #   gripper_code: 控制码
        #                 0x00=失能
        #                 0x01=使能
        #                 0x02=失能并清除错误
        #                 0x03=使能并清除错误
        #   set_zero: 零点设置，0=不设置，0xAE=设置当前位置为零点
        piper.GripperCtrl(0, 1000, 0x02, 0)  # 步骤1：失能并清除错误
        time.sleep(0.1)  # 等待指令执行
        piper.GripperCtrl(0, 1000, 0x01, 0)  # 步骤2：使能夹爪
        time.sleep(0.1)  # 等待使能完成

        # 设置控制模式为关节控制模式（MOVE J模式）
        # 此模式允许直接控制各个关节的角度
        # MotionCtrl_2参数说明：
        #   ctrl_mode: 控制模式
        #              0x01 = CAN指令控制模式（从机模式）
        #              0x00 = 示教器控制模式（主机模式）
        #   move_mode: 运动模式
        #              0x01 = MOVE J（关节空间运动）
        #              0x02 = MOVE L（直线运动）
        #              0x03 = MOVE C（圆弧运动）
        #              0x04 = MOVE P（点到点运动）
        #   move_spd_rate_ctrl: 运动速度百分比（0-100）
        #                        0表示最慢，100表示最快
        #   is_mit_mode: MIT模式标志
        #                0x00 = 位置速度模式（标准模式）
        #                0x01 = MIT模式（力矩控制模式）
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        time.sleep(0.1)  # 等待模式设置完成

        # 步骤1：将关节角度转换为SDK单位（0.001度）并发送关节控制指令
        # SDK使用0.001度作为单位，例如：30度 = 30000（0.001度单位）
        joint_angles_sdk = [deg_to_sdk_unit(angle) for angle in joint_angles_deg]
        piper.JointCtrl(
            joint_angles_sdk[0],  # J1：基座旋转
            joint_angles_sdk[1],  # J2：大臂抬起
            joint_angles_sdk[2],  # J3：小臂弯曲
            joint_angles_sdk[3],  # J4：腕部旋转
            joint_angles_sdk[4],  # J5：腕部俯仰
            joint_angles_sdk[5]   # J6：末端旋转
        )

        # 步骤2：将夹爪行程转换为SDK单位（0.001mm）并发送夹爪控制指令
        # SDK使用0.001mm作为单位，例如：0.05m = 50mm = 50000（0.001mm单位）
        # 参数说明：
        #   gripper_angle_sdk: 目标行程（0.001mm单位）
        #   gripper_effort: 夹爪力矩，1000 = 1N·m（固定值）
        #   gripper_code: 0x01 = 使能状态（保持使能）
        #   set_zero: 0 = 不设置零点
        gripper_angle_sdk = m_to_sdk_unit(gripper_m)
        piper.GripperCtrl(gripper_angle_sdk, 1000, 0x01, 0)

        # 输出执行结果
        print("✓ 关节控制指令已发送")
        if gripper_m > 0:
            print(f"✓ 夹爪控制指令已发送（行程: {gripper_m}m）")
        else:
            print("✓ 夹爪控制指令已发送（闭合）")

    except KeyboardInterrupt:
        # 用户主动中断，直接退出
        sys.exit(0)
    except Exception as e:
        # 控制失败时给出简要错误信息
        print(f"关节控制失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

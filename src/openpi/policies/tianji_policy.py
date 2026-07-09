"""天机双臂（bi_tianji_marvin）+ LeRobot dataset 的 pi0/pi05 policy 适配。

数据集 schema（新 30D 同构 obs / action）
=======================================
- ``observation.state`` = 30D，双臂对称，每侧 15D：

    [ 0.. 2]  ee_x / y / z          —— mm，SDK 原生单位
    [ 3.. 6]  ee_qx / qy / qz / qw  —— 四元数（scipy xyzw）
    [ 7..13]  joint_1 .. joint_7    —— deg
    [14]      trigger               —— [0, 1]

- ``action`` = 30D，格式跟 ``observation.state`` 完全一致。

送给 model 的 schema（老 20D EE-r6d，跟 robot ``_absolute_T`` 的 r6d fallback 对齐）
================================================================================
- action 恒为 **20D EE**（每侧 10D）：

    [0..2]  x / y / z    —— m（数据集 mm → ÷1000）
    [3..8]  r6d_1..6     —— 由 quat 转矩阵前两列
    [9]     trigger

- state 有两档，``state_mode``:

    - "ee"    （默认）20D EE：跟 action 同 schema
    - "joint" 16D：每侧 7 joints(deg) + 1 trigger（丢 ee 信息）

``both`` 模式已废弃，不再支持。
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import einops
import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi import transforms
from openpi.models import model as _model

# ---- 数据集常量 ----------------------------------------------------------

_PER_SIDE_DIMS = 15
_EE_XYZ_DIMS = 3        # v15[0:3]  ee 平移 (mm)
_EE_QUAT_DIMS = 4       # v15[3:7]  ee 四元数 (xyzw)
_JOINT_DIMS = 7         # v15[7:14] 关节角 (deg)
_TRIGGER_IDX = 14       # v15[14]   trigger

_MM_TO_M = 1e-3

# ---- Model 侧维度 --------------------------------------------------------
# action 恒 20D EE（每侧 10D：xyz(3) + r6d(6) + trigger(1)），跟推理客户端契约一致。
TIANJI_ACTION_DIM = 20

StateMode = Literal["ee", "joint"]


# ---- 单侧 15D → 10D / 8D 缩减 ---------------------------------------------


def _quat_to_r6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """quat (..., 4, xyzw) → r6d (..., 6)（旋转矩阵前两列拼接，Zhou et al. 2019）。"""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    orig_shape = q.shape[:-1]
    q_flat = q.reshape(-1, 4)
    rotmat = R.from_quat(q_flat).as_matrix()  # (N, 3, 3)
    col0 = rotmat[:, :, 0]  # (N, 3)
    col1 = rotmat[:, :, 1]  # (N, 3)
    r6d_flat = np.concatenate([col0, col1], axis=-1)  # (N, 6)
    return r6d_flat.reshape(*orig_shape, 6)


def _side_to_ee_r6d(v15: np.ndarray) -> np.ndarray:
    """单侧 15D (mm + quat + joints + trigger) → 10D (m + r6d + trigger)。支持 batch。"""
    v = np.asarray(v15, dtype=np.float32)
    xyz_m = v[..., 0:_EE_XYZ_DIMS] * _MM_TO_M
    quat = v[..., _EE_XYZ_DIMS : _EE_XYZ_DIMS + _EE_QUAT_DIMS]
    r6d = _quat_to_r6d(quat).astype(np.float32)
    trigger = v[..., _TRIGGER_IDX : _TRIGGER_IDX + 1]
    return np.concatenate([xyz_m, r6d, trigger], axis=-1)


def _side_to_joint(v15: np.ndarray) -> np.ndarray:
    """单侧 15D → 8D (7 joints + 1 trigger)。丢 ee。"""
    v = np.asarray(v15, dtype=np.float32)
    joints = v[..., 7 : 7 + _JOINT_DIMS]
    trigger = v[..., _TRIGGER_IDX : _TRIGGER_IDX + 1]
    return np.concatenate([joints, trigger], axis=-1)


# ---- 双臂 30D 缩减 --------------------------------------------------------


def _reduce_state_30d(v30: np.ndarray, mode: StateMode) -> np.ndarray:
    """双臂 30D state → 送给 model 的 state。

    - ``mode="ee"``:    30 → 20（每侧 10D）
    - ``mode="joint"``: 30 → 16（每侧 8D）
    """
    v = np.asarray(v30, dtype=np.float32)
    left = v[..., 0:_PER_SIDE_DIMS]
    right = v[..., _PER_SIDE_DIMS : 2 * _PER_SIDE_DIMS]
    if mode == "ee":
        return np.concatenate([_side_to_ee_r6d(left), _side_to_ee_r6d(right)], axis=-1)
    if mode == "joint":
        return np.concatenate([_side_to_joint(left), _side_to_joint(right)], axis=-1)
    raise ValueError(f"state_mode 须为 'ee' 或 'joint'，实际为 {mode!r}")


def _reduce_action_30d(a30: np.ndarray) -> np.ndarray:
    """双臂 30D action → 20D EE（无论 state_mode，action 恒 EE 格式）。

    支持 (30,) 单帧、(H, 30) chunk、(B, H, 30) batched chunk。
    """
    a = np.asarray(a30, dtype=np.float32)
    left = a[..., 0:_PER_SIDE_DIMS]
    right = a[..., _PER_SIDE_DIMS : 2 * _PER_SIDE_DIMS]
    return np.concatenate([_side_to_ee_r6d(left), _side_to_ee_r6d(right)], axis=-1)


# ---- 34D r6d 兼容分支（仅推理路径） --------------------------------------
# robocoin 推理客户端 ``_STATE_KEYS`` 的排布（xyz 已是 m，rotation 已是 r6d）：
#   [ 0.. 6]  left_joint_1..7_pos  (deg)
#   [ 7..13]  right_joint_1..7_pos (deg)
#   [14..16]  left_ee_x/y/z         (m)
#   [17..22]  left_ee_r6d_1..6
#   [23..25]  right_ee_x/y/z        (m)
#   [26..31]  right_ee_r6d_1..6
#   [32]      left_gripper
#   [33]      right_gripper
# 30D 训练路径出的 20D = [xyz_L(m), r6d_L, trig_L, xyz_R(m), r6d_R, trig_R]，
# 这里 34D 直接切片拼装出数值等价的 20D，无需 quat→r6d / mm→m 转换。


def _reduce_state_34d_r6d(v34: np.ndarray, mode: StateMode) -> np.ndarray:
    """双臂 34D r6d state（推理端 obs）→ 送给 model 的 state。

    输出跟 :func:`_reduce_state_30d` 数值等价，用来兼容还没升级到 30D 输出的推理客户端。
    """
    v = np.asarray(v34, dtype=np.float32)
    if mode == "ee":
        left = np.concatenate(
            [v[..., 14:17], v[..., 17:23], v[..., 32:33]], axis=-1
        )
        right = np.concatenate(
            [v[..., 23:26], v[..., 26:32], v[..., 33:34]], axis=-1
        )
        return np.concatenate([left, right], axis=-1)
    if mode == "joint":
        left = np.concatenate([v[..., 0:7], v[..., 32:33]], axis=-1)
        right = np.concatenate([v[..., 7:14], v[..., 33:34]], axis=-1)
        return np.concatenate([left, right], axis=-1)
    raise ValueError(f"state_mode 须为 'ee' 或 'joint'，实际为 {mode!r}")


# ---- 测试样本 -------------------------------------------------------------


def make_tianji_example() -> dict:
    """随机 example 用来测试 pipeline。obs.state 30D 新格式。"""
    state = np.zeros(30, dtype=np.float32)
    for side_start in (0, 15):
        # xyz mm
        state[side_start : side_start + 3] = np.random.uniform(-500, 500, 3)
        # quat（保证归一化）
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)
        state[side_start + 3 : side_start + 7] = q
        # joints deg
        state[side_start + 7 : side_start + 14] = np.random.uniform(-120, 120, 7)
        # trigger
        state[side_start + 14] = np.random.rand()
    return {
        "observation/state": state,
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "flip the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ---- Transforms ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TianjiInputs(transforms.DataTransformFn):
    """LeRobot tianji 30D frame → model 输入（state 20D 或 16D，action 20D EE）。

    ``state_mode``:
      - "ee"    （默认）20D：每侧 10D = xyz(m) + r6d + trigger，跟 action 同 schema
      - "joint" 16D：每侧 8D = 7 joints(deg) + trigger（丢 ee 信息）

    ``action`` 无论哪个 state_mode，都是 20D EE 格式（模型学的、推理客户端拿到的都是这个）。
    """

    model_type: _model.ModelType
    state_mode: StateMode = "ee"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        full_state = np.asarray(data["observation/state"])
        dim = full_state.shape[-1]
        if dim == 30:
            # 训练路径：数据集 30D (mm + quat + joints + trigger)
            state = _reduce_state_30d(full_state, self.state_mode)
        elif dim == 34:
            # 推理路径：robocoin 客户端 34D (m + r6d + joints + gripper)，直接切片拼装
            state = _reduce_state_34d_r6d(full_state, self.state_mode)
        else:
            raise ValueError(
                f"observation/state 期望 30D（训练）或 34D（推理），实际 {dim}D。"
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image_left,
                "right_wrist_0_rgb": wrist_image_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions_30d = np.asarray(data["actions"])
            if actions_30d.shape[-1] != 30:
                raise ValueError(
                    f"actions 期望 30D（新 action schema），实际 {actions_30d.shape[-1]}D。"
                )
            inputs["actions"] = _reduce_action_30d(actions_30d)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiOutputs(transforms.DataTransformFn):
    """model 输出 → 前 20 维 EE action（剩余是 pad）。

    输出 schema 跟推理客户端 ``_ACTION_KEYS``、robot 的 ``_absolute_T`` r6d 分支完全对齐：
    left_x / y / z / r6d_1..6 / trigger + right 同顺序 = 20D。
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :TIANJI_ACTION_DIM])}

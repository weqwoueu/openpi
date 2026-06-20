"""天机双臂（bi_tianji_marvin）+ LeRobot dataset 的 pi0/pi05 policy 适配。

- dataset.observation.state: 34D = 14 关节角(deg) + 9 左末端(xyz+r6d) + 9 右末端(xyz+r6d) + 2 grippers
- dataset.action:             20D = 3 左 xyz + 6 左 r6d + 1 trigger + 3 右 xyz + 6 右 r6d + 1 trigger
- 摄像头键名: observation.images.cam_high / .cam_left_wrist / .cam_right_wrist

``state_mode`` 选择喂给模型的 state schema：
  - "ee"    （默认）20D：左右末端(xyz+r6d) + grippers，跟 action 同 schema
  - "joint"        16D：左右 7 关节角(deg) + grippers
  - "both"         34D：全要（要求 model.action_dim ≥ 34）
"""

import dataclasses
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 数据集里 action 实际维度，model 内部会 pad 到 model_config.action_dim
TIANJI_ACTION_DIM = 20
# state 维度布局：[14 关节] + [9 左 ee] + [9 右 ee] + [2 grippers]
_JOINT_DIMS = 14
_EE_DIMS = 18  # 左 9 + 右 9
_GRIPPER_DIMS = 2

StateMode = Literal["ee", "joint", "both"]


def _select_state(full_state: np.ndarray, mode: StateMode) -> np.ndarray:
    """按 state_mode 从 34D dataset state 里抽出送给 model 的 state。"""
    if mode == "both":
        return full_state
    if mode == "ee":
        # 丢前 14 维关节，保留 ee + grippers → 20D
        return full_state[..., _JOINT_DIMS:]
    if mode == "joint":
        # 留前 14 维关节 + 末尾 2 维 grippers，丢中间 18 维 ee → 16D
        joints = full_state[..., :_JOINT_DIMS]
        grippers = full_state[..., _JOINT_DIMS + _EE_DIMS:]
        return np.concatenate([joints, grippers], axis=-1)
    raise ValueError(f"state_mode 须为 'ee' | 'joint' | 'both'，实际为 {mode!r}")


def make_tianji_example() -> dict:
    """Random example for testing the input pipeline."""
    return {
        # 34D: 14 joints + 9 left ee + 9 right ee + 2 grippers，模型实际只用后 20D
        "observation/state": np.random.rand(34).astype(np.float32),
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


@dataclasses.dataclass(frozen=True)
class TianjiInputs(transforms.DataTransformFn):
    """LeRobot tianji frame -> model 输入。

    ``state_mode``:
      - "ee"   （默认）20D：左右末端(xyz+r6d) + grippers，跟 action 同 schema，pi05 训练最稳
      - "joint"        16D：左右 7 关节角(deg) + grippers
      - "both"         34D：全部信息（要 model.action_dim ≥ 34）
    """

    model_type: _model.ModelType
    state_mode: StateMode = "ee"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        full_state = np.asarray(data["observation/state"])
        state = _select_state(full_state, self.state_mode)

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
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiOutputs(transforms.DataTransformFn):
    """model 输出 -> 前 20 维（剩余是 pad）。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :TIANJI_ACTION_DIM])}

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms

PIPER_ACTION_DIM = 7


def make_piper_example() -> dict:
    """Creates a random input example for Piper policy."""
    return {
        "state": np.ones((7,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for Piper single-arm policy.

    Expected inputs:
    - images: dict with "cam_high" and "cam_wrist"
    - state: [7] (6 joints + 1 gripper)
    - actions: [action_horizon, 7]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_wrist")

    def __call__(self, data: dict) -> dict:
        state = _require_state(data["state"])
        in_images = data["images"]
        missing_cameras = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing_cameras:
            raise ValueError(f"Missing PiperX cameras: {tuple(sorted(missing_cameras))}")

        base_image = _convert_image(in_images["cam_high"], "cam_high")
        wrist_image = _convert_image(in_images["cam_wrist"], "cam_wrist")

        images = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),
        }
        image_masks = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = _require_action_chunk(data["actions"], exact_dim=True)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "adv_ind" in data:
            inputs["adv_ind"] = data["adv_ind"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for Piper policy."""

    def __call__(self, data: dict) -> dict:
        actions = _require_action_chunk(data["actions"], exact_dim=False)
        return {"actions": actions[:, :PIPER_ACTION_DIM]}


def _require_state(value) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (PIPER_ACTION_DIM,):
        raise ValueError(f"PiperX state must have shape ({PIPER_ACTION_DIM},), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("PiperX state must contain only finite values")
    return state


def _require_action_chunk(value, *, exact_dim: bool) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    valid_dim = actions.ndim == 2 and (
        actions.shape[1] == PIPER_ACTION_DIM if exact_dim else actions.shape[1] >= PIPER_ACTION_DIM
    )
    if not valid_dim:
        expected = str(PIPER_ACTION_DIM) if exact_dim else f">={PIPER_ACTION_DIM}"
        raise ValueError(f"PiperX actions must have shape (horizon, {expected}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("PiperX actions must contain only finite values")
    return actions


def _convert_image(value, camera_name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"PiperX {camera_name} image must be rank 3, got {image.shape}")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"PiperX {camera_name} image must have 3 channels, got {image.shape}")

    if not np.all(np.isfinite(image)):
        raise ValueError(f"PiperX {camera_name} image must contain only finite values")
    if np.issubdtype(image.dtype, np.floating):
        if image.size and image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8, copy=False)

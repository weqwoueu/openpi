import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

PIPERX_ACTION_DIM = 14


def make_piperx_example() -> dict:
    """Creates a random input example for the dual-arm PiperX policy."""
    return {
        "state": np.ones((PIPERX_ACTION_DIM,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class PiperXInputs(transforms.DataTransformFn):
    """Converts dual-arm PiperX observations and actions to model inputs."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        state = _require_last_dim(data["state"], PIPERX_ACTION_DIM, "state")

        in_images = data["images"]
        if extra_cameras := set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Unexpected PiperX cameras: {tuple(sorted(extra_cameras))}")
        if "cam_high" not in in_images:
            raise ValueError('PiperX inputs must contain "cam_high"')

        converted_images = {name: _convert_image(image) for name, image in in_images.items()}
        base_image = converted_images["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        for destination, source in {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }.items():
            if source in converted_images:
                images[destination] = converted_images[source]
                image_masks[destination] = np.True_
            else:
                images[destination] = np.zeros_like(base_image)
                image_masks[destination] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data and "rtc_actions" in data:
            raise ValueError('PiperX inputs cannot contain both "actions" and "rtc_actions"')
        action_key = "rtc_actions" if "rtc_actions" in data else "actions"
        if action_key in data:
            actions = _require_last_dim(data[action_key], PIPERX_ACTION_DIM, action_key)
            if actions.ndim < 2:
                raise ValueError(f"PiperX {action_key} must be an action chunk, got shape {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperXOutputs(transforms.DataTransformFn):
    """Extracts the 14D PiperX runtime action chunk from model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :PIPERX_ACTION_DIM]}


def _require_last_dim(value, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[-1] != expected_dim:
        raise ValueError(f"PiperX {name} must have last dimension {expected_dim}, got shape {array.shape}")
    return array


def _convert_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"PiperX images must be rank 3, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[-1] != 3:
        raise ValueError(f"PiperX images must have 3 channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if image.size and np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image

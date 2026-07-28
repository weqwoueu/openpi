from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        policy_inputs = dict(obs)
        rtc_guidance_weights = policy_inputs.pop("rtc_guidance_weights", None)
        rtc_num_steps = policy_inputs.pop("rtc_num_steps", None)
        rtc_max_guidance_weight = policy_inputs.pop("rtc_max_guidance_weight", None)
        rtc_enabled = "rtc_actions" in policy_inputs
        rtc_fields = (
            rtc_guidance_weights is not None,
            rtc_num_steps is not None,
            rtc_max_guidance_weight is not None,
        )
        if not (rtc_enabled == rtc_fields[0] == rtc_fields[1] == rtc_fields[2]):
            raise ValueError(
                "RTC 请求必须同时提供 rtc_actions、rtc_guidance_weights、"
                "rtc_num_steps 和 rtc_max_guidance_weight。"
            )
        if rtc_enabled and self._is_pytorch_model:
            raise ValueError("Inference-time RTC 当前只支持 JAX Pi0/Pi0.5 模型。")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, policy_inputs)
        inputs = self._input_transform(inputs)
        guidance_actions = inputs.pop("actions", None) if rtc_enabled else None
        if rtc_enabled and guidance_actions is None:
            raise ValueError("当前 policy input transform 不支持 rtc_actions。")

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if rtc_enabled:
            guidance_actions_np = np.asarray(guidance_actions)
            expected_action_shape = (self._model.action_horizon, self._model.action_dim)
            if guidance_actions_np.shape != expected_action_shape:
                raise ValueError(
                    f"RTC guidance actions shape {guidance_actions_np.shape}，"
                    f"期望 {expected_action_shape}。"
                )
            if not np.all(np.isfinite(guidance_actions_np)):
                raise ValueError("RTC guidance actions 包含 NaN 或 Inf。")
            guidance_actions = jnp.asarray(guidance_actions_np)[np.newaxis, ...]
            guidance_weights_np = np.asarray(rtc_guidance_weights, dtype=np.float32)
            if not np.all(np.isfinite(guidance_weights_np)):
                raise ValueError("rtc_guidance_weights 包含 NaN 或 Inf。")
            if np.any((guidance_weights_np < 0) | (guidance_weights_np > 1)):
                raise ValueError("rtc_guidance_weights 必须在 [0, 1] 范围内。")
            guidance_weights = jnp.asarray(guidance_weights_np)
            if guidance_weights.shape != guidance_actions.shape[-2:-1]:
                raise ValueError(
                    "rtc_guidance_weights shape "
                    f"{guidance_weights.shape} 与动作 horizon {guidance_actions.shape[-2]} 不匹配。"
                )
            if int(rtc_num_steps) <= 0:
                raise ValueError("rtc_num_steps 必须大于 0。")
            rtc_max_guidance_weight = float(rtc_max_guidance_weight)
            if not np.isfinite(rtc_max_guidance_weight) or rtc_max_guidance_weight <= 0:
                raise ValueError("rtc_max_guidance_weight 必须大于 0。")
            sample_kwargs.update(
                guidance_actions=guidance_actions,
                guidance_weights=guidance_weights[np.newaxis, ...],
                max_guidance_weight=rtc_max_guidance_weight,
                num_steps=int(rtc_num_steps),
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

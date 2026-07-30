import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class _FakeModel:
    action_horizon = 6
    action_dim = 4

    def __init__(self, *, train_time_rtc: bool):
        self.train_time_rtc = train_time_rtc
        self.last_sample_kwargs = None

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation
        self.last_sample_kwargs = kwargs
        return jnp.zeros((1, self.action_horizon, self.action_dim), dtype=jnp.float32)


def _map_rtc_actions(data: dict) -> dict:
    data = dict(data)
    if "rtc_actions" in data:
        data["actions"] = data.pop("rtc_actions")
    return data


def _rtc_request() -> dict:
    return {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.True_},
        "state": np.zeros(4, dtype=np.float32),
        "rtc_actions": np.ones((6, 4), dtype=np.float32),
        "rtc_prefix_length": 2,
        "rtc_guidance_weights": np.ones(6, dtype=np.float32),
        "rtc_num_steps": 5,
        "rtc_max_guidance_weight": 5.0,
    }


def test_rtc_mode_auto_follows_training_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)

    standard = _policy.Policy(_FakeModel(train_time_rtc=False), rtc_mode=_policy.RtcMode.AUTO)
    trained = _policy.Policy(_FakeModel(train_time_rtc=True), rtc_mode=_policy.RtcMode.AUTO)
    forced_inference = _policy.Policy(
        _FakeModel(train_time_rtc=True), rtc_mode=_policy.RtcMode.INFERENCE_TIME
    )

    assert standard.metadata["rtc_mode"] == "inference_time"
    assert trained.metadata["rtc_mode"] == "training_time"
    assert forced_inference.metadata["rtc_mode"] == "inference_time"


def test_training_time_mode_rejects_standard_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)

    with pytest.raises(ValueError, match="train_time_rtc=True"):
        _policy.Policy(_FakeModel(train_time_rtc=False), rtc_mode=_policy.RtcMode.TRAINING_TIME)


@pytest.mark.parametrize(
    ("train_time_rtc", "rtc_mode", "expected_key", "unexpected_key"),
    [
        (0, _policy.RtcMode.AUTO, "guidance_weights", "prefix_lengths"),
        (1, _policy.RtcMode.AUTO, "prefix_lengths", "guidance_weights"),
        (1, _policy.RtcMode.INFERENCE_TIME, "guidance_weights", "prefix_lengths"),
    ],
)
def test_rtc_request_routes_selected_sampler(
    monkeypatch: pytest.MonkeyPatch,
    train_time_rtc: int,
    rtc_mode: _policy.RtcMode,
    expected_key: str,
    unexpected_key: str,
) -> None:
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)
    model = _FakeModel(train_time_rtc=bool(train_time_rtc))
    policy = _policy.Policy(model, transforms=[_map_rtc_actions], rtc_mode=rtc_mode)

    policy.infer(_rtc_request())

    assert expected_key in model.last_sample_kwargs
    assert unexpected_key not in model.last_sample_kwargs


def test_training_time_model_plain_request_uses_standard_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda fn: fn)
    model = _FakeModel(train_time_rtc=True)
    policy = _policy.Policy(model, transforms=[_map_rtc_actions], rtc_mode=_policy.RtcMode.AUTO)
    request = _rtc_request()
    for key in (
        "rtc_actions",
        "rtc_prefix_length",
        "rtc_guidance_weights",
        "rtc_num_steps",
        "rtc_max_guidance_weight",
    ):
        request.pop(key)

    policy.infer(request)

    assert "guidance_actions" not in model.last_sample_kwargs
    assert "guidance_weights" not in model.last_sample_kwargs
    assert "prefix_lengths" not in model.last_sample_kwargs


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)

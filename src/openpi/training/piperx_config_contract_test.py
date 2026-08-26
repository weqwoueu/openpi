import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config


def test_pi05_piperx_plug_sft_contract():
    config = _config.get_config("pi05_piperx_plug_sft")

    assert config.model.pi05 is True
    assert config.model.pistar is False
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50
    assert config.num_workers == 16

    assert isinstance(config.data, _config.LeRobotPiperDataConfig)
    assert config.data.action_sequence_keys == ("action",)
    assert config.data.extra_delta_transform is False


def test_pistar_inference_transform_builds_unconditional_prompt():
    model_config = pi0_config.Pi0Config(pi05=True, pistar=True)
    transform_group = _config.ModelTransformFactory(adv_ind_dropout=False)(model_config)
    tokenize_transform = next(
        transform for transform in transform_group.inputs if isinstance(transform, transforms.TokenizePrompt)
    )

    assert tokenize_transform.adv_ind_input is True
    assert tokenize_transform.adv_ind_dropout is False
    assert tokenize_transform.adv_guidance_input is True


def test_pistar_training_transform_does_not_build_guidance_prompt():
    model_config = pi0_config.Pi0Config(pi05=True, pistar=True)
    transform_group = _config.ModelTransformFactory(adv_ind_dropout=True)(model_config)
    tokenize_transform = next(
        transform for transform in transform_group.inputs if isinstance(transform, transforms.TokenizePrompt)
    )

    assert tokenize_transform.adv_ind_input is True
    assert tokenize_transform.adv_ind_dropout is True
    assert tokenize_transform.adv_guidance_input is False


def test_pistar_inference_tokenizer_emits_conditional_and_unconditional_prompts():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def tokenize(self, prompt, state, adv_ind, *, adv_ind_dropout):
            self.calls.append((prompt, state, adv_ind, adv_ind_dropout))
            marker = 1 if adv_ind is not None else 0
            return np.array([marker], dtype=np.int32), np.array([True])

    tokenizer = FakeTokenizer()
    transform = transforms.TokenizePrompt(
        tokenizer,
        adv_ind_input=True,
        adv_ind_dropout=False,
        adv_guidance_input=True,
    )
    result = transform({"prompt": "insert the plug", "adv_ind": "positive"})

    assert tokenizer.calls == [
        ("insert the plug", None, "positive", False),
        ("insert the plug", None, None, False),
    ]
    np.testing.assert_array_equal(result["tokenized_prompt"], np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(result["tokenized_prompt_uncond"], np.array([0], dtype=np.int32))

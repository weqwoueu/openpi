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

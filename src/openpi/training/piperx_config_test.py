from openpi.models import pi0_config
from openpi.training import config as _config


def test_piperx_rtc_training_config_contract(tmp_path) -> None:
    config = _config.get_config("pi05_piperx_dagger_train_time_rtc")

    assert isinstance(config.model, pi0_config.Pi0Config)
    assert config.model.pi05
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 30
    assert config.model.train_time_rtc
    assert config.model.rtc_max_delay_steps == 6

    data_config = config.data.create(tmp_path, config.model)
    assert data_config.repo_id == "piperx/dagger/piperx_grab_bigbox_yellow_0706_0713"
    assert data_config.prompt_from_task
    assert data_config.action_sequence_keys == ("action",)
    assert len(data_config.repack_transforms.inputs) == 2
    assert len(data_config.data_transforms.inputs) == 2
    assert len(data_config.data_transforms.outputs) == 2

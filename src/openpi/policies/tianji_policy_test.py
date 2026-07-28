import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import tianji_policy


def test_tianji_inputs_accepts_rtc_actions() -> None:
    data = tianji_policy.make_tianji_example()
    data["rtc_actions"] = np.ones((50, tianji_policy.TIANJI_ACTION_DIM), dtype=np.float32)

    result = tianji_policy.TianjiInputs(_model.ModelType.PI05)(data)

    assert result["actions"].shape == (50, tianji_policy.TIANJI_ACTION_DIM)
    assert result["actions"].dtype == np.float32


def test_tianji_inputs_rejects_wrong_rtc_action_dim() -> None:
    data = tianji_policy.make_tianji_example()
    data["rtc_actions"] = np.zeros((50, 14), dtype=np.float32)

    with pytest.raises(ValueError, match="rtc_actions"):
        tianji_policy.TianjiInputs(_model.ModelType.PI05)(data)

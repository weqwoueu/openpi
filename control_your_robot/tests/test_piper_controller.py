import sys
from pathlib import Path
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Keep this controller test independent from the data stack and real Piper SDK.
data_handler_stub = ModuleType("robot.utils.base.data_handler")
data_handler_stub.debug_print = lambda *args, **kwargs: None
sys.modules["robot.utils.base.data_handler"] = data_handler_stub

piper_sdk_stub = ModuleType("piper_sdk")
piper_sdk_stub.C_PiperInterface_V2 = object
sys.modules["piper_sdk"] = piper_sdk_stub

from robot.controller.Piper_controller import PiperController


class FakePiper:
    def __init__(self):
        self.motion_calls = []
        self.joint_calls = []

    def MotionCtrl_2(self, *args):
        self.motion_calls.append(args)

    def JointCtrl(self, *args):
        self.joint_calls.append(args)


def make_controller(use_mit_mode):
    controller = PiperController("test_arm", use_mit_mode=use_mit_mode)
    controller.controller = FakePiper()
    return controller


def test_joint_control_uses_mit_mode_when_enabled():
    controller = make_controller(use_mit_mode=True)
    joint = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    controller.set_joint(joint, speed_percent=42)

    assert controller.controller.motion_calls == [(0x01, 0x01, 42, 0xAD)]
    assert controller.controller.joint_calls == [
        tuple(int(value * 57295.7795) for value in joint)
    ]


def test_joint_control_keeps_position_mode_by_default():
    controller = make_controller(use_mit_mode=False)

    controller.set_joint(np.zeros(6), speed_percent=37)

    assert controller.controller.motion_calls == [(0x01, 0x01, 37, 0x00)]
    assert controller.controller.joint_calls == [(0, 0, 0, 0, 0, 0)]

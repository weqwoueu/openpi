# ruff: noqa: SLF001

import importlib.util
import json
from pathlib import Path
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

SCRIPT_PATH = (
    PROJECT_ROOT
    / "example"
    / "collect"
    / "collect_lerobot_master_slave_teleop.py"
)


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


_IMPORT_STUBS = {
    "my_robot.piper_single_lerobot": _stub_module(
        "my_robot.piper_single_lerobot", PiperSingleLeRobot=object
    ),
    "robot.controller.Piper_controller": _stub_module(
        "robot.controller.Piper_controller", PiperController=object
    ),
    "robot.data.collect_lerobot_rl": _stub_module(
        "robot.data.collect_lerobot_rl", CollectLeRobotRL=object
    ),
    "robot.utils.worker.time_scheduler": _stub_module(
        "robot.utils.worker.time_scheduler", TimeScheduler=object
    ),
    "robot.utils.base.data_handler": _stub_module(
        "robot.utils.base.data_handler", debug_print=lambda *args, **kwargs: None
    ),
    "robot.utils.teleop_filter": _stub_module(
        "robot.utils.teleop_filter", EmaSlewFilter=object, FixedRateControlLoop=object
    ),
}
_PREVIOUS_MODULES = {name: sys.modules.get(name) for name in _IMPORT_STUBS}
sys.modules.update(_IMPORT_STUBS)
SPEC = importlib.util.spec_from_file_location("collect_lerobot_master_slave_teleop", SCRIPT_PATH)
RECORDER = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(RECORDER)
finally:
    for name, previous in _PREVIOUS_MODULES.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class FakeCollection:
    def __init__(self):
        self.saved = 0
        self.cleared = 0
        self.collected = []

    def save_episode(self, **kwargs):
        self.saved += 1

    def clear_current_episode(self):
        self.cleared += 1

    def collect(self, controllers, sensors, **kwargs):
        self.collected.append((controllers, sensors, kwargs))


def test_read_existing_episode_count_for_resume(tmp_path):
    assert RECORDER._read_existing_episode_count(tmp_path, "piperx/demo") == 0

    meta_dir = tmp_path / "piperx" / "demo" / "meta"
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(json.dumps({"total_episodes": 37}), encoding="utf-8")

    assert RECORDER._read_existing_episode_count(tmp_path, "piperx/demo") == 37


def test_exit_or_discard_never_saves_current_episode():
    for save_requested, discard_requested, exit_requested in (
        (False, False, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ):
        collection = FakeCollection()
        saved = RECORDER._complete_current_episode(
            collection,
            save_requested=save_requested,
            discard_requested=discard_requested,
            exit_requested=exit_requested,
        )
        assert not saved
        assert collection.saved == 0
        assert collection.cleared == 1


def test_explicit_finish_saves_episode():
    collection = FakeCollection()

    saved = RECORDER._complete_current_episode(
        collection,
        save_requested=True,
        discard_requested=False,
        exit_requested=False,
    )

    assert saved
    assert collection.saved == 1
    assert collection.cleared == 0


def test_first_synchronized_sample_can_be_consumed_without_recording():
    collection = FakeCollection()

    class FakeRobot:
        def __init__(self):
            self.collection = collection
            self.reads = 0

        def get(self):
            self.reads += 1
            return {"joint": self.reads}, {"image": self.reads}

    class FakeControlLoop:
        def get_latest(self):
            return {"joint": [0.0] * 6, "gripper": 0.0}

    robot = FakeRobot()
    control_loop = FakeControlLoop()

    assert RECORDER._collect_synchronized_sample(robot, control_loop, discard_sample=True)
    assert robot.reads == 1
    assert collection.collected == []

    assert RECORDER._collect_synchronized_sample(robot, control_loop, discard_sample=False)
    assert robot.reads == 2
    assert len(collection.collected) == 1

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

CONTROL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = CONTROL_ROOT / "example" / "collect" / "collect_lerobot_dagger_websocket.py"
SPEC = importlib.util.spec_from_file_location("collect_lerobot_dagger_websocket", SCRIPT_PATH)
DAGGER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DAGGER
SPEC.loader.exec_module(DAGGER)


def test_key_decoder_handles_split_arrow_sequences():
    decoder = DAGGER.KeyDecoder()

    assert decoder.feed(b"\r ") == [DAGGER.KeyEvent.ENTER, DAGGER.KeyEvent.SPACE]
    assert decoder.feed(b"\x1b") == []
    assert decoder.feed(b"[") == []
    assert decoder.feed(b"C") == [DAGGER.KeyEvent.RIGHT]
    assert decoder.feed(b"\x1b[D") == [DAGGER.KeyEvent.LEFT]
    assert decoder.feed(b"x") == []


def test_space_switches_to_expert_only_once_per_episode():
    session = DAGGER.DaggerSession(num_episodes=-1)

    assert session.handle(DAGGER.KeyEvent.ENTER) is DAGGER.SessionCommand.START_EPISODE
    assert session.state is DAGGER.SessionState.AUTONOMOUS
    assert session.generation == 1

    assert session.handle(DAGGER.KeyEvent.SPACE) is DAGGER.SessionCommand.START_TAKEOVER
    assert session.state is DAGGER.SessionState.SWITCHING_TO_EXPERT
    assert session.generation == 2
    assert session.handle(DAGGER.KeyEvent.SPACE) is DAGGER.SessionCommand.NONE

    session.takeover_ready()
    assert session.state is DAGGER.SessionState.INTERVENTION
    assert session.handle(DAGGER.KeyEvent.SPACE) is DAGGER.SessionCommand.NONE
    assert session.state is DAGGER.SessionState.INTERVENTION


def test_failed_takeover_returns_to_autonomous_and_allows_retry():
    session = DAGGER.DaggerSession(num_episodes=-1)
    session.handle(DAGGER.KeyEvent.ENTER)
    session.handle(DAGGER.KeyEvent.SPACE)

    session.takeover_failed()

    assert session.state is DAGGER.SessionState.AUTONOMOUS
    assert not session.intervention_used
    assert session.handle(DAGGER.KeyEvent.SPACE) is DAGGER.SessionCommand.START_TAKEOVER
    assert session.handle(DAGGER.KeyEvent.SPACE) is DAGGER.SessionCommand.NONE
    session.takeover_ready()
    assert session.state is DAGGER.SessionState.INTERVENTION


def test_label_requires_arrow_then_enter_and_save_ignores_keys():
    session = DAGGER.DaggerSession(num_episodes=2)
    session.handle(DAGGER.KeyEvent.ENTER)
    assert session.handle(DAGGER.KeyEvent.ENTER) is DAGGER.SessionCommand.STOP_EPISODE
    assert session.state is DAGGER.SessionState.AWAIT_LABEL
    assert session.generation == 2

    assert session.handle(DAGGER.KeyEvent.ENTER) is DAGGER.SessionCommand.NONE
    assert session.handle(DAGGER.KeyEvent.LEFT) is DAGGER.SessionCommand.NONE
    assert session.label is False
    assert session.handle(DAGGER.KeyEvent.ENTER) is DAGGER.SessionCommand.SAVE_EPISODE
    assert session.state is DAGGER.SessionState.SAVING
    assert session.handle(DAGGER.KeyEvent.RIGHT) is DAGGER.SessionCommand.NONE

    session.complete_save()
    assert session.state is DAGGER.SessionState.READY_NEXT
    assert session.handle(DAGGER.KeyEvent.ENTER) is DAGGER.SessionCommand.START_EPISODE
    assert session.state is DAGGER.SessionState.AUTONOMOUS
    assert not session.intervention_used


class FakeInferenceWorker:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []

    def submit(self, request):
        self.requests.append(request)

    def drain_responses(self):
        responses, self.responses = self.responses, []
        return responses


class FakeCollector:
    def __init__(self):
        self.collected = []
        self.cleared = 0

    def collect(self, controllers, sensors, **kwargs):
        self.collected.append((controllers, sensors, kwargs))

    def clear_current_episode(self):
        self.cleared += 1


class FakeSavingCollector:
    def __init__(self, frame_count=1, intervention_flags=None, episode_index=0):
        self.calls = []
        self.episode_buffer = [object()] * frame_count
        self.intervention_flags = list(
            intervention_flags if intervention_flags is not None else [0] * frame_count
        )
        self.dataset = types.SimpleNamespace(
            episode_buffer={"episode_index": episode_index},
            num_episodes=episode_index,
        )

    def save_episode(self, **kwargs):
        self.calls.append(kwargs)
        self.episode_buffer = []
        self.intervention_flags = []


class FakeRobot:
    def __init__(self):
        self.sent = []

    def move_follower(self, move_data, bypass_policy=False):
        requested = np.concatenate(
            [np.asarray(move_data["joint"], dtype=np.float32), [move_data["gripper"]]]
        )
        sent = requested + np.float32(0.125)
        self.sent.append((sent, bypass_policy))
        return sent

    def get(self):
        arm_state = {"joint": np.zeros(6), "gripper": 0.0}
        return [
            {"left_arm": arm_state, "right_arm": arm_state},
            {"cam_head": {"color": 1}, "cam_wrist": {"color": 2}},
        ]


def _make_runtime(worker=None, robot=None, collector=None, **runtime_kwargs):
    kwargs = dict(
        robot=robot or FakeRobot(),
        collector=collector or FakeCollector(),
        inference_worker=worker or FakeInferenceWorker(),
        prompt="task",
        inference_adv_ind=None,
        sample_fps=30,
        teleop_fps=60,
        chunk_size=3,
        reset_settle_seconds=0,
        takeover_align_speed=60,
        teleop_mapping=DAGGER.TeleopMapping(
            joint_sign=[1] * 6,
            joint_offset=[0] * 6,
            joint_scale=1,
            gripper_scale=1,
            gripper_offset=0,
        ),
        filter_kwargs={},
    )
    kwargs.update(runtime_kwargs)
    return DAGGER.DaggerRuntime(**kwargs)


def test_stale_inference_response_is_discarded_by_generation():
    stale_actions = np.ones((3, 7), dtype=np.float32)
    current_actions = np.full((3, 7), 2.0, dtype=np.float32)
    worker = FakeInferenceWorker(
        [
            DAGGER.InferenceResponse(1, result={"actions": stale_actions}),
            DAGGER.InferenceResponse(2, result={"actions": current_actions}),
        ]
    )
    runtime = _make_runtime(worker=worker)
    runtime.generation = 2
    runtime._pending_generation = 2

    runtime._drain_policy_responses()

    assert len(runtime._actions) == 3
    np.testing.assert_array_equal(np.asarray(runtime._actions), current_actions)


def test_prefetched_chunk_skips_actions_for_old_queue_lead():
    old_tail = np.full((2, 7), -1.0, dtype=np.float32)
    new_chunk = np.repeat(np.arange(5, dtype=np.float32)[:, None], 7, axis=1)
    worker = FakeInferenceWorker(
        [DAGGER.InferenceResponse(3, lead_steps=2, result={"actions": new_chunk})]
    )
    runtime = _make_runtime(worker=worker, prefetch_threshold=2, chunk_size=5)
    runtime.generation = 3
    runtime._pending_generation = 3
    runtime._actions.extend(old_tail)

    runtime._drain_policy_responses()

    expected = np.concatenate([old_tail, new_chunk[2:]], axis=0)
    np.testing.assert_array_equal(np.asarray(runtime._actions), expected)


def test_prefetch_request_records_remaining_action_count(monkeypatch):
    worker = FakeInferenceWorker()
    runtime = _make_runtime(worker=worker, prefetch_threshold=1)
    runtime.generation = 4
    runtime._actions.extend(np.zeros((2, 7), dtype=np.float32))
    runtime._next_sample_at = 0
    monkeypatch.setattr(DAGGER, "make_policy_observation", lambda *args, **kwargs: {"obs": 1})

    assert runtime.tick_autonomous()

    assert len(worker.requests) == 1
    assert worker.requests[0].lead_steps == 1


def test_waiting_for_inference_does_not_resend_or_collect():
    robot = FakeRobot()
    collector = FakeCollector()
    runtime = _make_runtime(robot=robot, collector=collector)
    runtime.generation = 5
    runtime._pending_generation = 5
    runtime._last_policy_action = np.ones(7, dtype=np.float32)
    runtime._next_sample_at = 0

    assert not runtime.tick_autonomous()

    assert robot.sent == []
    assert collector.collected == []
    assert runtime.step_count == 0


def test_master_feedback_action_does_not_depend_on_joint_ctrl_frames():
    expected_joint = np.linspace(-0.3, 0.3, 6)

    class FeedbackMaster:
        controller = types.SimpleNamespace(
            GetArmJointCtrl=lambda: (_ for _ in ()).throw(
                AssertionError("takeover must not read JointCtrl")
            )
        )

        @staticmethod
        def get_state():
            return {"joint": expected_joint, "gripper": 0.4}

    action = DAGGER._read_master_feedback_action(FeedbackMaster())

    assert action is not None
    np.testing.assert_allclose(action[0], expected_joint)
    assert action[1:] == (0.4, "feedback")


def test_runtime_takeover_uses_master_feedback_for_two_cycles(monkeypatch):
    from robot.utils import teleop_filter

    class DeterministicLoop:
        def __init__(self, _fps, step):
            self.step = step

        def start(self):
            pass

        def stop(self, timeout=None):
            return True

        def get_latest(self):
            return None

    class FeedbackMaster:
        def __init__(self):
            self.state = {"joint": np.zeros(6), "gripper": 0.0}
            self.controller = types.SimpleNamespace(
                GetArmJointCtrl=lambda: (_ for _ in ()).throw(
                    AssertionError("runtime takeover must not read JointCtrl")
                )
            )

        def get_state(self):
            return {
                "joint": np.asarray(self.state["joint"], dtype=float).copy(),
                "gripper": float(self.state["gripper"]),
            }

    class FeedbackTakeoverRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.master = FeedbackMaster()
            self.follower_state = {"joint": np.zeros(6), "gripper": 0.0}
            self.controllers = {
                "arm": {
                    "left_arm": types.SimpleNamespace(controller=object()),
                    "right_arm": self.master,
                }
            }
            self.transitions = []

        def set_policy_enabled(self, enabled):
            self.transitions.append(("policy", bool(enabled)))

        def get_follower_state(self):
            return {
                "joint": np.asarray(self.follower_state["joint"], dtype=float).copy(),
                "gripper": float(self.follower_state["gripper"]),
            }

        def hold_follower_position(self):
            self.transitions.append(("hold",))

        def align_master_to_follower(self, follower_state, *, speed):
            self.transitions.append(("align", int(speed)))
            self.master.state = {
                "joint": np.asarray(follower_state["joint"], dtype=float).copy(),
                "gripper": float(follower_state["gripper"]),
            }

        def enable_master_drag_mode(self):
            self.transitions.append(("role", 0xFA))

        def disable_master_drag_mode(self):
            self.transitions.append(("role", 0xFC))

    monkeypatch.setattr(teleop_filter, "FixedRateControlLoop", DeterministicLoop)
    robot = FeedbackTakeoverRobot()
    runtime = _make_runtime(robot=robot, takeover_align_speed=60)

    for generation, joint in enumerate((np.full(6, 0.1), np.full(6, -0.2)), start=1):
        robot.follower_state = {"joint": joint, "gripper": generation / 10}
        runtime.begin_takeover(generation)
        assert runtime.teleop_mapping.source == "feedback"
        np.testing.assert_allclose(robot.master.state["joint"], joint)
        runtime.stop_episode(generation + 10)

    assert [event for event in robot.transitions if event[0] == "align"] == [
        ("align", 60),
        ("align", 60),
    ]
    assert [event for event in robot.transitions if event[0] == "role"] == [
        ("role", 0xFA),
        ("role", 0xFC),
        ("role", 0xFA),
        ("role", 0xFC),
    ]


def test_takeover_only_wraps_expected_transition_timeouts():
    class TransitionRobot(FakeRobot):
        def __init__(self, alignment_error):
            super().__init__()
            self.alignment_error = alignment_error
            self.controllers = {
                "arm": {
                    "left_arm": types.SimpleNamespace(controller=object()),
                    "right_arm": types.SimpleNamespace(controller=object()),
                }
            }

        def set_policy_enabled(self, _enabled):
            pass

        def get_master_state(self):
            return {"joint": np.zeros(6), "gripper": 0.0}

        def get_follower_state(self):
            return {"joint": np.zeros(6), "gripper": 0.0}

        def hold_follower_position(self):
            pass

        def align_master_to_follower(self, *_args, **_kwargs):
            raise self.alignment_error

    timeout_runtime = _make_runtime(
        robot=TransitionRobot(TimeoutError("alignment timed out")),
    )
    with pytest.raises(DAGGER.TakeoverTransitionError, match="alignment timed out"):
        timeout_runtime.begin_takeover(1)

    can_runtime = _make_runtime(
        robot=TransitionRobot(RuntimeError("master CAN send failed")),
    )
    with pytest.raises(RuntimeError, match="master CAN send failed") as error:
        can_runtime.begin_takeover(1)
    assert not isinstance(error.value, DAGGER.TakeoverTransitionError)


def test_failed_takeover_recovery_preserves_episode_and_restarts_policy(monkeypatch):
    class RecoveryRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.policy_enabled = False
            self.holds = 0
            self.restores = 0
            self.controllers = {
                "arm": {
                    "left_arm": types.SimpleNamespace(controller=object()),
                    "right_arm": types.SimpleNamespace(controller=object()),
                }
            }

        def set_policy_enabled(self, enabled):
            self.policy_enabled = bool(enabled)

        def hold_follower_position(self):
            self.holds += 1

        def disable_master_drag_mode(self):
            self.restores += 1

    robot = RecoveryRobot()
    worker = FakeInferenceWorker()
    collector = FakeCollector()
    runtime = _make_runtime(robot=robot, worker=worker, collector=collector)
    runtime.generation = 4
    runtime.step_count = 123
    runtime._master_drag_enabled = True
    runtime._actions.append(np.ones(7))
    runtime._teleop_errors.put(RuntimeError("stale teleop error"))
    monkeypatch.setattr(
        DAGGER,
        "make_policy_observation",
        lambda *args, **kwargs: {"observation": "current"},
    )

    runtime.recover_failed_takeover(4)

    assert robot.restores == 1
    assert robot.policy_enabled
    assert runtime.step_count == 123
    assert collector.cleared == 0
    assert not runtime._actions
    assert runtime._teleop_errors.empty()
    assert len(worker.requests) == 1
    assert worker.requests[0].generation == 4


def test_reset_runs_existing_two_arm_init_script(tmp_path, monkeypatch):
    reset_script = tmp_path / "2_arm_go_init.sh"
    reset_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        DAGGER.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )
    runtime = _make_runtime(reset_script=reset_script)

    runtime._run_reset_script()

    assert calls == [(["bash", str(reset_script)], True)]


def test_next_episode_clears_previous_teleop_error(tmp_path, monkeypatch):
    class EpisodeRobot(FakeRobot):
        def __init__(self):
            super().__init__()
            self.controllers = {
                "arm": {"left_arm": types.SimpleNamespace(controller=object())}
            }

        def set_policy_enabled(self, _enabled):
            pass

        def hold_follower_position(self):
            pass

    reset_script = tmp_path / "2_arm_go_init.sh"
    reset_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(DAGGER.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        DAGGER,
        "make_policy_observation",
        lambda *_args, **_kwargs: {"observation": "current"},
    )
    runtime = _make_runtime(robot=EpisodeRobot(), reset_script=reset_script)
    runtime._teleop_errors.put(RuntimeError("previous episode"))

    runtime.start_episode(2)

    assert runtime._teleop_errors.empty()
    assert runtime.generation == 2


def test_raw_dagger_save_keeps_advantage_unlabeled():
    for success in (True, False):
        collector = FakeSavingCollector()
        assert DAGGER.save_dagger_episode(collector, success)
        assert collector.calls == [{"success": success, "adv_ind_value": "none"}]

    empty = FakeSavingCollector(frame_count=0)
    assert not DAGGER.save_dagger_episode(empty, True)
    assert empty.calls == []


@pytest.mark.parametrize(
    ("flags", "success", "episode_type", "policy_frames", "intervention_frames"),
    [
        ([0, 0], True, "autonomous_success", 2, 0),
        ([0, 0], False, "autonomous_failure", 2, 0),
        ([0, 1], True, "intervention_success", 1, 1),
        ([0, 1], False, "intervention_failure", 1, 1),
        ([1, 1], True, "intervention_success", 0, 2),
    ],
)
def test_dagger_episode_classification(
    flags,
    success,
    episode_type,
    policy_frames,
    intervention_frames,
):
    summary = DAGGER.classify_dagger_episode(flags, success, episode_index=7)

    assert summary.episode_index == 7
    assert summary.episode_type == episode_type
    assert summary.total_frames == len(flags)
    assert summary.policy_frames == policy_frames
    assert summary.intervention_frames == intervention_frames


def test_dagger_tracker_records_only_after_dataset_save(tmp_path):
    tracker = DAGGER.DaggerEpisodeTracker(
        tmp_path,
        targets={
            "autonomous_success": 1,
            "autonomous_failure": -1,
            "intervention_success": 1,
            "intervention_failure": -1,
        },
    )
    collector = FakeSavingCollector(
        frame_count=3,
        intervention_flags=[0, 0, 1],
        episode_index=37,
    )

    summary = DAGGER.save_dagger_episode(collector, True, tracker=tracker)

    assert summary.episode_index == 37
    assert summary.episode_type == "intervention_success"
    assert tracker.counts == {"intervention_success": 1}
    assert tracker.summary_path.exists()
    restored = DAGGER.DaggerEpisodeTracker(tmp_path, targets=tracker.targets)
    assert restored.records == (summary,)
    assert restored.counts == tracker.counts


def test_failed_dataset_save_does_not_update_dagger_summary(tmp_path):
    class FailingCollector(FakeSavingCollector):
        def save_episode(self, **kwargs):
            raise RuntimeError("save failed")

    tracker = DAGGER.DaggerEpisodeTracker(tmp_path)
    collector = FailingCollector(frame_count=2, intervention_flags=[0, 1])

    with pytest.raises(RuntimeError, match="save failed"):
        DAGGER.save_dagger_episode(collector, False, tracker=tracker)

    assert tracker.records == ()
    assert not tracker.summary_path.exists()


def test_dagger_tracker_backfills_existing_parquet(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    parquet.write_table(
        pyarrow.table(
            {
                "episode_index": [0, 0],
                "intervention": [[0], [0]],
                "reward": [[0.0], [1.0]],
            }
        ),
        data_dir / "episode_000000.parquet",
    )
    parquet.write_table(
        pyarrow.table(
            {
                "episode_index": [1, 1, 1],
                "intervention": [[0], [0], [1]],
                "reward": [[0.0], [0.0], [0.0]],
            }
        ),
        data_dir / "episode_000001.parquet",
    )

    tracker = DAGGER.DaggerEpisodeTracker(tmp_path)

    assert [record.episode_type for record in tracker.records] == [
        "autonomous_success",
        "intervention_failure",
    ]
    assert tracker.counts == {
        "autonomous_success": 1,
        "intervention_failure": 1,
    }
    assert tracker.summary_path.exists()


def test_dagger_target_progress_is_advisory(tmp_path):
    tracker = DAGGER.DaggerEpisodeTracker(
        tmp_path,
        targets={
            "autonomous_success": 1,
            "autonomous_failure": -1,
            "intervention_success": 1,
            "intervention_failure": -1,
        },
    )
    tracker.record(DAGGER.classify_dagger_episode([0], True, 0))
    tracker.record(DAGGER.classify_dagger_episode([0, 1], True, 1))

    progress = tracker.format_progress()
    assert "policy success       1/1" in progress
    assert "takeover success     1/1" in progress
    assert tracker.configured_targets_met()

    tracker.record(DAGGER.classify_dagger_episode([0], False, 2))
    assert tracker.counts["autonomous_failure"] == 1


def test_policy_frame_records_returned_sent_action_and_follower_only(monkeypatch):
    robot = FakeRobot()
    collector = FakeCollector()
    runtime = _make_runtime(robot=robot, collector=collector)
    requested = np.arange(7, dtype=np.float32)
    runtime.generation = 4
    runtime._actions.append(requested)
    runtime._next_sample_at = 0
    monkeypatch.setattr(DAGGER, "make_policy_observation", lambda *args, **kwargs: {"obs": 1})

    assert runtime.tick_autonomous()

    controllers, _sensors, kwargs = collector.collected[0]
    assert list(controllers) == ["left_arm"]
    np.testing.assert_array_equal(kwargs["action_data"], requested + np.float32(0.125))
    assert kwargs["is_intervention"] is False


def test_teleop_mapping_starts_at_follower_and_preserves_master_delta():
    mapping = DAGGER.TeleopMapping(
        joint_sign=[1] * 6,
        joint_offset=[0] * 6,
        joint_scale=1,
        gripper_scale=1,
        gripper_offset=0,
    )
    master_joint = np.linspace(0.0, 0.5, 6)
    follower_joint = np.linspace(1.0, 1.5, 6)
    mapping.align(
        (master_joint, 0.2, "ctrl"),
        {"joint": follower_joint, "gripper": 0.6},
    )

    aligned = mapping.transform((master_joint, 0.2, "ctrl"))
    moved = mapping.transform((master_joint + 0.1, 0.3, "ctrl"))

    np.testing.assert_allclose(aligned["joint"], follower_joint)
    assert aligned["gripper"] == 0.6
    np.testing.assert_allclose(moved["joint"], follower_joint + 0.1)
    assert moved["gripper"] == 0.7


class FakeSdk:
    def __init__(self):
        self.events = []
        self.mode_flags = []
        self.motion_ctrl_2_calls = []
        self.motion_ctrl_1_calls = []
        self.master_slave_calls = []
        self.teaching_param_calls = []
        self.joint_ctrl_timestamp = 12.5
        self.status_timestamp = 20.0
        self.feedback_joint = np.zeros(6, dtype=float)
        self.feedback_gripper = 0.0
        self.status = types.SimpleNamespace(
            ctrl_mode=0x01,
            teach_status=0x00,
            arm_status=0x00,
            motion_status=0x00,
            mode_feed=0x01,
        )

    def MotionCtrl_1(self, *args):
        self.events.append(("motion1", args))
        self.motion_ctrl_1_calls.append(args)
        if args[2] == 0x01:
            if self.status.ctrl_mode != 0x06:
                self.status.ctrl_mode = 0x02
            self.status.teach_status = 0x01
        elif args[2] == 0x02:
            self.status.teach_status = 0x02
        elif args[2] == 0x00:
            self.status.teach_status = 0x00

    def MotionCtrl_2(self, ctrl_mode, move_mode, speed, mode_flag):
        self.events.append(("motion2", ctrl_mode))
        self.motion_ctrl_2_calls.append((ctrl_mode, move_mode, speed, mode_flag))
        self.mode_flags.append(mode_flag)
        self.status.ctrl_mode = ctrl_mode
        self.status.motion_status = 0x00

    def MasterSlaveConfig(self, *args):
        self.events.append(("role", args[0]))
        self.master_slave_calls.append(args)
        if args[0] == 0xFA:
            self.status.ctrl_mode = 0x06
        elif args[0] == 0xFC:
            self.status.ctrl_mode = 0x01

    def GripperTeachingPendantParamConfig(self, **kwargs):
        self.teaching_param_calls.append(kwargs)

    def GetArmStatus(self):
        self.status_timestamp += 0.1
        return types.SimpleNamespace(
            time_stamp=self.status_timestamp,
            arm_status=self.status,
        )

    def GetArmJointCtrl(self):
        joint_cmd = np.rint(self.feedback_joint * 57295.7795).astype(int)
        return types.SimpleNamespace(
            Hz=0.0,
            time_stamp=self.joint_ctrl_timestamp,
            joint_ctrl=types.SimpleNamespace(
                joint_1=int(joint_cmd[0]),
                joint_2=int(joint_cmd[1]),
                joint_3=int(joint_cmd[2]),
                joint_4=int(joint_cmd[3]),
                joint_5=int(joint_cmd[4]),
                joint_6=int(joint_cmd[5]),
            ),
        )

    def GetArmJointMsgs(self):
        joint = np.rint(self.feedback_joint * 57295.7795).astype(int)
        return types.SimpleNamespace(
            joint_state=types.SimpleNamespace(
                joint_1=int(joint[0]),
                joint_2=int(joint[1]),
                joint_3=int(joint[2]),
                joint_4=int(joint[3]),
                joint_5=int(joint[4]),
                joint_6=int(joint[5]),
            )
        )

    def GetArmGripperMsgs(self):
        return types.SimpleNamespace(
            gripper_state=types.SimpleNamespace(
                grippers_angle=int(round(self.feedback_gripper * 70 * 1000))
            )
        )

    def GetArmGripperCtrl(self):
        return types.SimpleNamespace(
            Hz=0.0,
            gripper_ctrl=types.SimpleNamespace(
                grippers_angle=int(round(self.feedback_gripper * 70 * 1000))
            ),
        )

    def JointCtrl(self, *joints):
        self.events.append(("joint", tuple(joints)))
        self.feedback_joint = np.asarray(joints, dtype=float) / 57295.7795

    def GripperCtrl(self, angle, *args):
        self.events.append(("gripper", (angle, *args)))
        self.feedback_gripper = float(angle) / (70 * 1000)

    def EnableArm(self, _motor_num):
        pass


class FakeCanStatus:
    SEND_MESSAGE_SUCCESS = 1
    SEND_MESSAGE_FAILED = 2


class FakeCanBus:
    CAN_STATUS = FakeCanStatus

    def __init__(self):
        self.next_status = self.CAN_STATUS.SEND_MESSAGE_SUCCESS

    def SendCanMessage(self, *_args, **_kwargs):
        return self.next_status


class FakeControllerWrapper:
    def __init__(self):
        self.controller = FakeSdk()

    def get_state(self):
        return {
            "joint": self.controller.feedback_joint.copy(),
            "gripper": float(self.controller.feedback_gripper),
        }


def _load_piper_dagger_module():
    class StubRobot:
        pass

    stubs = {
        "my_robot.base_robot": types.SimpleNamespace(Robot=StubRobot),
        "my_robot.camera_config": types.SimpleNamespace(
            get_piper_camera_serials=lambda _profile: {}
        ),
        "robot.controller.Piper_controller": types.SimpleNamespace(PiperController=object),
        "robot.sensor.Realsense_sensor": types.SimpleNamespace(RealsenseSensor=object),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    piper_spec = importlib.util.spec_from_file_location(
        "piper_dagger_test_module", CONTROL_ROOT / "my_robot" / "piper_dagger.py"
    )
    piper_module = importlib.util.module_from_spec(piper_spec)
    try:
        piper_spec.loader.exec_module(piper_module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return piper_module


def _make_piper_dagger_robot(piper_module):
    robot = piper_module.PiperDAgger.__new__(piper_module.PiperDAgger)
    robot._policy_enabled = True
    robot._follower_gripper_effort = 5000
    robot._master_gripper_effort = 1000
    robot.follower_use_mit_mode = True
    follower = FakeControllerWrapper()
    master = FakeControllerWrapper()
    robot.controllers = {"arm": {"left_arm": follower, "right_arm": master}}
    robot.reassert_follower_hold = lambda: None
    return robot, follower, master


def test_piper_dagger_autonomous_moves_only_follower_in_mit():
    piper_module = _load_piper_dagger_module()
    robot, follower, master = _make_piper_dagger_robot(piper_module)
    command = {"joint": np.arange(6, dtype=float) / 10, "gripper": 0.5}
    initial_master_state = master.get_state()

    sent = robot.move_follower(command)

    assert master.controller.events == []
    np.testing.assert_allclose(master.get_state()["joint"], initial_master_state["joint"])
    assert master.get_state()["gripper"] == initial_master_state["gripper"]
    assert follower.controller.mode_flags == [0xAD]
    np.testing.assert_allclose(
        sent,
        np.concatenate([np.arange(6, dtype=np.float32) / 10, [0.5]]),
    )


def test_piper_dagger_alignment_ramp_is_bounded(monkeypatch):
    piper_module = _load_piper_dagger_module()
    robot, _follower, master = _make_piper_dagger_robot(piper_module)
    monkeypatch.setattr(piper_module.time, "sleep", lambda _seconds: None)
    follower_target = {
        "joint": np.array([0.05, -0.04, 0.03, -0.02, 0.01, 0.0]),
        "gripper": 0.08,
    }

    robot.align_master_to_follower(
        follower_target,
        speed=60,
    )

    joint_commands = [
        np.asarray(event[1], dtype=float) / 57295.7795
        for event in master.controller.events
        if event[0] == "joint"
    ]
    command_steps = np.diff(np.vstack([np.zeros(6), joint_commands]), axis=0)
    assert np.max(np.abs(command_steps)) <= 2501 / 57295.7795
    np.testing.assert_allclose(joint_commands[-1], follower_target["joint"], atol=2e-5)
    assert master.controller.feedback_gripper == pytest.approx(
        follower_target["gripper"], abs=2e-5
    )
    assert master.controller.master_slave_calls == [
        (piper_module.FOLLOWER_ROLE, 0x00, 0x00, 0x00)
    ]
    assert master.controller.mode_flags
    assert set(master.controller.mode_flags) == {0x00}

    master.controller.events.clear()
    master.controller.JointCtrl = lambda *joints: master.controller.events.append(
        ("joint", tuple(joints))
    )
    with pytest.raises(TimeoutError, match="after 3 ramp attempts"):
        robot.align_master_to_follower(
            {"joint": np.ones(6), "gripper": 1.0},
            speed=60,
        )
    assert len([event for event in master.controller.events if event[0] == "joint"]) > 1


def test_piper_dagger_feedback_takeover_repeats_for_two_episodes(monkeypatch):
    piper_module = _load_piper_dagger_module()
    robot, follower, master = _make_piper_dagger_robot(piper_module)
    monkeypatch.setattr(piper_module.time, "sleep", lambda _seconds: None)
    fixed_ctrl_timestamp = master.controller.joint_ctrl_timestamp
    episode_targets = (
        np.array([0.05, -0.04, 0.03, -0.02, 0.01, 0.0]),
        np.array([-0.03, 0.06, -0.02, 0.04, -0.01, 0.02]),
    )

    for episode_index, target_joint in enumerate(episode_targets):
        # The reset script puts the master at init. Autonomous policy movement
        # must affect only the follower until Space starts takeover.
        master.controller.feedback_joint = np.zeros(6, dtype=float)
        master.controller.feedback_gripper = 0.0
        master.controller.events.clear()
        master.controller.mode_flags.clear()
        master.controller.motion_ctrl_1_calls.clear()
        master.controller.motion_ctrl_2_calls.clear()
        master.controller.master_slave_calls.clear()

        policy_action = {
            "joint": target_joint,
            "gripper": 0.2 + 0.1 * episode_index,
        }
        robot.set_policy_enabled(True)
        robot.move_follower(policy_action)
        assert master.controller.events == []
        np.testing.assert_allclose(master.get_state()["joint"], np.zeros(6))

        frozen_follower = follower.get_state()
        robot.set_policy_enabled(False)
        robot.align_master_to_follower(
            frozen_follower,
            speed=60,
        )
        robot.enable_master_drag_mode()

        np.testing.assert_allclose(
            master.get_state()["joint"], frozen_follower["joint"], atol=2e-5
        )
        assert master.controller.master_slave_calls == [
            (piper_module.FOLLOWER_ROLE, 0x00, 0x00, 0x00),
            (piper_module.MASTER_ROLE, 0x00, 0x00, 0x00),
        ]
        assert master.controller.motion_ctrl_1_calls == []
        assert master.controller.motion_ctrl_2_calls
        assert all(call[0] == 0x01 and call[3] == 0x00 for call in master.controller.motion_ctrl_2_calls)
        assert master.controller.joint_ctrl_timestamp == fixed_ctrl_timestamp

        baseline = DAGGER._read_master_feedback_action(master)
        assert baseline is not None and baseline[2] == "feedback"
        mapping = DAGGER.TeleopMapping(
            joint_sign=[1] * 6,
            joint_offset=[0] * 6,
            joint_scale=1,
            gripper_scale=1,
            gripper_offset=0,
        )
        mapping.align(baseline, frozen_follower)

        master.controller.feedback_joint = target_joint + 0.02
        master.controller.feedback_gripper = policy_action["gripper"] + 0.05
        dragged = DAGGER._read_master_feedback_action(master)
        sent = robot.move_follower(mapping.transform(dragged), bypass_policy=True)
        np.testing.assert_allclose(sent[:6], target_joint + 0.02, atol=2e-5)
        assert sent[6] == pytest.approx(policy_action["gripper"] + 0.05, abs=2e-5)
        assert follower.controller.mode_flags[-1] == 0xAD

        robot.disable_master_drag_mode()
        assert master.controller.master_slave_calls[-1] == (
            piper_module.FOLLOWER_ROLE,
            0x00,
            0x00,
            0x00,
        )
        assert master.controller.motion_ctrl_2_calls[-1][0] == 0x01
        assert master.controller.motion_ctrl_2_calls[-1][3] == 0x00
        assert master.controller.joint_ctrl_timestamp == fixed_ctrl_timestamp


def test_checked_can_send_raises_on_failure():
    piper_module = _load_piper_dagger_module()
    PiperDAgger = piper_module.PiperDAgger

    can_bus = FakeCanBus()

    class CheckedSdk:
        def GetCanBus(self):
            return can_bus

    PiperDAgger._raise_on_can_send_failure(CheckedSdk(), "follower")
    assert can_bus.SendCanMessage(0x155, b"data") == FakeCanStatus.SEND_MESSAGE_SUCCESS
    can_bus.next_status = FakeCanStatus.SEND_MESSAGE_FAILED
    with pytest.raises(RuntimeError, match="follower CAN send failed"):
        can_bus.SendCanMessage(0x155, b"data")

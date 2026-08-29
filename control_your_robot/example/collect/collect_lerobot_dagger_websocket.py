"""Collect PiperX policy rollouts with one-way expert intervention over WebSocket."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
from dataclasses import dataclass
from enum import Enum
from enum import auto
import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROL_ROOT = PROJECT_ROOT / "control_your_robot"
sys.path.insert(0, str(CONTROL_ROOT))
sys.path.insert(0, str(CONTROL_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "openpi-client" / "src"))

PIPER_ACTION_DIM = 7

PRIMARY_EPISODE_CATEGORIES = (
    "autonomous_success",
    "autonomous_failure",
    "intervention_success",
    "intervention_failure",
)
CATEGORY_LABELS = {
    "autonomous_success": "policy success",
    "autonomous_failure": "policy failure",
    "intervention_success": "takeover success",
    "intervention_failure": "takeover failure",
}


class KeyEvent(Enum):
    ENTER = auto()
    SPACE = auto()
    RIGHT = auto()
    LEFT = auto()


class SessionState(Enum):
    READY = auto()
    AUTONOMOUS = auto()
    SWITCHING_TO_EXPERT = auto()
    INTERVENTION = auto()
    AWAIT_LABEL = auto()
    SAVING = auto()
    READY_NEXT = auto()
    COMPLETE = auto()


class SessionCommand(Enum):
    NONE = auto()
    START_EPISODE = auto()
    START_TAKEOVER = auto()
    STOP_EPISODE = auto()
    SAVE_EPISODE = auto()


class KeyDecoder:
    """Decode Enter, Space, and split ANSI left/right arrow sequences."""

    _EXACT_ARROWS = {
        b"\x1b[C": KeyEvent.RIGHT,
        b"\x1b[D": KeyEvent.LEFT,
    }
    _ARROW_PREFIXES = (b"\x1b", b"\x1b[")

    def __init__(self):
        self._escape_buffer = bytearray()

    def feed(self, data: bytes | str) -> list[KeyEvent]:
        if isinstance(data, str):
            data = data.encode()
        events = []
        for byte in data:
            events.extend(self._feed_byte(byte))
        return events

    def _feed_byte(self, byte: int) -> list[KeyEvent]:
        if self._escape_buffer:
            self._escape_buffer.append(byte)
            sequence = bytes(self._escape_buffer)
            event = self._EXACT_ARROWS.get(sequence)
            if event is not None:
                self._escape_buffer.clear()
                return [event]
            if sequence in self._ARROW_PREFIXES:
                return []
            self._escape_buffer.clear()
            return self._feed_byte(byte)

        if byte == 0x1B:
            self._escape_buffer.append(byte)
            return []
        if byte in (ord("\n"), ord("\r")):
            return [KeyEvent.ENTER]
        if byte == ord(" "):
            return [KeyEvent.SPACE]
        return []


class TerminalKeyListener:
    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._events: queue.Queue[KeyEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        self._old_settings = None
        self._decoder = KeyDecoder()

    def start(self):
        if not self._stream.isatty():
            raise RuntimeError("DAgger keyboard control requires an interactive terminal")
        fd = self._stream.fileno()
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._run, name="dagger-keyboard", daemon=True)
        self._thread.start()

    def _run(self):
        fd = self._stream.fileno()
        while not self._stop_event.is_set():
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            for event in self._decoder.feed(os.read(fd, 16)):
                if not self._paused.is_set():
                    self._events.put(event)

    def get(self, timeout=0.0):
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return

    def pause(self):
        self._paused.set()
        self.clear()

    def resume(self):
        self.clear()
        self._paused.clear()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            termios.tcsetattr(self._stream.fileno(), termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


class DaggerSession:
    """Pure state machine for the one-way policy-to-expert episode workflow."""

    def __init__(self, num_episodes: int):
        if num_episodes == 0 or num_episodes < -1:
            raise ValueError("num_episodes must be -1 or a positive integer")
        self.num_episodes = num_episodes
        self.state = SessionState.READY
        self.generation = 0
        self.saved_episodes = 0
        self.intervention_used = False
        self.label: bool | None = None

    def handle(self, event: KeyEvent) -> SessionCommand:
        if self.state in (SessionState.READY, SessionState.READY_NEXT):
            if event is KeyEvent.ENTER:
                self.generation += 1
                self.intervention_used = False
                self.label = None
                self.state = SessionState.AUTONOMOUS
                return SessionCommand.START_EPISODE
            return SessionCommand.NONE

        if self.state is SessionState.AUTONOMOUS:
            if event is KeyEvent.SPACE and not self.intervention_used:
                self.generation += 1
                self.intervention_used = True
                self.state = SessionState.SWITCHING_TO_EXPERT
                return SessionCommand.START_TAKEOVER
            if event is KeyEvent.ENTER:
                return self.finish_active_episode()
            return SessionCommand.NONE

        if self.state is SessionState.INTERVENTION:
            if event is KeyEvent.ENTER:
                return self.finish_active_episode()
            return SessionCommand.NONE

        if self.state is SessionState.AWAIT_LABEL:
            if event is KeyEvent.RIGHT:
                self.label = True
            elif event is KeyEvent.LEFT:
                self.label = False
            elif event is KeyEvent.ENTER and self.label is not None:
                self.state = SessionState.SAVING
                return SessionCommand.SAVE_EPISODE
            return SessionCommand.NONE

        return SessionCommand.NONE

    def takeover_ready(self):
        if self.state is not SessionState.SWITCHING_TO_EXPERT:
            raise RuntimeError(f"cannot complete takeover from {self.state.name}")
        self.state = SessionState.INTERVENTION

    def finish_active_episode(self) -> SessionCommand:
        if self.state not in (
            SessionState.AUTONOMOUS,
            SessionState.SWITCHING_TO_EXPERT,
            SessionState.INTERVENTION,
        ):
            return SessionCommand.NONE
        self.generation += 1
        self.state = SessionState.AWAIT_LABEL
        self.label = None
        return SessionCommand.STOP_EPISODE

    def complete_save(self):
        if self.state is not SessionState.SAVING:
            raise RuntimeError(f"cannot complete save from {self.state.name}")
        self.saved_episodes += 1
        if self.num_episodes != -1 and self.saved_episodes >= self.num_episodes:
            self.state = SessionState.COMPLETE
        else:
            self.state = SessionState.READY_NEXT

    def cancel_empty_save(self):
        if self.state is not SessionState.SAVING:
            raise RuntimeError(f"cannot cancel save from {self.state.name}")
        self.state = SessionState.READY_NEXT


@dataclass(frozen=True)
class DaggerEpisodeSummary:
    episode_index: int
    episode_type: str
    success: bool
    total_frames: int
    policy_frames: int
    intervention_frames: int

    def to_dict(self):
        return {
            "episode_index": self.episode_index,
            "episode_type": self.episode_type,
            "success": self.success,
            "total_frames": self.total_frames,
            "policy_frames": self.policy_frames,
            "intervention_frames": self.intervention_frames,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            episode_index=int(value["episode_index"]),
            episode_type=str(value["episode_type"]),
            success=bool(value["success"]),
            total_frames=int(value["total_frames"]),
            policy_frames=int(value["policy_frames"]),
            intervention_frames=int(value["intervention_frames"]),
        )


def classify_dagger_episode(intervention_flags, success: bool, episode_index: int):
    flags = np.asarray(intervention_flags, dtype=np.int64).reshape(-1)
    if flags.size == 0:
        raise ValueError("cannot classify an episode without frames")
    if not np.all(np.isin(flags, [0, 1])):
        raise ValueError("intervention flags must contain only 0 or 1")

    intervention_frames = int(np.count_nonzero(flags))
    policy_frames = int(flags.size - intervention_frames)
    outcome = "success" if success else "failure"
    episode_type = (
        f"intervention_{outcome}"
        if intervention_frames > 0
        else f"autonomous_{outcome}"
    )
    return DaggerEpisodeSummary(
        episode_index=int(episode_index),
        episode_type=episode_type,
        success=bool(success),
        total_frames=int(flags.size),
        policy_frames=policy_frames,
        intervention_frames=intervention_frames,
    )


def _scalar_value(value):
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError("empty scalar field")
    return values[0].item() if hasattr(values[0], "item") else values[0]


class DaggerEpisodeTracker:
    """Persist episode categories without changing the LeRobot training schema."""

    SUMMARY_RELATIVE_PATH = Path("meta/dagger_episode_summary.jsonl")

    def __init__(self, dataset_root, targets=None):
        self.dataset_root = Path(dataset_root)
        self.summary_path = self.dataset_root / self.SUMMARY_RELATIVE_PATH
        self.targets = {episode_type: -1 for episode_type in PRIMARY_EPISODE_CATEGORIES}
        if targets:
            unknown = set(targets) - set(PRIMARY_EPISODE_CATEGORIES)
            if unknown:
                raise ValueError(f"unknown DAgger target categories: {sorted(unknown)}")
            self.targets.update({key: int(value) for key, value in targets.items()})
        for episode_type, target in self.targets.items():
            if target < -1:
                raise ValueError(f"target for {episode_type} must be -1 or non-negative")

        self._records = self._load_summary()
        scanned = self._scan_dataset()
        if scanned:
            self._records = scanned
            self._write_summary()

    @property
    def records(self):
        return tuple(self._records[index] for index in sorted(self._records))

    @property
    def counts(self):
        return Counter(record.episode_type for record in self._records.values())

    def next_episode_index(self):
        return max(self._records, default=-1) + 1

    def record(self, summary: DaggerEpisodeSummary):
        self._records[summary.episode_index] = summary
        self._write_summary()

    def configured_targets_met(self):
        configured = {key: value for key, value in self.targets.items() if value >= 0}
        return bool(configured) and all(
            self.counts[key] >= value for key, value in configured.items()
        )

    def format_progress(self):
        counts = self.counts
        lines = ["DAgger episode totals:"]
        for episode_type in PRIMARY_EPISODE_CATEGORIES:
            target = self.targets[episode_type]
            progress = (
                f"{counts[episode_type]}/{target}"
                if target >= 0
                else str(counts[episode_type])
            )
            lines.append(f"  {CATEGORY_LABELS[episode_type]:20s} {progress}")
        return "\n".join(lines)

    def _load_summary(self):
        if not self.summary_path.exists():
            return {}
        records = {}
        with self.summary_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = DaggerEpisodeSummary.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    print(
                        f"Warning: ignoring invalid DAgger summary line "
                        f"{self.summary_path}:{line_number}; parquet data will be used."
                    )
                    continue
                records[record.episode_index] = record
        return records

    def _scan_dataset(self):
        parquet_files = sorted((self.dataset_root / "data").glob("**/*.parquet"))
        if not parquet_files:
            return {}

        import pyarrow.parquet as pq

        episodes = {}
        for parquet_path in parquet_files:
            available_columns = set(pq.read_schema(parquet_path).names)
            if "intervention" not in available_columns or "reward" not in available_columns:
                continue
            columns = ["intervention", "reward"]
            if "episode_index" in available_columns:
                columns.append("episode_index")
            table = pq.read_table(parquet_path, columns=columns)
            fallback_match = re.search(r"episode_(\d+)", parquet_path.stem)
            fallback_index = int(fallback_match.group(1)) if fallback_match else None
            for row_index in range(table.num_rows):
                if "episode_index" in columns:
                    episode_index = int(
                        _scalar_value(table["episode_index"][row_index].as_py())
                    )
                elif fallback_index is not None:
                    episode_index = fallback_index
                else:
                    continue
                episode = episodes.setdefault(
                    episode_index,
                    {"intervention_flags": [], "success": False},
                )
                episode["intervention_flags"].append(
                    int(_scalar_value(table["intervention"][row_index].as_py()))
                )
                episode["success"] = episode["success"] or (
                    float(_scalar_value(table["reward"][row_index].as_py())) > 0.5
                )

        return {
            episode_index: classify_dagger_episode(
                episode["intervention_flags"],
                episode["success"],
                episode_index,
            )
            for episode_index, episode in episodes.items()
        }

    def _write_summary(self):
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.summary_path.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        os.replace(temporary_path, self.summary_path)


@dataclass(frozen=True)
class InferenceRequest:
    generation: int
    observation: dict
    lead_steps: int = 0


@dataclass(frozen=True)
class InferenceResponse:
    generation: int
    lead_steps: int = 0
    result: dict | None = None
    error: BaseException | None = None


class InferenceWorker:
    """Own the blocking WebSocket infer calls and tag every result by generation."""

    def __init__(self, policy_client):
        self.policy_client = policy_client
        self._requests: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
        self._responses: queue.Queue[InferenceResponse] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dagger-inference", daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, request: InferenceRequest):
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        self._requests.put_nowait(request)

    def drain_responses(self) -> list[InferenceResponse]:
        responses = []
        while True:
            try:
                responses.append(self._responses.get_nowait())
            except queue.Empty:
                return responses

    def _run(self):
        while not self._stop_event.is_set():
            try:
                request = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return
            try:
                result = self.policy_client.infer(request.observation)
                self._responses.put(
                    InferenceResponse(
                        request.generation,
                        lead_steps=request.lead_steps,
                        result=result,
                    )
                )
            except BaseException as error:
                self._responses.put(
                    InferenceResponse(
                        request.generation,
                        lead_steps=request.lead_steps,
                        error=error,
                    )
                )

    def stop(self):
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        websocket = getattr(self.policy_client, "_ws", None)
        close = getattr(websocket, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        self._thread.join(timeout=2.0)


def _get_color_image(sensors: dict, keys: list[str]):
    for key in keys:
        camera_data = sensors.get(key)
        if camera_data is not None and "color" in camera_data:
            return camera_data["color"]
    return None


def _preprocess_image(image):
    from openpi_client import image_tools

    resized = image_tools.resize_with_pad(np.asarray(image), 224, 224)
    return image_tools.convert_to_uint8(np.asarray(resized))


def make_policy_observation(robot_data, prompt: str, adv_ind: str | None = None):
    controllers, sensors = robot_data
    follower = controllers["left_arm"]
    state = np.concatenate(
        [
            np.asarray(follower["joint"], dtype=np.float32).reshape(-1),
            np.asarray(follower["gripper"], dtype=np.float32).reshape(-1),
        ]
    )
    if state.shape != (PIPER_ACTION_DIM,):
        raise ValueError(f"follower state must have shape (7,), got {state.shape}")

    head = _get_color_image(sensors, ["cam_head", "image"])
    wrist = _get_color_image(sensors, ["cam_wrist", "wrist_image"])
    if wrist is None:
        raise KeyError("cam_wrist image is missing")
    if head is None:
        head = np.zeros_like(wrist)

    observation = {
        "state": state,
        "images": {
            "cam_high": _preprocess_image(head),
            "cam_wrist": _preprocess_image(wrist),
        },
        "prompt": prompt,
    }
    if adv_ind is not None:
        observation["adv_ind"] = adv_ind
    return observation


def policy_action_to_move(action) -> dict:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < PIPER_ACTION_DIM:
        raise ValueError(f"policy action must have at least 7 values, got {action.shape}")
    action = action[:PIPER_ACTION_DIM]
    if not np.all(np.isfinite(action)):
        raise ValueError("policy action must contain only finite values")
    return {"joint": action[:6].copy(), "gripper": float(action[6])}


def select_action_chunk(actions, chunk_size: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] < PIPER_ACTION_DIM:
        raise ValueError(f"policy actions must have shape (horizon, >=7), got {actions.shape}")
    return actions[:chunk_size, :PIPER_ACTION_DIM]


def _next_dataset_episode_index(collector, tracker):
    dataset = getattr(collector, "dataset", None)
    dataset_buffer = getattr(dataset, "episode_buffer", None)
    if isinstance(dataset_buffer, dict) and "episode_index" in dataset_buffer:
        return int(_scalar_value(dataset_buffer["episode_index"]))
    num_episodes = getattr(dataset, "num_episodes", None)
    if num_episodes is not None:
        return int(num_episodes)
    total_episodes = getattr(getattr(dataset, "meta", None), "total_episodes", None)
    if total_episodes is not None:
        return int(total_episodes)
    return tracker.next_episode_index()


def save_dagger_episode(collector, success: bool, tracker: DaggerEpisodeTracker | None = None):
    """Save raw rollout labels without assigning final PiStar advantage labels."""
    episode_buffer = getattr(collector, "episode_buffer", None)
    if episode_buffer is not None and len(episode_buffer) == 0:
        return None if tracker is not None else False

    summary = None
    if tracker is not None:
        intervention_flags = getattr(collector, "intervention_flags", None)
        if intervention_flags is None or len(intervention_flags) != len(episode_buffer):
            raise RuntimeError("episode frames and intervention flags are inconsistent")
        summary = classify_dagger_episode(
            intervention_flags,
            success,
            _next_dataset_episode_index(collector, tracker),
        )

    collector.save_episode(success=bool(success), adv_ind_value="none")
    if tracker is not None:
        tracker.record(summary)
        return summary
    return True


def _read_master_ctrl_action(master, *, after_timestamp: float | None = None):
    try:
        ctrl = master.controller.GetArmJointCtrl()
        if getattr(ctrl, "Hz", 0) <= 0:
            return None
        if after_timestamp is not None and getattr(ctrl, "time_stamp", 0.0) <= after_timestamp:
            return None
        joints = ctrl.joint_ctrl
        joint = np.asarray(
            [
                joints.joint_1,
                joints.joint_2,
                joints.joint_3,
                joints.joint_4,
                joints.joint_5,
                joints.joint_6,
            ],
            dtype=float,
        ) / 57295.7795
        gripper_ctrl = master.controller.GetArmGripperCtrl()
        if getattr(gripper_ctrl, "Hz", 0) <= 0:
            return None
        gripper = float(gripper_ctrl.gripper_ctrl.grippers_angle) / (70 * 1000)
        return joint, gripper, "ctrl"
    except Exception:
        return None


def _read_master_feedback_action(master):
    try:
        state = master.get_state()
        return np.asarray(state["joint"], dtype=float), float(state["gripper"]), "feedback"
    except Exception:
        return None


class TeleopMapping:
    def __init__(
        self,
        *,
        joint_sign,
        joint_offset,
        joint_scale,
        gripper_scale,
        gripper_offset,
    ):
        self.joint_sign = np.asarray(joint_sign, dtype=float)
        self.joint_offset = np.asarray(joint_offset, dtype=float)
        if self.joint_sign.shape != (6,) or self.joint_offset.shape != (6,):
            raise ValueError("joint_sign and joint_offset must each contain 6 values")
        self.joint_scale = float(joint_scale)
        self.gripper_scale = float(gripper_scale)
        self.gripper_offset = float(gripper_offset)
        self.runtime_joint_offset = np.zeros(6, dtype=float)
        self.runtime_gripper_offset = 0.0
        self.source = "ctrl"

    def align(self, master_raw, follower_state):
        master_joint, master_gripper, source = master_raw
        follower_joint = np.asarray(follower_state["joint"], dtype=float)
        follower_gripper = float(follower_state["gripper"])
        self.runtime_joint_offset = follower_joint - (
            np.asarray(master_joint) * self.joint_sign * self.joint_scale + self.joint_offset
        )
        self.runtime_gripper_offset = follower_gripper - (
            float(master_gripper) * self.gripper_scale + self.gripper_offset
        )
        self.source = source

    def transform(self, master_raw):
        joint, gripper, _ = master_raw
        mapped_joint = (
            np.asarray(joint, dtype=float) * self.joint_sign * self.joint_scale
            + self.joint_offset
            + self.runtime_joint_offset
        )
        mapped_gripper = (
            float(gripper) * self.gripper_scale
            + self.gripper_offset
            + self.runtime_gripper_offset
        )
        return {
            "joint": mapped_joint.tolist(),
            "gripper": float(np.clip(mapped_gripper, 0.0, 1.0)),
        }


class DaggerRuntime:
    def __init__(
        self,
        *,
        robot,
        collector,
        inference_worker: InferenceWorker,
        prompt: str,
        inference_adv_ind: str | None,
        sample_fps: float,
        teleop_fps: float,
        chunk_size: int,
        reset_settle_seconds: float,
        alignment_timeout: float,
        feedback_fallback: bool,
        teleop_mapping: TeleopMapping,
        filter_kwargs: dict,
        prefetch_threshold: int = 0,
        reset_script=None,
    ):
        self.robot = robot
        self.collector = collector
        self.inference_worker = inference_worker
        self.prompt = prompt
        self.inference_adv_ind = inference_adv_ind
        self.sample_period = 1.0 / float(sample_fps)
        self.teleop_fps = float(teleop_fps)
        self.chunk_size = int(chunk_size)
        self.prefetch_threshold = int(prefetch_threshold)
        if not 0 <= self.prefetch_threshold < self.chunk_size:
            raise ValueError("prefetch_threshold must be in [0, chunk_size)")
        self.reset_settle_seconds = float(reset_settle_seconds)
        self.reset_script = Path(
            reset_script
            if reset_script is not None
            else CONTROL_ROOT / "scripts" / "piperx" / "2_arm_go_init.sh"
        )
        self.alignment_timeout = float(alignment_timeout)
        self.feedback_fallback = bool(feedback_fallback)
        self.teleop_mapping = teleop_mapping
        self.filter_kwargs = dict(filter_kwargs)
        self.generation = 0
        self.step_count = 0
        self._actions = deque()
        self._last_policy_action = None
        self._latest_observation = None
        self._pending_generation = None
        self._next_sample_at = 0.0
        self._teleop_loop = None
        self._teleop_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._master_drag_enabled = False

    def start_episode(self, generation: int):
        self.stop_active_control(restore_master=True)
        self.collector.clear_current_episode()
        self.robot.set_policy_enabled(False)
        self._run_reset_script()
        if self.reset_settle_seconds > 0:
            time.sleep(self.reset_settle_seconds)
        self.robot.set_policy_enabled(True)

        self._invalidate_policy(generation)
        self.step_count = 0
        self._last_policy_action = None
        self._latest_observation = make_policy_observation(
            self.robot.get(), self.prompt, self.inference_adv_ind
        )
        self._submit_inference()
        self._next_sample_at = time.monotonic()

    def _run_reset_script(self):
        if not self.reset_script.is_file():
            raise FileNotFoundError(f"reset script not found: {self.reset_script}")
        subprocess.run(["bash", str(self.reset_script)], check=True)

    def _invalidate_policy(self, generation: int):
        self.generation = generation
        self._actions.clear()
        self._pending_generation = None
        self.inference_worker.drain_responses()

    def _submit_inference(self):
        if self._pending_generation is not None or self._latest_observation is None:
            return
        lead_steps = len(self._actions)
        self.inference_worker.submit(
            InferenceRequest(
                self.generation,
                self._latest_observation,
                lead_steps=lead_steps,
            )
        )
        self._pending_generation = self.generation

    def _drain_policy_responses(self):
        for response in self.inference_worker.drain_responses():
            if response.generation != self.generation:
                continue
            self._pending_generation = None
            if response.error is not None:
                raise RuntimeError("WebSocket inference failed") from response.error
            action_chunk = select_action_chunk(response.result["actions"], self.chunk_size)
            if not 0 <= response.lead_steps < len(action_chunk):
                raise RuntimeError(
                    f"invalid inference lead_steps={response.lead_steps} "
                    f"for chunk length {len(action_chunk)}"
                )
            self._actions.extend(action_chunk[response.lead_steps :])

    def tick_autonomous(self):
        self._drain_policy_responses()
        now = time.monotonic()
        if now < self._next_sample_at:
            return False

        if not self._actions:
            if self._pending_generation is None:
                self._latest_observation = make_policy_observation(
                    self.robot.get(), self.prompt, self.inference_adv_ind
                )
            self._submit_inference()
            self._advance_sample_clock(now)
            return False

        requested_action = self._actions.popleft()
        self._last_policy_action = requested_action.copy()

        sent_action = self.robot.move_follower(policy_action_to_move(requested_action))
        if sent_action is None:
            raise RuntimeError("policy command was rejected before CAN send")
        robot_data = self.robot.get()
        self._collect(robot_data, sent_action, is_intervention=False)
        self._latest_observation = make_policy_observation(
            robot_data, self.prompt, self.inference_adv_ind
        )
        self.step_count += 1
        if len(self._actions) <= self.prefetch_threshold:
            self._submit_inference()
        self._advance_sample_clock(now)
        return True

    def begin_takeover(self, generation: int):
        self._invalidate_policy(generation)
        self.robot.set_policy_enabled(False)
        master = self.robot.controllers["arm"]["right_arm"]
        previous_ctrl = master.controller.GetArmJointCtrl()
        previous_ctrl_timestamp = float(getattr(previous_ctrl, "time_stamp", 0.0))
        self.robot.hold_follower_position()
        self.robot.enable_master_drag_mode()
        self._master_drag_enabled = True

        master_raw = self._wait_for_master_action(
            master, after_ctrl_timestamp=previous_ctrl_timestamp
        )
        follower_state = self.robot.get_follower_state()
        self.teleop_mapping.align(master_raw, follower_state)

        from robot.utils.teleop_filter import EmaSlewFilter
        from robot.utils.teleop_filter import FixedRateControlLoop

        action_filter = EmaSlewFilter(**self.filter_kwargs)
        action_filter.seed(follower_state["joint"], follower_state["gripper"])

        def teleop_step():
            try:
                raw_action = self._read_aligned_master_action(master)
                if raw_action is None:
                    return None
                move_data = action_filter.process(self.teleop_mapping.transform(raw_action))
                sent_action = self.robot.move_follower(move_data, bypass_policy=True)
                if sent_action is None:
                    raise RuntimeError("expert command was rejected before CAN send")
                return sent_action
            except BaseException as error:
                try:
                    self._teleop_errors.put_nowait(error)
                except queue.Full:
                    pass
                return None

        self._teleop_loop = FixedRateControlLoop(self.teleop_fps, teleop_step)
        self._teleop_loop.start()
        self._next_sample_at = time.monotonic() + self.sample_period

    def _wait_for_master_action(self, master, *, after_ctrl_timestamp: float):
        deadline = time.monotonic() + self.alignment_timeout
        while time.monotonic() < deadline:
            action = _read_master_ctrl_action(master, after_timestamp=after_ctrl_timestamp)
            if action is None and self.feedback_fallback:
                action = _read_master_feedback_action(master)
            if action is not None:
                return action
            time.sleep(0.02)
        raise RuntimeError("master control frame is not ready after switching to drag mode")

    def _read_aligned_master_action(self, master):
        if self.teleop_mapping.source == "feedback":
            return _read_master_feedback_action(master)
        action = _read_master_ctrl_action(master)
        if action is None and self.feedback_fallback:
            return _read_master_feedback_action(master)
        return action

    def tick_intervention(self):
        try:
            error = self._teleop_errors.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            raise RuntimeError("expert teleoperation loop failed") from error

        now = time.monotonic()
        if now < self._next_sample_at:
            return False
        sent_action = self._teleop_loop.get_latest() if self._teleop_loop is not None else None
        if sent_action is None:
            self._advance_sample_clock(now)
            return False
        self._collect(self.robot.get(), sent_action, is_intervention=True)
        self.step_count += 1
        self._advance_sample_clock(now)
        return True

    def _collect(self, robot_data, sent_action, *, is_intervention: bool):
        controllers, sensors = robot_data
        follower_only = {"left_arm": controllers["left_arm"]}
        self.collector.collect(
            follower_only,
            sensors,
            action_data=np.asarray(sent_action, dtype=np.float32),
            is_intervention=is_intervention,
        )

    def _advance_sample_clock(self, now):
        self._next_sample_at += self.sample_period
        if self._next_sample_at <= now:
            self._next_sample_at = now + self.sample_period

    def stop_active_control(self, *, restore_master: bool):
        if self._teleop_loop is not None:
            if not self._teleop_loop.stop(timeout=None):
                raise RuntimeError("expert teleoperation loop did not stop")
            self._teleop_loop = None
        self.robot.set_policy_enabled(False)
        if getattr(self.robot.controllers["arm"]["left_arm"], "controller", None) is not None:
            self.robot.hold_follower_position()
        if restore_master and self._master_drag_enabled:
            self.robot.disable_master_drag_mode()
            self._master_drag_enabled = False

    def stop_episode(self, generation: int):
        self._invalidate_policy(generation)
        self.stop_active_control(restore_master=True)


def _cleanup_robot(robot):
    for group in getattr(robot, "sensors", {}).values():
        for sensor in group.values():
            cleanup = getattr(sensor, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    print(f"Warning: camera cleanup failed: {error}")
    for group in getattr(robot, "controllers", {}).values():
        for controller in group.values():
            sdk = getattr(controller, "controller", None)
            disconnect = getattr(sdk, "DisconnectPort", None)
            if disconnect is not None:
                try:
                    disconnect()
                except Exception as error:
                    print(f"Warning: CAN cleanup failed: {error}")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _build_parser():
    parser = argparse.ArgumentParser(description="PiperX WebSocket rollout + one-way DAgger collector")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--master-can", default="can_left_mas")
    parser.add_argument("--follower-can", default="can_left_slave")
    parser.add_argument("--sample-fps", type=float, default=30.0)
    parser.add_argument("--teleop-fps", type=float, default=60.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument(
        "--prefetch-threshold",
        type=int,
        default=0,
        help="request the next chunk with this many current actions remaining; 0 disables overlap",
    )
    parser.add_argument("--num-episode", type=int, default=-1)
    parser.add_argument("--max-step", type=int, default=900, help="-1 disables automatic episode stop")
    parser.add_argument("--target-autonomous-success", type=int, default=-1)
    parser.add_argument("--target-autonomous-failure", type=int, default=-1)
    parser.add_argument("--target-intervention-success", type=int, default=-1)
    parser.add_argument("--target-intervention-failure", type=int, default=-1)
    parser.add_argument("--reset-settle-seconds", type=float, default=2.0)
    parser.add_argument("--alignment-timeout", type=float, default=2.0)
    parser.add_argument("--feedback-fallback", type=_parse_bool, default=False)
    parser.add_argument("--adv-ind", default=None, help="optional inference condition; raw dataset still stores none")
    parser.add_argument("--ema-enabled", type=_parse_bool, default=True)
    parser.add_argument("--ema-alpha", type=float, default=0.8)
    parser.add_argument("--slew-enabled", type=_parse_bool, default=True)
    parser.add_argument("--max-joint-step", type=float, default=0.04)
    parser.add_argument("--max-gripper-step", type=float, default=0.025 / 0.07)
    parser.add_argument("--joint-sign", type=float, nargs=6, default=[1.0] * 6)
    parser.add_argument("--joint-offset", type=float, nargs=6, default=[0.0] * 6)
    parser.add_argument("--joint-scale", type=float, default=1.0)
    parser.add_argument("--gripper-scale", type=float, default=1.0)
    parser.add_argument("--gripper-offset", type=float, default=0.0)
    return parser


def main():
    from my_robot.piper_dagger import PiperDAgger
    from openpi_client import websocket_client_policy
    from robot.data.collect_lerobot_rl import CollectLeRobotRL

    args = _build_parser().parse_args()
    if args.sample_fps <= 0 or args.teleop_fps <= 0:
        raise SystemExit("--sample-fps and --teleop-fps must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if not 0 <= args.prefetch_threshold < args.chunk_size:
        raise SystemExit("--prefetch-threshold must be in [0, chunk-size)")
    if args.num_episode == 0 or args.num_episode < -1:
        raise SystemExit("--num-episode must be -1 or a positive integer")
    if args.max_step == 0 or args.max_step < -1:
        raise SystemExit("--max-step must be -1 or a positive integer")
    category_targets = {
        "autonomous_success": args.target_autonomous_success,
        "autonomous_failure": args.target_autonomous_failure,
        "intervention_success": args.target_intervention_success,
        "intervention_failure": args.target_intervention_failure,
    }
    if any(target < -1 for target in category_targets.values()):
        raise SystemExit("DAgger category targets must be -1 or non-negative")

    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        if not stop_requested.is_set():
            print("\nCtrl+C received; finishing any active dataset save, then exiting.")
        stop_requested.set()

    robot = None
    collector = None
    policy_client = None
    inference_worker = None
    runtime = None
    episode_tracker = None
    listener = TerminalKeyListener()
    session = DaggerSession(args.num_episode)
    episode_unsaved = False

    try:
        print(f"Connecting to ws://{args.server_host}:{args.server_port} ...")
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host, port=args.server_port
        )
        metadata = policy_client.get_server_metadata()
        requires_adv_ind = bool(metadata.get("requires_adv_ind")) or metadata.get("deploy_mode") == "pi05star"
        if requires_adv_ind and not args.adv_ind:
            raise RuntimeError("connected PiStar server requires --adv-ind")
        inference_adv_ind = args.adv_ind
        if metadata.get("requires_adv_ind") is False:
            inference_adv_ind = None
        print(f"Server metadata: {metadata}")

        robot = PiperDAgger(
            master_can=args.master_can,
            follower_can=args.follower_can,
            follower_use_mit_mode=True,
        )
        robot.set_up()

        collector = CollectLeRobotRL(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            task_name=args.task_name,
            fps=int(args.sample_fps),
            robot_type="piperx",
            state_dim=7,
            action_dim=7,
            image_size=(480, 640),
            camera_keys={
                "cam_head": "observation.images.cam_head",
                "cam_wrist": "observation.images.cam_wrist",
            },
            move_check=False,
        )
        episode_tracker = DaggerEpisodeTracker(
            collector.get_dataset_path(),
            targets=category_targets,
        )
        print(episode_tracker.format_progress())

        inference_worker = InferenceWorker(policy_client)
        inference_worker.start()
        runtime = DaggerRuntime(
            robot=robot,
            collector=collector,
            inference_worker=inference_worker,
            prompt=args.task_name,
            inference_adv_ind=inference_adv_ind,
            sample_fps=args.sample_fps,
            teleop_fps=args.teleop_fps,
            chunk_size=args.chunk_size,
            prefetch_threshold=args.prefetch_threshold,
            reset_settle_seconds=args.reset_settle_seconds,
            reset_script=CONTROL_ROOT / "scripts" / "piperx" / "2_arm_go_init.sh",
            alignment_timeout=args.alignment_timeout,
            feedback_fallback=args.feedback_fallback,
            teleop_mapping=TeleopMapping(
                joint_sign=args.joint_sign,
                joint_offset=args.joint_offset,
                joint_scale=args.joint_scale,
                gripper_scale=args.gripper_scale,
                gripper_offset=args.gripper_offset,
            ),
            filter_kwargs={
                "ema_enabled": args.ema_enabled,
                "ema_alpha": args.ema_alpha,
                "slew_enabled": args.slew_enabled,
                "max_joint_step": args.max_joint_step,
                "max_gripper_step": args.max_gripper_step,
            },
        )

        listener.start()
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        print("\nREADY: Enter starts policy rollout; Ctrl+C exits.")

        while session.state is not SessionState.COMPLETE:
            if stop_requested.is_set():
                break

            event = listener.get(timeout=0.005)
            command = session.handle(event) if event is not None else SessionCommand.NONE

            if command is SessionCommand.START_EPISODE:
                listener.pause()
                print("\nResetting both arms and starting policy rollout...")
                runtime.start_episode(session.generation)
                episode_unsaved = True
                print("AUTONOMOUS: Space hands control to expert once; Enter ends episode.")
                listener.resume()
            elif command is SessionCommand.START_TAKEOVER:
                listener.pause()
                print("\nSwitching to expert: invalidating policy chunk and aligning master...")
                runtime.begin_takeover(session.generation)
                session.takeover_ready()
                print("TAKEOVER READY: expert controls follower; Space is now disabled; Enter ends episode.")
                listener.resume()
            elif command is SessionCommand.STOP_EPISODE:
                listener.pause()
                runtime.stop_episode(session.generation)
                print("\nLABEL: Right arrow = success, Left arrow = failure, then Enter saves.")
                listener.resume()
            elif command is SessionCommand.SAVE_EPISODE:
                label_text = "success" if session.label else "failure"
                print(f"\nSaving {label_text} episode ({runtime.step_count} frames)...")
                listener.pause()
                saved_summary = save_dagger_episode(
                    collector,
                    success=bool(session.label),
                    tracker=episode_tracker,
                )
                if saved_summary is not None:
                    episode_unsaved = False
                    session.complete_save()
                    print(
                        f"Saved dataset episode {saved_summary.episode_index}: "
                        f"{saved_summary.episode_type}, {saved_summary.total_frames} frames. "
                        "Raw adv_ind=none."
                    )
                    print(episode_tracker.format_progress())
                    if episode_tracker.configured_targets_met():
                        print("All configured category targets are met; Ctrl+C exits normally.")
                else:
                    collector.clear_current_episode()
                    episode_unsaved = False
                    session.cancel_empty_save()
                    print("Empty episode was not saved or counted.")
                listener.resume()
                if session.state is SessionState.READY_NEXT:
                    print("READY NEXT: reset the scene, then press Enter for the next rollout.")
                if stop_requested.is_set():
                    break

            if session.state is SessionState.AWAIT_LABEL and event in (KeyEvent.RIGHT, KeyEvent.LEFT):
                selected = "success" if session.label else "failure"
                print(f"Label selected: {selected}. Press Enter to save.")

            if session.state is SessionState.AUTONOMOUS:
                runtime.tick_autonomous()
            elif session.state is SessionState.INTERVENTION:
                runtime.tick_intervention()

            if (
                args.max_step > 0
                and session.state in (SessionState.AUTONOMOUS, SessionState.INTERVENTION)
                and runtime.step_count >= args.max_step
            ):
                session.finish_active_episode()
                listener.pause()
                runtime.stop_episode(session.generation)
                print("\nMaximum step count reached.")
                print("LABEL: Right arrow = success, Left arrow = failure, then Enter saves.")
                listener.resume()

        if session.state is SessionState.COMPLETE:
            print(f"\nCompleted {session.saved_episodes} DAgger episodes.")
    finally:
        if runtime is not None:
            try:
                runtime.stop_active_control(restore_master=True)
            except Exception as error:
                print(f"Warning: control cleanup failed: {error}")
        if collector is not None and episode_unsaved:
            collector.clear_current_episode()
            print("Unsaved current episode discarded.")
        dataset = getattr(collector, "dataset", None)
        if dataset is not None:
            try:
                dataset.stop_image_writer()
            except Exception as error:
                print(f"Warning: image writer cleanup failed: {error}")
        listener.stop()
        if inference_worker is not None:
            inference_worker.stop()
        if robot is not None:
            _cleanup_robot(robot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C received; DAgger collector stopped normally.")

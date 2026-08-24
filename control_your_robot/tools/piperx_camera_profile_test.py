#!/usr/bin/env python3
"""Validate the two RealSense cameras with PiStar's production profile order."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import importlib
import sys


@dataclass(frozen=True)
class CameraSpec:
    role: str
    serial: str
    model_token: str


@dataclass
class RunningCamera:
    spec: CameraSpec
    pipeline: object
    profile: tuple[int, int, int]


CAMERAS = (
    CameraSpec(role="head", serial="337122071685", model_token="D435I"),
    CameraSpec(role="wrist", serial="230322274885", model_token="D405"),
)

# Keep this order aligned with robot/sensor/Realsense_sensor.py.
PROFILE_ORDER = (
    (1280, 720, 10),
    (1280, 720, 30),
    (1280, 720, 15),
    (640, 480, 30),
)


def device_info(device: object, field: object) -> str:
    return device.get_info(field) if device.supports(field) else "unknown"


def find_expected_devices(context: object, rs_module: object) -> dict[str, object]:
    devices = {device_info(device, rs_module.camera_info.serial_number): device for device in context.query_devices()}
    errors: list[str] = []

    for spec in CAMERAS:
        device = devices.get(spec.serial)
        if device is None:
            errors.append(f"missing {spec.role} camera with serial {spec.serial}")
            continue

        name = device_info(device, rs_module.camera_info.name)
        if spec.model_token not in name:
            errors.append(f"serial {spec.serial} is {name!r}, expected model containing {spec.model_token!r}")
        print(f"FOUND: role={spec.role} name={name} serial={spec.serial}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return devices


def supported_color_profiles(device: object, rs_module: object) -> set[tuple[int, int, int]]:
    profiles: set[tuple[int, int, int]] = set()
    for sensor in device.query_sensors():
        for stream_profile in sensor.get_stream_profiles():
            if stream_profile.stream_type() != rs_module.stream.color:
                continue
            if stream_profile.format() != rs_module.format.bgr8:
                continue
            video_profile = stream_profile.as_video_stream_profile()
            profiles.add((video_profile.width(), video_profile.height(), video_profile.fps()))
    return profiles


def capture_frames(running: RunningCamera, frame_count: int, timeout_ms: int) -> None:
    width, height, _ = running.profile
    for index in range(frame_count):
        frames = running.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"{running.spec.role} returned no color frame at sample {index + 1}")
        if color.get_width() != width or color.get_height() != height:
            raise RuntimeError(
                f"{running.spec.role} frame is {color.get_width()}x{color.get_height()}, expected {width}x{height}"
            )


def start_with_production_fallback(
    spec: CameraSpec, frame_count: int, timeout_ms: int, rs_module: object
) -> RunningCamera:
    errors: list[str] = []

    for width, height, fps in PROFILE_ORDER:
        pipeline = rs_module.pipeline()
        config = rs_module.config()
        config.enable_device(spec.serial)
        config.disable_all_streams()
        config.enable_stream(rs_module.stream.color, width, height, rs_module.format.bgr8, fps)
        print(f"TRY: role={spec.role} serial={spec.serial} profile={width}x{height}@{fps}")

        try:
            active = pipeline.start(config)
            color_profile = active.get_stream(rs_module.stream.color).as_video_stream_profile()
            selected = (
                color_profile.width(),
                color_profile.height(),
                color_profile.fps(),
            )
            running = RunningCamera(spec=spec, pipeline=pipeline, profile=selected)
            capture_frames(running, frame_count, timeout_ms)
            print(f"STARTED: role={spec.role} serial={spec.serial} profile={selected[0]}x{selected[1]}@{selected[2]}")
            return running
        except RuntimeError as exc:
            errors.append(f"{width}x{height}@{fps}: {exc}")
            with contextlib.suppress(RuntimeError):
                pipeline.stop()

    raise RuntimeError(f"unable to start {spec.role} camera {spec.serial}; " + "; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the fixed D435I head and D405 wrist serials, then start them "
            "together using PiStar's RealSense profile fallback order."
        )
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="enumerate target BGR8 profiles without starting either stream",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="frames to validate after each start and during the simultaneous check",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=5000,
        help="per-frame wait timeout in milliseconds",
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.timeout_ms < 1:
        parser.error("--timeout-ms must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        rs_module = importlib.import_module("pyrealsense2")
    except ModuleNotFoundError:
        print(
            "ERROR: pyrealsense2 is not installed; run 'uv pip install -r pistar_requirements.txt' first.",
            file=sys.stderr,
        )
        return 2

    context = rs_module.context()
    devices = find_expected_devices(context, rs_module)

    support_error = False
    for spec in CAMERAS:
        supported = supported_color_profiles(devices[spec.serial], rs_module)
        print(f"TARGET PROFILES: role={spec.role}")
        for profile in PROFILE_ORDER:
            status = "yes" if profile in supported else "no"
            print(f"  {profile[0]}x{profile[1]}@{profile[2]}: {status}")
        if not any(width == 1280 and height == 720 for width, height, _ in supported):
            print(f"ERROR: {spec.role} has no 1280x720 BGR8 profile", file=sys.stderr)
            support_error = True

    if support_error:
        return 1
    if args.list_only:
        print("CAMERA_PROFILE_ENUMERATION=PASS")
        return 0

    running_cameras: list[RunningCamera] = []
    try:
        # Production initializes head first and keeps it running while wrist starts.
        for spec in CAMERAS:
            running_cameras.append(  # noqa: PERF401 - preserve partial state for cleanup
                start_with_production_fallback(spec, args.frames, args.timeout_ms, rs_module)
            )

        # Re-read both after both pipelines are active to expose USB bandwidth issues.
        for running in running_cameras:
            capture_frames(running, args.frames, args.timeout_ms)
            width, height, fps = running.profile
            print(f"SIMULTANEOUS PASS: role={running.spec.role} profile={width}x{height}@{fps} frames={args.frames}")

        non_720p = [running for running in running_cameras if running.profile[0:2] != (1280, 720)]
        if non_720p:
            for running in non_720p:
                print(
                    f"ERROR: {running.spec.role} fell back to {running.profile[0]}x{running.profile[1]}",
                    file=sys.stderr,
                )
            return 1

        print("CAMERA_PROFILE_CHECK=PASS")
        return 0
    finally:
        for running in reversed(running_cameras):
            with contextlib.suppress(RuntimeError):
                running.pipeline.stop()


if __name__ == "__main__":
    try:
        exit_code = main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    raise SystemExit(exit_code)

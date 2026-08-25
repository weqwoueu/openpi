#!/usr/bin/env python3
"""Rebuild a LeRobot v2.1 dataset into a clean, independent copy."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _set_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"Missing required parquet column: {name}")
    column_type = table.schema.field(column_index).type
    return table.set_column(column_index, name, pa.array(values, type=column_type))


def _success_labels(length: int, *, success: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if success:
        value_label = -(length - np.arange(length, dtype=np.float32)) / float(length)
        value_label[-1] = 0.0
        reward = np.zeros(length, dtype=np.float32)
        reward[-1] = 1.0
        reward_label = np.full(length, -1.0 / float(length), dtype=np.float32)
        reward_label[-1] = 0.0
    else:
        value_label = np.full(length, -1.0, dtype=np.float32)
        reward = np.zeros(length, dtype=np.float32)
        reward_label = np.full(length, -1.0 / float(length), dtype=np.float32)
        reward_label[-1] = -1.0
    return value_label, reward, reward_label


def _rewrite_episode_table(
    parquet_path: Path,
    *,
    episode_index: int,
    global_start: int,
    fps: int,
    drop_first_frame: bool,
) -> tuple[pa.Table, bool]:
    parquet_file = pq.ParquetFile(parquet_path)
    compression = parquet_file.metadata.row_group(0).column(0).compression.lower()
    table = parquet_file.read()
    if drop_first_frame:
        if table.num_rows < 2:
            raise ValueError(f"Cannot drop the first frame from a one-frame episode: {parquet_path}")
        table = table.slice(1)

    length = table.num_rows
    frame_index = np.arange(length, dtype=np.int64)
    timestamp = (frame_index / float(fps)).astype(np.float32)
    global_index = np.arange(global_start, global_start + length, dtype=np.int64)
    episode_indices = np.full(length, episode_index, dtype=np.int64)

    terminal_reward = float(table.column("reward")[-1].as_py())
    success = terminal_reward > 0.5
    value_label, reward, reward_label = _success_labels(length, success=success)

    table = _set_column(table, "timestamp", timestamp)
    table = _set_column(table, "frame_index", frame_index)
    table = _set_column(table, "episode_index", episode_indices)
    table = _set_column(table, "index", global_index)
    table = _set_column(table, "value_label", value_label)
    table = _set_column(table, "reward", reward)
    table = _set_column(table, "reward_label", reward_label)

    temporary = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression=compression)
    os.replace(temporary, parquet_path)
    return table, success


def _trim_video_first_frame(video_path: Path, fps: int, *, expected_length: int) -> None:
    temporary = video_path.with_name(video_path.stem + ".trimmed.mp4")
    av.logging.set_level(av.logging.ERROR)
    try:
        with av.open(str(video_path)) as input_container, av.open(str(temporary), mode="w") as output_container:
            input_stream = input_container.streams.video[0]
            output_stream = output_container.add_stream(
                "libsvtav1",
                rate=fps,
                options={"preset": "8", "g": "2", "crf": "30"},
            )
            output_stream.width = input_stream.width
            output_stream.height = input_stream.height
            output_stream.pix_fmt = "yuv420p"
            output_stream.time_base = Fraction(1, fps)

            output_index = 0
            for input_index, frame in enumerate(input_container.decode(input_stream)):
                if input_index == 0:
                    continue
                frame.pts = output_index
                frame.time_base = Fraction(1, fps)
                for packet in output_stream.encode(frame):
                    output_container.mux(packet)
                output_index += 1
            for packet in output_stream.encode():
                output_container.mux(packet)

        if output_index != expected_length:
            raise ValueError(
                f"Input video frame count mismatch after trimming: {video_path} "
                f"decoded={output_index + 1} expected={expected_length + 1}"
            )
        with av.open(str(temporary)) as verification_container:
            verification_stream = verification_container.streams.video[0]
            encoded_length = sum(1 for _ in verification_container.decode(verification_stream))
        if encoded_length != expected_length:
            raise ValueError(
                f"Encoded video frame count mismatch: {video_path} "
                f"decoded={encoded_length} expected={expected_length}"
            )
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)


def _feature_stats(array: np.ndarray) -> dict[str, list[Any]]:
    keepdims = array.ndim == 1
    return {
        "min": np.min(array, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=0, keepdims=keepdims).tolist(),
        "count": [len(array)],
    }


def _estimate_num_samples(length: int) -> int:
    minimum = min(100, length)
    return max(minimum, min(int(length**0.75), 10_000))


def _sample_indices(length: int) -> list[int]:
    count = _estimate_num_samples(length)
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()


def _video_stats(video_path: Path, length: int) -> dict[str, list[Any]]:
    wanted = set(_sample_indices(length))
    channel_min = np.full(3, np.inf, dtype=np.float64)
    channel_max = np.full(3, -np.inf, dtype=np.float64)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    sampled_count = 0
    decoded_count = 0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            decoded_count += 1
            if frame_index not in wanted:
                continue
            image = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
            height, width = image.shape[1:]
            downsample_factor = int(width / 150) if width > height else int(height / 150)
            if max(width, height) >= 300 and downsample_factor > 1:
                image = image[:, ::downsample_factor, ::downsample_factor]
            image = image.astype(np.float64) / 255.0
            channel_min = np.minimum(channel_min, image.min(axis=(1, 2)))
            channel_max = np.maximum(channel_max, image.max(axis=(1, 2)))
            channel_sum += image.sum(axis=(1, 2))
            channel_square_sum += np.square(image).sum(axis=(1, 2))
            pixel_count += image.shape[1] * image.shape[2]
            sampled_count += 1

    if decoded_count != length:
        raise ValueError(
            f"Video frame count mismatch: {video_path} decoded={decoded_count} expected={length}"
        )
    if sampled_count != len(wanted):
        raise ValueError(
            f"Video frame count mismatch while computing stats: {video_path} "
            f"sampled={sampled_count} expected={len(wanted)}"
        )
    channel_mean = channel_sum / pixel_count
    channel_variance = np.maximum(channel_square_sum / pixel_count - np.square(channel_mean), 0.0)

    def shaped(values: np.ndarray) -> list[Any]:
        return values.reshape(3, 1, 1).tolist()

    return {
        "min": shaped(channel_min),
        "max": shaped(channel_max),
        "mean": shaped(channel_mean),
        "std": shaped(np.sqrt(channel_variance)),
        "count": [sampled_count],
    }


def _table_stats(
    table: pa.Table,
    features: dict[str, dict[str, Any]],
    video_paths: dict[str, Path],
) -> dict[str, dict[str, list[Any]]]:
    stats: dict[str, dict[str, list[Any]]] = {}
    for key, feature in features.items():
        dtype = feature["dtype"]
        if dtype == "string":
            continue
        if dtype == "video":
            stats[key] = _video_stats(video_paths[key], table.num_rows)
            continue
        if dtype == "image":
            raise ValueError(f"Image features are not supported by this rebuild tool: {key}")
        if key not in table.column_names:
            raise KeyError(f"Missing parquet feature declared in info.json: {key}")
        column = table.column(key)
        if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type):
            array = np.asarray(column.to_pylist())
        else:
            array = column.to_numpy(zero_copy_only=False)
        stats[key] = _feature_stats(array)
    return stats


def _update_readme(readme_path: Path, info: dict[str, Any]) -> None:
    if not readme_path.exists():
        return
    content = readme_path.read_text(encoding="utf-8")
    marker = "[meta/info.json](meta/info.json):\n```json\n"
    start = content.find(marker)
    if start < 0:
        return
    json_start = start + len(marker)
    json_end = content.find("\n```", json_start)
    if json_end < 0:
        return
    replacement = json.dumps(info, ensure_ascii=False, indent=4)
    readme_path.write_text(content[:json_start] + replacement + content[json_end:], encoding="utf-8")


def _parse_episode_set(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def rebuild_dataset(source: Path, output: Path, drop_first_frame_episodes: set[int]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_info = _load_json(source / "meta" / "info.json")
    if source_info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {source_info.get('codebase_version')}")
    fps = int(source_info["fps"])
    features = source_info["features"]
    video_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    episode_rows = sorted(_load_jsonl(source / "meta" / "episodes.jsonl"), key=lambda row: row["episode_index"])
    episode_indices = [int(row["episode_index"]) for row in episode_rows]
    if episode_indices != list(range(len(episode_rows))):
        raise ValueError(f"Episode indices must be contiguous from zero, got {episode_indices[:5]}...")
    unknown = drop_first_frame_episodes - set(episode_indices)
    if unknown:
        raise ValueError(f"Unknown episode indices requested for trimming: {sorted(unknown)}")

    staging_output = output.with_name(output.name + ".building")
    if staging_output.exists():
        raise FileExistsError(f"Staging output already exists: {staging_output}")
    print(f"Copying {source} -> {staging_output}", flush=True)
    shutil.copytree(source, staging_output)

    rebuilt_episode_rows: list[dict[str, Any]] = []
    rewritten_tables: dict[int, pa.Table] = {}
    success_by_episode: dict[int, bool] = {}
    global_start = 0
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        chunk = episode_index // int(source_info["chunks_size"])
        parquet_path = staging_output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        table, success = _rewrite_episode_table(
            parquet_path,
            episode_index=episode_index,
            global_start=global_start,
            fps=fps,
            drop_first_frame=episode_index in drop_first_frame_episodes,
        )
        rewritten_tables[episode_index] = table
        success_by_episode[episode_index] = success
        rebuilt_episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": row["tasks"],
                "length": table.num_rows,
            }
        )
        global_start += table.num_rows

    for episode_index in sorted(drop_first_frame_episodes):
        chunk = episode_index // int(source_info["chunks_size"])
        for video_key in video_keys:
            video_path = (
                staging_output
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            print(f"Trimming first frame: episode={episode_index} camera={video_key}", flush=True)
            _trim_video_first_frame(
                video_path,
                fps,
                expected_length=rewritten_tables[episode_index].num_rows,
            )

    episode_stats_rows: list[dict[str, Any]] = []
    for row in rebuilt_episode_rows:
        episode_index = int(row["episode_index"])
        chunk = episode_index // int(source_info["chunks_size"])
        video_paths = {
            video_key: (
                staging_output
                / "videos"
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            for video_key in video_keys
        }
        print(f"Computing stats: episode={episode_index}", flush=True)
        stats = _table_stats(rewritten_tables[episode_index], features, video_paths)
        episode_stats_rows.append({"episode_index": episode_index, "stats": stats})

    info = dict(source_info)
    info["total_episodes"] = len(rebuilt_episode_rows)
    info["total_frames"] = global_start
    info["total_videos"] = len(rebuilt_episode_rows) * len(video_keys)
    info["total_chunks"] = math.ceil(len(rebuilt_episode_rows) / int(info["chunks_size"]))
    info["splits"] = {"train": f"0:{len(rebuilt_episode_rows)}"}

    _write_json(staging_output / "meta" / "info.json", info)
    _write_jsonl(staging_output / "meta" / "episodes.jsonl", rebuilt_episode_rows)
    _write_jsonl(staging_output / "meta" / "episodes_stats.jsonl", episode_stats_rows)
    _update_readme(staging_output / "README.md", info)
    os.replace(staging_output, output)

    successful = sum(success_by_episode.values())
    print(
        f"Done: episodes={len(rebuilt_episode_rows)} frames={global_start} "
        f"videos={info['total_videos']} success_labels={successful}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--drop-first-frame-episodes",
        default="",
        help="Comma-separated episode indices whose first synchronized sample should be removed.",
    )
    args = parser.parse_args()
    rebuild_dataset(
        args.source.expanduser().resolve(),
        args.output.expanduser().resolve(),
        _parse_episode_set(args.drop_first_frame_episodes),
    )


if __name__ == "__main__":
    main()

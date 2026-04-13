from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload


RUNNING_ACTION_ID = 10  # run/jog


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AVA run/jog annotations into behavior-training JSONL tracks."
    )
    parser.add_argument(
        "--train-csv",
        default=(
            "data/external/ava/train/research/action_recognition/ava/website/www/download/"
            "ava_train_v2.2.csv"
        ),
    )
    parser.add_argument(
        "--val-csv",
        default=(
            "data/external/ava/val/research/action_recognition/ava/website/www/download/"
            "ava_val_v2.2.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-track-frames", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--max-windows-per-track", type=int, default=6)
    return parser.parse_args()


def _load_rows(csv_path: Path) -> dict[tuple[str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], dict[int, dict[str, object]]] = defaultdict(dict)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 8:
                continue
            video_id = row[0]
            timestamp = int(row[1])
            x1 = float(row[2])
            y1 = float(row[3])
            x2 = float(row[4])
            y2 = float(row[5])
            action_id = int(row[6])
            person_id = int(row[7])
            key = (video_id, person_id)
            frame_entry = grouped[key].setdefault(
                timestamp,
                {
                    "timestamp": timestamp,
                    "box": [x1, y1, x2, y2],
                    "action_ids": set(),
                },
            )
            frame_entry["action_ids"].add(action_id)

    output: dict[tuple[str, int], list[dict[str, object]]] = {}
    for key, timestamp_map in grouped.items():
        output[key] = [timestamp_map[t] for t in sorted(timestamp_map)]
    return output


def _split_running_segments(
    rows: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    running_rows = [row for row in rows if RUNNING_ACTION_ID in row["action_ids"]]
    if not running_rows:
        return []

    segments: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in running_rows:
        if not current:
            current = [row]
            continue
        if int(row["timestamp"]) - int(current[-1]["timestamp"]) <= 1:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
    if current:
        segments.append(current)
    return segments


def _segment_to_rows(
    video_id: str,
    person_id: int,
    segment: list[dict[str, object]],
    *,
    split: str,
    window_size: int,
    window_stride: int,
    max_windows_per_track: int,
    min_track_frames: int,
) -> list[dict[str, object]]:
    if len(segment) < int(min_track_frames):
        return []

    timestamps = [int(row["timestamp"]) for row in segment]
    boxes = [[float(v) for v in row["box"]] for row in segment]
    centers = [[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in boxes]

    if len(segment) <= int(window_size):
        windows = [(0, len(segment))]
    else:
        last_start = len(segment) - int(window_size)
        starts = list(range(0, last_start + 1, int(window_stride)))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows = [(start, start + int(window_size)) for start in starts[: int(max_windows_per_track)]]

    results: list[dict[str, object]] = []
    for window_idx, (start_idx, end_idx) in enumerate(windows):
        window_frames = timestamps[start_idx:end_idx]
        window_boxes = boxes[start_idx:end_idx]
        window_centers = centers[start_idx:end_idx]
        speeds = []
        for idx in range(1, len(window_centers)):
            dx = window_centers[idx][0] - window_centers[idx - 1][0]
            dy = window_centers[idx][1] - window_centers[idx - 1][1]
            speeds.append((dx * dx + dy * dy) ** 0.5)
        trajectory = {
            "frames": window_frames,
            "boxes": window_boxes,
            "centers": window_centers,
            "speeds": speeds,
        }
        feature_dict = features_from_trajectory_payload(trajectory)
        if feature_dict is None:
            continue
        results.append(
            {
                "sample_id": f"ava_{video_id}_p{person_id}_s{window_frames[0]}_{window_frames[-1]}_w{window_idx}",
                "source_track_id": f"ava_{video_id}:{person_id}",
                "sequence_id": video_id,
                "track_id": int(person_id),
                "start_frame": int(window_frames[0]),
                "end_frame": int(window_frames[-1]),
                "frame_count": len(window_frames),
                "primary_label": "running",
                "pseudo_labels": ["running"],
                "label_reasons": ["official_ava_run_jog"],
                "features": feature_dict,
                "trajectory": trajectory,
                "dataset_name": "ava_running",
                "split": split,
            }
        )
    return results


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_split(
    csv_path: Path,
    *,
    split: str,
    window_size: int,
    window_stride: int,
    max_windows_per_track: int,
    min_track_frames: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    grouped = _load_rows(csv_path)
    rows: list[dict[str, object]] = []
    stats = {
        "tracks_total": 0,
        "running_segments": 0,
        "skipped_short": 0,
    }
    for (video_id, person_id), person_rows in grouped.items():
        stats["tracks_total"] += 1
        segments = _split_running_segments(person_rows)
        stats["running_segments"] += len(segments)
        for segment in segments:
            if len(segment) < int(min_track_frames):
                stats["skipped_short"] += 1
                continue
            rows.extend(
                _segment_to_rows(
                    video_id,
                    person_id,
                    segment,
                    split=split,
                    window_size=window_size,
                    window_stride=window_stride,
                    max_windows_per_track=max_windows_per_track,
                    min_track_frames=min_track_frames,
                )
            )
    return rows, stats


def main() -> None:
    args = _parse_args()
    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_stats = _build_split(
        train_csv,
        split="train",
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        max_windows_per_track=int(args.max_windows_per_track),
        min_track_frames=int(args.min_track_frames),
    )
    val_rows, val_stats = _build_split(
        val_csv,
        split="val",
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        max_windows_per_track=int(args.max_windows_per_track),
        min_track_frames=int(args.min_track_frames),
    )

    rng = random.Random(int(args.seed))
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    _write_jsonl(train_dir / "tracks.jsonl", train_rows)
    _write_jsonl(val_dir / "tracks.jsonl", val_rows)

    summary = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "output_dir": str(output_dir),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "label_counts": {
            "train": dict(Counter(row["primary_label"] for row in train_rows)),
            "val": dict(Counter(row["primary_label"] for row in val_rows)),
        },
        "config": {
            "seed": int(args.seed),
            "min_track_frames": int(args.min_track_frames),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "max_windows_per_track": int(args.max_windows_per_track),
        },
        "stats": {
            "train": train_stats,
            "val": val_stats,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

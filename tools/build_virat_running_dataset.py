from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload


RUNNING_EVENT_TYPE = 10
PERSON_OBJECT_TYPE = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VIRAT running events into behavior-training JSONL tracks."
    )
    parser.add_argument(
        "--annotations-dir",
        default="data/external/virat/annotations-and-docs/annotations",
        help=(
            "Directory containing either VIRAT Release2 *.events/*.mapping/*.objects files "
            "or public DIVA *.activities.yml/*.geom.yml/*.types.yml files."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-track-frames", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=6)
    parser.add_argument("--max-windows-per-track", type=int, default=4)
    return parser.parse_args()


_DIVA_TYPE_RE = re.compile(r"id1:\s*(\d+)\s*,\s*cset3:\s*\{\s*([A-Za-z_]+):")
_DIVA_GEOM_RE = re.compile(
    r"id1:\s*(\d+),\s*id0:\s*\d+,\s*ts0:\s*(\d+),.*?g0:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*,"
)
_DIVA_RUNNING_RE = re.compile(
    r"activity_running:\s*1\.0.*?timespan:\s*\[\{tsr0:\s*\[(\d+),\s*(\d+)\]\}\].*?actors:\s*\[(.*)\]\s*\}\s*\}\s*$"
)
_DIVA_ACTOR_RE = re.compile(r"id1:\s*(\d+)")


def _load_object_tracks(object_path: Path) -> dict[int, dict[str, object]]:
    tracks: dict[int, dict[str, object]] = {}
    with object_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            object_id = int(parts[0])
            frame_idx = int(parts[2])
            left = float(parts[3])
            top = float(parts[4])
            width = float(parts[5])
            height = float(parts[6])
            object_type = int(parts[7])
            track = tracks.setdefault(
                object_id,
                {
                    "object_type": object_type,
                    "frames": [],
                    "boxes": [],
                    "centers": [],
                },
            )
            x2 = left + width
            y2 = top + height
            track["frames"].append(frame_idx)
            track["boxes"].append([left, top, x2, y2])
            track["centers"].append([left + width / 2.0, top + height / 2.0])
    return tracks


def _load_running_events(mapping_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            event_type = int(parts[1])
            if event_type != RUNNING_EVENT_TYPE:
                continue
            event_id = int(parts[0])
            start_frame = int(parts[3])
            end_frame = int(parts[4])
            flags = [int(value) for value in parts[6:]]
            object_ids = [idx + 1 for idx, flag in enumerate(flags) if flag == 1]
            if not object_ids:
                continue
            events.append(
                {
                    "event_id": event_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "object_ids": object_ids,
                }
            )
    return events


def _load_diva_person_ids(types_path: Path) -> set[int]:
    person_ids: set[int] = set()
    with types_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _DIVA_TYPE_RE.search(line)
            if not match:
                continue
            object_id = int(match.group(1))
            label = match.group(2)
            if label == "Person":
                person_ids.add(object_id)
    return person_ids


def _load_diva_tracks(geom_path: Path, person_ids: set[int]) -> dict[int, dict[str, object]]:
    tracks: dict[int, dict[str, object]] = {}
    with geom_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _DIVA_GEOM_RE.search(line)
            if not match:
                continue
            object_id = int(match.group(1))
            if object_id not in person_ids:
                continue
            frame_idx = int(match.group(2))
            x1 = float(match.group(3))
            y1 = float(match.group(4))
            x2 = float(match.group(5))
            y2 = float(match.group(6))
            track = tracks.setdefault(
                object_id,
                {
                    "object_type": PERSON_OBJECT_TYPE,
                    "frames": [],
                    "boxes": [],
                    "centers": [],
                },
            )
            track["frames"].append(frame_idx)
            track["boxes"].append([x1, y1, x2, y2])
            track["centers"].append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    return tracks


def _load_diva_running_events(activities_path: Path, person_ids: set[int]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    next_event_id = 1
    with activities_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "activity_running" not in line:
                continue
            match = _DIVA_RUNNING_RE.search(line.strip())
            if not match:
                continue
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            actor_blob = match.group(3)
            object_ids = sorted(
                {
                    int(actor_id)
                    for actor_id in _DIVA_ACTOR_RE.findall(actor_blob)
                    if int(actor_id) in person_ids
                }
            )
            if not object_ids:
                continue
            events.append(
                {
                    "event_id": next_event_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "object_ids": object_ids,
                }
            )
            next_event_id += 1
    return events


def _slice_track(
    clip_id: str,
    event: dict[str, object],
    object_id: int,
    track: dict[str, object],
    *,
    window_size: int,
    window_stride: int,
    max_windows_per_track: int,
    min_track_frames: int,
) -> list[dict[str, object]]:
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])

    segment_frames = []
    segment_boxes = []
    segment_centers = []
    for frame_idx, box, center in zip(track["frames"], track["boxes"], track["centers"]):
        if start_frame <= int(frame_idx) <= end_frame:
            segment_frames.append(int(frame_idx))
            segment_boxes.append([float(value) for value in box])
            segment_centers.append([float(value) for value in center])

    if len(segment_frames) < int(min_track_frames):
        return []

    segment_speeds = []
    for idx in range(1, len(segment_centers)):
        dx = segment_centers[idx][0] - segment_centers[idx - 1][0]
        dy = segment_centers[idx][1] - segment_centers[idx - 1][1]
        segment_speeds.append((dx * dx + dy * dy) ** 0.5)

    if len(segment_frames) <= int(window_size):
        windows = [(0, len(segment_frames))]
    else:
        last_start = len(segment_frames) - int(window_size)
        starts = list(range(0, last_start + 1, int(window_stride)))
        if starts[-1] != last_start:
            starts.append(last_start)
        windows = [(start, start + int(window_size)) for start in starts[: int(max_windows_per_track)]]

    rows = []
    for window_idx, (start_idx, end_idx) in enumerate(windows):
        centers = segment_centers[start_idx:end_idx]
        boxes = segment_boxes[start_idx:end_idx]
        frames = segment_frames[start_idx:end_idx]
        speeds = []
        for idx in range(1, len(centers)):
            dx = centers[idx][0] - centers[idx - 1][0]
            dy = centers[idx][1] - centers[idx - 1][1]
            speeds.append((dx * dx + dy * dy) ** 0.5)
        trajectory = {
            "frames": frames,
            "boxes": boxes,
            "centers": centers,
            "speeds": speeds,
        }
        feature_dict = features_from_trajectory_payload(trajectory)
        if feature_dict is None:
            continue
        rows.append(
            {
                "sample_id": f"virat_{clip_id}_event{event['event_id']}_obj{object_id}_w{window_idx}",
                "source_track_id": f"virat_{clip_id}:{object_id}",
                "sequence_id": clip_id,
                "track_id": int(object_id),
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "frame_count": len(frames),
                "primary_label": "running",
                "pseudo_labels": ["running"],
                "label_reasons": ["official_virat_running_event"],
                "features": feature_dict,
                "trajectory": trajectory,
                "dataset_name": "virat_running",
                "split": "",
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    annotations_dir = Path(args.annotations_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    release2_mode = any(annotations_dir.glob("*.viratdata.mapping.txt"))
    diva_mode = not release2_mode and (
        (annotations_dir / "train").is_dir() or (annotations_dir / "validate").is_dir()
    )
    if not release2_mode and not diva_mode:
        raise FileNotFoundError(
            f"Unsupported VIRAT annotation layout under {annotations_dir}"
        )

    if release2_mode:
        clip_ids = sorted(
            path.name.replace(".viratdata.mapping.txt", "")
            for path in annotations_dir.glob("*.viratdata.mapping.txt")
            if path.is_file()
        )
        rng = random.Random(int(args.seed))
        shuffled_clip_ids = list(clip_ids)
        rng.shuffle(shuffled_clip_ids)
        val_count = max(1, int(round(len(shuffled_clip_ids) * float(args.val_ratio))))
        val_clip_ids = set(shuffled_clip_ids[:val_count])
    else:
        train_dir_src = annotations_dir / "train"
        val_dir_src = annotations_dir / "validate"
        train_clip_ids = sorted(
            path.name.replace(".activities.yml", "")
            for path in train_dir_src.glob("*.activities.yml")
            if path.is_file()
        )
        val_clip_ids = set(
            path.name.replace(".activities.yml", "")
            for path in val_dir_src.glob("*.activities.yml")
            if path.is_file()
        )
        clip_ids = train_clip_ids + sorted(val_clip_ids)

    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    skipped_non_person = 0
    skipped_short = 0
    running_event_count = 0

    for clip_id in clip_ids:
        if release2_mode:
            mapping_path = annotations_dir / f"{clip_id}.viratdata.mapping.txt"
            object_path = annotations_dir / f"{clip_id}.viratdata.objects.txt"
            if (
                not mapping_path.exists()
                or not object_path.exists()
                or not mapping_path.is_file()
                or not object_path.is_file()
            ):
                continue
            tracks = _load_object_tracks(object_path)
            events = _load_running_events(mapping_path)
            target_rows = val_rows if clip_id in val_clip_ids else train_rows
            target_split = "val" if clip_id in val_clip_ids else "train"
        else:
            source_dir = annotations_dir / ("validate" if clip_id in val_clip_ids else "train")
            activities_path = source_dir / f"{clip_id}.activities.yml"
            geom_path = source_dir / f"{clip_id}.geom.yml"
            types_path = source_dir / f"{clip_id}.types.yml"
            if not activities_path.exists() or not geom_path.exists() or not types_path.exists():
                continue
            person_ids = _load_diva_person_ids(types_path)
            tracks = _load_diva_tracks(geom_path, person_ids)
            events = _load_diva_running_events(activities_path, person_ids)
            target_rows = val_rows if clip_id in val_clip_ids else train_rows
            target_split = "val" if clip_id in val_clip_ids else "train"

        running_event_count += len(events)

        for event in events:
            for object_id in event["object_ids"]:
                track = tracks.get(int(object_id))
                if track is None or int(track["object_type"]) != PERSON_OBJECT_TYPE:
                    skipped_non_person += 1
                    continue
                windows = _slice_track(
                    clip_id,
                    event,
                    int(object_id),
                    track,
                    window_size=int(args.window_size),
                    window_stride=int(args.window_stride),
                    max_windows_per_track=int(args.max_windows_per_track),
                    min_track_frames=int(args.min_track_frames),
                )
                if not windows:
                    skipped_short += 1
                    continue
                for row in windows:
                    row["split"] = target_split
                    target_rows.append(row)

    _write_jsonl(train_dir / "tracks.jsonl", train_rows)
    _write_jsonl(val_dir / "tracks.jsonl", val_rows)

    summary = {
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "source_format": "release2" if release2_mode else "diva_public",
        "clip_count": len(clip_ids),
        "train_clip_count": len(clip_ids) - len(val_clip_ids),
        "val_clip_count": len(val_clip_ids),
        "running_event_count": int(running_event_count),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "label_counts": {
            "train": dict(Counter(row["primary_label"] for row in train_rows)),
            "val": dict(Counter(row["primary_label"] for row in val_rows)),
        },
        "config": {
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "min_track_frames": int(args.min_track_frames),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "max_windows_per_track": int(args.max_windows_per_track),
        },
        "skipped_non_person_or_missing": int(skipped_non_person),
        "skipped_short": int(skipped_short),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a binary normal/anomaly trajectory dataset from UBnormal."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--object-names-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--min-anomaly-frames", type=int, default=6)
    parser.add_argument("--min-anomaly-ratio", type=float, default=0.2)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_non_person_ids(path: Path) -> dict[str, dict[int, str]]:
    raw = pickle.loads(path.read_bytes())
    result: dict[str, dict[int, str]] = {}
    for video_name, mapping in raw.items():
        result[str(video_name)] = {int(k): str(v) for k, v in dict(mapping).items()}
    return result


def _load_track_intervals(track_path: Path | None) -> dict[int, list[tuple[int, int]]]:
    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if track_path is None or not track_path.exists():
        return intervals
    for line in track_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        object_id, start_frame, end_frame = (int(float(value)) for value in parts)
        intervals[object_id].append((start_frame, end_frame))
    return intervals


def _is_anomalous_frame(intervals: dict[int, list[tuple[int, int]]], object_id: int, frame_idx: int) -> bool:
    for start_frame, end_frame in intervals.get(object_id, []):
        if start_frame <= frame_idx <= end_frame:
            return True
    return False


def _compute_boxes_for_ids(mask: np.ndarray, object_ids: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for object_id in object_ids:
        ys, xs = np.where(mask == object_id)
        if xs.size == 0 or ys.size == 0:
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        boxes[int(object_id)] = (x1, y1, x2, y2)
    return boxes


def _build_track_rows(
    video_row: dict[str, object],
    non_person_ids: dict[str, dict[int, str]],
    min_frames: int,
    min_anomaly_frames: int,
    min_anomaly_ratio: float,
) -> list[dict[str, object]]:
    video_name = str(video_row["video_name"])
    annotation_dir = Path(video_row["annotation_dir"])
    track_path = Path(video_row["track_path"]) if video_row.get("track_path") else None
    non_person = set(non_person_ids.get(video_name, {}).keys())
    anomaly_intervals = _load_track_intervals(track_path)

    tracks: dict[int, dict[str, object]] = {}
    mask_paths = sorted(annotation_dir.glob("*_gt.png"))

    for frame_idx, mask_path in enumerate(mask_paths):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids > 0]
        object_ids = np.asarray([int(obj_id) for obj_id in object_ids if int(obj_id) not in non_person], dtype=np.int32)
        if object_ids.size == 0:
            continue
        boxes = _compute_boxes_for_ids(mask, object_ids)
        for object_id in object_ids.tolist():
            box = boxes.get(object_id)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            track = tracks.setdefault(
                object_id,
                {
                    "frames": [],
                    "boxes": [],
                    "centers": [],
                    "anomaly_flags": [],
                },
            )
            track["frames"].append(frame_idx)
            track["boxes"].append([x1, y1, x2, y2])
            track["centers"].append(center)
            track["anomaly_flags"].append(_is_anomalous_frame(anomaly_intervals, object_id, frame_idx))

    rows: list[dict[str, object]] = []
    for object_id, track in tracks.items():
        frames = list(track["frames"])
        centers = list(track["centers"])
        if len(frames) < min_frames:
            continue
        speeds = []
        for idx in range(1, len(centers)):
            dx = float(centers[idx][0] - centers[idx - 1][0])
            dy = float(centers[idx][1] - centers[idx - 1][1])
            speeds.append(float((dx * dx + dy * dy) ** 0.5))

        feature_dict = features_from_trajectory_payload(
            {
                "centers": centers,
                "speeds": speeds,
            },
            high_speed_threshold=15.57,
            low_speed_threshold=2.2,
        )
        if feature_dict is None:
            continue

        anomaly_flags = list(track["anomaly_flags"])
        anomaly_frames = int(sum(1 for flag in anomaly_flags if flag))
        anomaly_ratio = float(anomaly_frames / max(len(anomaly_flags), 1))
        label = (
            "anomaly"
            if anomaly_frames >= min_anomaly_frames and anomaly_ratio >= min_anomaly_ratio
            else "normal"
        )

        rows.append(
            {
                "sample_id": f"{video_name}_obj_{object_id}",
                "source_track_id": f"{video_name}:{object_id}",
                "sequence_id": video_name,
                "track_id": int(object_id),
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "frame_count": len(frames),
                "primary_label": label,
                "pseudo_labels": [label],
                "label_reasons": [
                    f"ubnormal_track anomaly_frames={anomaly_frames} anomaly_ratio={anomaly_ratio:.3f}"
                ],
                "features": {
                    **{key: round(float(value), 4) for key, value in feature_dict.items()},
                    "anomaly_frames": anomaly_frames,
                    "anomaly_ratio": round(anomaly_ratio, 4),
                },
                "trajectory": {
                    "frames": frames,
                    "boxes": list(track["boxes"]),
                    "centers": centers,
                    "speeds": [round(float(value), 3) for value in speeds],
                    "anomaly_flags": anomaly_flags,
                },
                "video_name": video_name,
                "scene": video_row["scene"],
                "split": video_row["split"],
                "video_path": video_row["video_path"],
                "annotation_dir": video_row["annotation_dir"],
                "track_path": video_row.get("track_path"),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    object_names_pkl = Path(args.object_names_pkl)
    output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains files. Pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(manifest_path)
    if args.max_videos is not None:
        rows = rows[: args.max_videos]

    non_person_ids = _load_non_person_ids(object_names_pkl)
    final_rows: list[dict[str, object]] = []
    for row in rows:
        final_rows.extend(
            _build_track_rows(
                row,
                non_person_ids=non_person_ids,
                min_frames=args.min_frames,
                min_anomaly_frames=args.min_anomaly_frames,
                min_anomaly_ratio=args.min_anomaly_ratio,
            )
        )

    label_counts = defaultdict(int)
    for row in final_rows:
        label_counts[str(row["primary_label"])] += 1

    _write_jsonl(output_dir / "tracks.jsonl", final_rows)
    summary = {
        "manifest_path": str(manifest_path),
        "object_names_pkl": str(object_names_pkl),
        "output_dir": str(output_dir),
        "videos_processed": len(rows),
        "samples_total": len(final_rows),
        "label_counts": dict(label_counts),
        "config": {
            "min_frames": int(args.min_frames),
            "min_anomaly_frames": int(args.min_anomaly_frames),
            "min_anomaly_ratio": float(args.min_anomaly_ratio),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

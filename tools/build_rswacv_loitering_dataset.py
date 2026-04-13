from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the RS-WACV24 loitering dataset into behavior-training JSONL tracks."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/external/loitering_github_probe/dataset",
        help="Path to the RS-WACV24 dataset directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which official split to export.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tracks.jsonl and summary.json will be written.",
    )
    parser.add_argument("--min-frames", type=int, default=24)
    return parser.parse_args()


def _load_labels(csv_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            track_id = str(row["id"]).strip()
            labels[track_id] = int(row["loitering"])
    return labels


def _load_tracks(pickle_path: Path) -> dict[str, object]:
    with pickle_path.open("rb") as handle:
        return pickle.load(handle)


def _trajectory_payload(track_array: object) -> dict[str, object]:
    centers = [[float(point[0]), float(point[1])] for point in track_array]
    speeds = []
    for idx in range(1, len(centers)):
        dx = centers[idx][0] - centers[idx - 1][0]
        dy = centers[idx][1] - centers[idx - 1][1]
        speeds.append((dx * dx + dy * dy) ** 0.5)
    return {
        "frames": list(range(len(centers))),
        "centers": centers,
        "speeds": speeds,
    }


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(dataset_root / f"{args.split}_df.csv")
    tracks = _load_tracks(dataset_root / f"{args.split}_dict_smooth.pkl")

    rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    skipped_short = 0
    missing_tracks = 0

    for track_id, label_value in labels.items():
        track = tracks.get(int(track_id)) if int(track_id) in tracks else tracks.get(track_id)
        if track is None:
            missing_tracks += 1
            continue
        if len(track) < int(args.min_frames):
            skipped_short += 1
            continue

        trajectory = _trajectory_payload(track)
        feature_dict = features_from_trajectory_payload(trajectory)
        if feature_dict is None:
            continue

        primary_label = "loitering" if int(label_value) == 1 else "normal"
        row = {
            "sample_id": f"rswacv_{args.split}_{track_id}",
            "source_track_id": f"rswacv_{args.split}:{track_id}",
            "sequence_id": f"rswacv_{args.split}",
            "track_id": int(track_id),
            "start_frame": 0,
            "end_frame": len(trajectory["frames"]) - 1,
            "frame_count": len(trajectory["frames"]),
            "primary_label": primary_label,
            "pseudo_labels": [primary_label],
            "label_reasons": ["official_rswacv24_loitering_label"],
            "features": feature_dict,
            "trajectory": trajectory,
            "dataset_name": "rswacv24_loitering",
            "split": args.split,
        }
        rows.append(row)
        label_counts[primary_label] += 1

    tracks_path = output_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "output_dir": str(output_dir),
        "total_label_rows": len(labels),
        "total_track_arrays": len(tracks),
        "samples_total": len(rows),
        "label_counts": dict(label_counts),
        "min_frames": int(args.min_frames),
        "skipped_short": int(skipped_short),
        "missing_tracks": int(missing_tracks),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

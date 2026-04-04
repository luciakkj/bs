from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload
from training.config import resolve_path


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_window(sample: dict[str, object], start_idx: int, end_idx: int) -> dict[str, object]:
    trajectory = sample["trajectory"]
    frames = trajectory["frames"][start_idx:end_idx]
    boxes = trajectory["boxes"][start_idx:end_idx]
    centers = trajectory["centers"][start_idx:end_idx]
    overlap_flags = trajectory["mask_overlap_flags"][start_idx:end_idx]
    overlap_ratios = trajectory["mask_overlap_ratios"][start_idx:end_idx]

    speeds = []
    for idx in range(1, len(centers)):
        prev = centers[idx - 1]
        curr = centers[idx]
        speeds.append(_distance((prev[0], prev[1]), (curr[0], curr[1])))

    xs = [point[0] for point in centers]
    ys = [point[1] for point in centers]
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    path_length = sum(
        _distance((centers[idx - 1][0], centers[idx - 1][1]), (centers[idx][0], centers[idx][1]))
        for idx in range(1, len(centers))
    )
    displacement = _distance((centers[0][0], centers[0][1]), (centers[-1][0], centers[-1][1])) if len(centers) >= 2 else 0.0
    movement_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    centroid_radius = max(
        (_distance((point[0], point[1]), centroid) for point in centers),
        default=0.0,
    )
    support_frames = sum(1 for flag in overlap_flags if flag)
    support_ratio = _safe_divide(support_frames, len(centers))
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds, default=0.0)
    p90_speed = float(np.percentile(speeds, 90)) if speeds else 0.0
    high_speed_ratio = _safe_divide(
        sum(1 for speed in speeds if speed >= 15.57),
        len(speeds),
    )
    derived_feature_dict = features_from_trajectory_payload(
        {
            "centers": centers,
            "speeds": speeds,
        },
        high_speed_threshold=15.57,
        low_speed_threshold=2.2,
    ) or {}

    return {
        "sample_id": f"{sample['sample_id']}_w_{start_idx}_{end_idx}",
        "source_track_id": sample.get("source_track_id", sample["sample_id"]),
        "sequence_id": sample["sequence_id"],
        "track_id": sample["track_id"],
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "frame_count": len(frames),
        "primary_label": "unknown",
        "pseudo_labels": [],
        "label_reasons": [],
        "features": {
            "mean_confidence": float(sample["features"].get("mean_confidence", 0.0)),
            "avg_speed": round(avg_speed, 4),
            "speed_std": round(float(derived_feature_dict.get("speed_std", 0.0)), 4),
            "max_speed": round(max_speed, 4),
            "p90_speed": round(p90_speed, 4),
            "high_speed_ratio": round(high_speed_ratio, 4),
            "stationary_ratio": round(float(derived_feature_dict.get("stationary_ratio", 0.0)), 4),
            "path_length": round(path_length, 4),
            "displacement": round(displacement, 4),
            "movement_extent": round(movement_extent, 4),
            "centroid_radius": round(centroid_radius, 4),
            "straightness": round(float(derived_feature_dict.get("straightness", 0.0)), 4),
            "direction_change_rate": round(float(derived_feature_dict.get("direction_change_rate", 0.0)), 4),
            "mean_turn_angle": round(float(derived_feature_dict.get("mean_turn_angle", 0.0)), 4),
            "revisit_ratio": round(float(derived_feature_dict.get("revisit_ratio", 0.0)), 4),
            "unique_cell_ratio": round(float(derived_feature_dict.get("unique_cell_ratio", 0.0)), 4),
            "max_cell_occupancy_ratio": round(float(derived_feature_dict.get("max_cell_occupancy_ratio", 0.0)), 4),
            "support_frames": int(support_frames),
            "support_ratio": round(support_ratio, 4),
            "max_support_run": int(support_frames),
            "gt_anomaly_frames": int(sum(1 for ratio in overlap_ratios if ratio > 0.0)),
            "mean_overlap_ratio": round(sum(overlap_ratios) / len(overlap_ratios), 4),
            "max_overlap_ratio": round(max(overlap_ratios, default=0.0), 4),
        },
        "trajectory": {
            "frames": frames,
            "boxes": boxes,
            "centers": centers,
            "speeds": [round(float(value), 3) for value in speeds],
            "mask_overlap_flags": overlap_flags,
            "mask_overlap_ratios": overlap_ratios,
        },
    }


def _label_augmented_sample(sample: dict[str, object]) -> dict[str, object]:
    frame_count = int(sample["frame_count"])
    features = sample["features"]
    support_frames = int(features["support_frames"])
    support_ratio = float(features["support_ratio"])
    avg_speed = float(features["avg_speed"])
    p90_speed = float(features["p90_speed"])
    movement_extent = float(features["movement_extent"])
    centroid_radius = float(features["centroid_radius"])

    label = "unknown"
    reasons = []
    if (
        frame_count >= 12
        and support_frames >= 6
        and support_ratio >= 0.35
        and p90_speed >= 14.0
        and avg_speed >= 8.0
        and movement_extent >= 120.0
    ):
        label = "running"
        reasons.append(
            f"expanded_running avg_speed={avg_speed:.3f} p90_speed={p90_speed:.3f} support_ratio={support_ratio:.3f}"
        )
    elif (
        frame_count >= 48
        and support_frames >= 12
        and support_ratio >= 0.2
        and avg_speed <= 2.2
        and movement_extent <= 55.0
        and centroid_radius <= 28.0
    ):
        label = "loitering"
        reasons.append(
            f"expanded_loitering avg_speed={avg_speed:.3f} movement_extent={movement_extent:.3f} support_ratio={support_ratio:.3f}"
        )
    elif support_frames == 0 and frame_count >= 24:
        label = "normal"
        reasons.append("expanded_normal_no_mask_overlap")

    sample["primary_label"] = label
    sample["pseudo_labels"] = [label] if label != "unknown" else []
    sample["label_reasons"] = reasons
    return sample


def expand_behavior_dataset(
    input_path: str | Path = "output/avenue_pseudo_labels/tracks.jsonl",
    output_dir: str | Path = "output/avenue_pseudo_labels_expanded",
) -> dict[str, object]:
    input_path = resolve_path(input_path)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _load_jsonl(input_path)
    expanded_rows: list[dict[str, object]] = []

    for row in base_rows:
        row = dict(row)
        row["source_track_id"] = row.get("source_track_id", row["sample_id"])
        expanded_rows.append(row)

        label = row["primary_label"]
        frame_count = int(row["frame_count"])
        if label == "running":
            window_size = min(24, frame_count)
            stride = max(4, window_size // 3)
        elif label == "loitering":
            window_size = min(96, frame_count)
            stride = max(12, window_size // 4)
        elif label == "unknown" and int(row["features"]["support_frames"]) >= 6:
            window_size = min(24 if float(row["features"]["p90_speed"]) >= 14.0 else 72, frame_count)
            stride = max(4, window_size // 3)
        else:
            continue

        if frame_count < max(12, window_size):
            continue

        last_start = max(frame_count - window_size, 0)
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)

        for start_idx in starts:
            end_idx = start_idx + window_size
            if end_idx > frame_count:
                continue
            window_row = _extract_window(row, start_idx, end_idx)
            window_row["source_track_id"] = row["source_track_id"]

            if label in {"running", "loitering"}:
                window_row["primary_label"] = label
                window_row["pseudo_labels"] = [label]
                window_row["label_reasons"] = [f"window_from_{label}_track"]
                expanded_rows.append(window_row)
            elif label == "unknown":
                window_row = _label_augmented_sample(window_row)
                if window_row["primary_label"] != "unknown":
                    expanded_rows.append(window_row)

    deduped = []
    seen_ids = set()
    for row in expanded_rows:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        deduped.append(row)

    summary_counts = {}
    for row in deduped:
        summary_counts[row["primary_label"]] = summary_counts.get(row["primary_label"], 0) + 1

    tracks_path = output_dir / "tracks_expanded.jsonl"
    _write_jsonl(tracks_path, deduped)
    for label in ("running", "loitering", "normal", "unknown"):
        _write_jsonl(
            output_dir / f"{label}_expanded.jsonl",
            [row for row in deduped if row["primary_label"] == label],
        )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "summary": {
            "base_tracks": len(base_rows),
            "expanded_tracks": len(deduped),
            "label_counts": summary_counts,
        },
        "artifacts": {
            "tracks_expanded": str(tracks_path),
            "running_expanded": str(output_dir / "running_expanded.jsonl"),
            "loitering_expanded": str(output_dir / "loitering_expanded.jsonl"),
            "normal_expanded": str(output_dir / "normal_expanded.jsonl"),
            "unknown_expanded": str(output_dir / "unknown_expanded.jsonl"),
        },
        "notes": [
            "Expanded samples include positive windows sliced from long running/loitering tracks.",
            "Unknown tracks with strong anomaly-mask support are re-evaluated with relaxed heuristics.",
            "source_track_id is preserved so training can split by original track rather than by window.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

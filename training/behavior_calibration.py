from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from training.config import CalibrationConfig
from training.mot_dataset import discover_sequences, load_gt_annotations


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def calibrate_behavior_thresholds(config: CalibrationConfig) -> dict[str, object]:
    cfg = config.resolved()
    sequences = discover_sequences(
        cfg.mot_root,
        split_name=cfg.split_name,
        detector_filter=cfg.detector_filter,
    )

    speed_values: list[float] = []
    avg_track_speeds: list[float] = []
    track_durations: list[int] = []
    track_spans: list[float] = []
    low_motion_durations: list[int] = []
    low_motion_spans: list[float] = []
    low_motion_avg_speeds: list[float] = []
    sequence_stats: dict[str, dict[str, int]] = {}

    for sequence in sequences:
        frame_annotations = load_gt_annotations(
            sequence,
            include_classes=cfg.include_classes,
            min_visibility=cfg.min_visibility,
        )
        tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        annotation_count = 0

        for frame_id, annotations in frame_annotations.items():
            for ann in annotations:
                center_x = float(ann["left"]) + float(ann["width"]) / 2.0
                center_y = float(ann["top"]) + float(ann["height"]) / 2.0
                tracks[int(ann["track_id"])].append((frame_id, center_x, center_y))
                annotation_count += 1

        sequence_stats[sequence.name] = {
            "tracks": len(tracks),
            "annotations": annotation_count,
        }

        for points in tracks.values():
            points.sort(key=lambda item: item[0])
            duration = len(points)
            if duration < 2:
                continue

            xs = [point[1] for point in points]
            ys = [point[2] for point in points]
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

            speeds = []
            for prev, curr in zip(points, points[1:]):
                frame_gap = max(curr[0] - prev[0], 1)
                dist = math.hypot(curr[1] - prev[1], curr[2] - prev[2])
                speeds.append(dist / frame_gap)

            if not speeds:
                continue

            avg_speed = sum(speeds) / len(speeds)
            speed_values.extend(speeds)
            avg_track_speeds.append(avg_speed)
            track_durations.append(duration)
            track_spans.append(span)

            if avg_speed <= 3.0 and span <= 80.0:
                low_motion_durations.append(duration)
                low_motion_spans.append(span)
                low_motion_avg_speeds.append(avg_speed)

    stats = {
        "total_sequences": len(sequences),
        "total_tracks": len(track_durations),
        "speed_px_per_frame": {
            "p50": round(_percentile(speed_values, 0.50), 3),
            "p90": round(_percentile(speed_values, 0.90), 3),
            "p95": round(_percentile(speed_values, 0.95), 3),
            "p97": round(_percentile(speed_values, 0.97), 3),
            "p99": round(_percentile(speed_values, 0.99), 3),
        },
        "track_duration_frames": {
            "p50": round(_percentile([float(v) for v in track_durations], 0.50), 3),
            "p75": round(_percentile([float(v) for v in track_durations], 0.75), 3),
            "p90": round(_percentile([float(v) for v in track_durations], 0.90), 3),
        },
        "avg_track_speed_px_per_frame": {
            "p50": round(_percentile(avg_track_speeds, 0.50), 3),
            "p75": round(_percentile(avg_track_speeds, 0.75), 3),
            "p90": round(_percentile(avg_track_speeds, 0.90), 3),
        },
        "track_span_pixels": {
            "p50": round(_percentile(track_spans, 0.50), 3),
            "p75": round(_percentile(track_spans, 0.75), 3),
            "p90": round(_percentile(track_spans, 0.90), 3),
        },
        "low_motion_track_count": len(low_motion_durations),
        "low_motion_duration_frames": {
            "p50": round(_percentile([float(v) for v in low_motion_durations], 0.50), 3),
            "p75": round(_percentile([float(v) for v in low_motion_durations], 0.75), 3),
            "p90": round(_percentile([float(v) for v in low_motion_durations], 0.90), 3),
        },
        "low_motion_span_pixels": {
            "p50": round(_percentile(low_motion_spans, 0.50), 3),
            "p75": round(_percentile(low_motion_spans, 0.75), 3),
            "p90": round(_percentile(low_motion_spans, 0.90), 3),
        },
        "low_motion_avg_speed_px_per_frame": {
            "p50": round(_percentile(low_motion_avg_speeds, 0.50), 3),
            "p75": round(_percentile(low_motion_avg_speeds, 0.75), 3),
            "p90": round(_percentile(low_motion_avg_speeds, 0.90), 3),
        },
    }

    suggestions = {
        "loiter_frames": int(max(45, round(_percentile([float(v) for v in low_motion_durations], 0.75)))),
        "loiter_radius": round(max(40.0, _percentile(low_motion_spans, 0.75)), 2),
        "loiter_speed": round(max(1.0, min(3.0, _percentile(low_motion_avg_speeds, 0.75))), 2),
        "running_speed": round(max(12.0, _percentile(speed_values, 0.97)), 2),
        "running_frames": 3,
    }

    report = {
        "source": {
            "mot_root": str(cfg.mot_root),
            "split_name": cfg.split_name,
            "detector_filter": cfg.detector_filter,
            "include_classes": list(cfg.include_classes),
            "min_visibility": cfg.min_visibility,
        },
        "sequence_stats": sequence_stats,
        "stats": stats,
        "suggestions": suggestions,
        "notes": [
            "ByteTrack is association-based and does not require supervised training.",
            "MOT17 does not provide explicit abnormal-event labels, so these behavior values are calibration suggestions derived from trajectory statistics.",
        ],
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

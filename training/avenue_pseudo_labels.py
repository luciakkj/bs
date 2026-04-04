from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from training.config import AvenuePseudoLabelConfig


@dataclass(slots=True)
class AvenueSequence:
    video_path: Path
    label_path: Path
    sequence_id: str


@dataclass(slots=True)
class TrackObservation:
    frame_index: int
    bbox: list[int]
    center: tuple[float, float]
    conf: float
    speed: float
    mask_overlap_ratio: float
    center_on_mask: bool
    gt_anomaly: bool


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _load_avenue_labels(label_path: Path) -> list[np.ndarray]:
    mat = loadmat(label_path)
    vol = mat["volLabel"]
    return [np.asarray(vol[0, idx], dtype=np.uint8) for idx in range(vol.shape[1])]


def _discover_sequences(cfg: AvenuePseudoLabelConfig) -> list[AvenueSequence]:
    video_dir = cfg.avenue_root / "testing_videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"Avenue testing videos not found: {video_dir}")
    if not cfg.ground_truth_root.exists():
        raise FileNotFoundError(f"Avenue ground truth masks not found: {cfg.ground_truth_root}")

    sequences = []
    selected_ids = {str(item).zfill(2) for item in cfg.sequence_ids}
    for video_path in sorted(video_dir.glob("*.avi")):
        stem = video_path.stem
        if selected_ids and stem not in selected_ids:
            continue
        label_path = cfg.ground_truth_root / f"{int(stem)}_label.mat"
        if not label_path.exists():
            continue
        sequences.append(
            AvenueSequence(
                video_path=video_path,
                label_path=label_path,
                sequence_id=stem,
            )
        )

    if cfg.max_videos is not None:
        sequences = sequences[: cfg.max_videos]
    if not sequences:
        raise FileNotFoundError("No matched Avenue testing video / label pairs were found.")
    return sequences


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _mask_positive(mask: np.ndarray) -> bool:
    return bool(np.any(mask > 0))


def _bbox_mask_support(bbox: list[int], mask: np.ndarray) -> tuple[float, bool]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), mask.shape[1] - 1))
    x2 = max(0, min(int(x2), mask.shape[1]))
    y1 = max(0, min(int(y1), mask.shape[0] - 1))
    y2 = max(0, min(int(y2), mask.shape[0]))

    if x2 <= x1 or y2 <= y1:
        return 0.0, False

    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0, False

    positive = int(np.count_nonzero(region > 0))
    overlap_ratio = _safe_divide(positive, region.size)

    cx = max(0, min(int((x1 + x2) / 2), mask.shape[1] - 1))
    cy = max(0, min(int((y1 + y2) / 2), mask.shape[0] - 1))
    center_on_mask = bool(mask[cy, cx] > 0)
    return overlap_ratio, center_on_mask


def _longest_true_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _round_list(values: list[float], digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _build_track_sample(
    cfg: AvenuePseudoLabelConfig,
    sequence_id: str,
    track_id: int,
    observations: list[TrackObservation],
) -> dict[str, object]:
    observations = sorted(observations, key=lambda item: item.frame_index)
    centers = [obs.center for obs in observations]
    bboxes = [obs.bbox for obs in observations]
    frame_indices = [obs.frame_index for obs in observations]
    confs = [obs.conf for obs in observations]
    speeds = [obs.speed for obs in observations[1:]]
    support_flags = [
        (obs.mask_overlap_ratio > 0.0) or obs.center_on_mask
        for obs in observations
    ]
    overlap_ratios = [obs.mask_overlap_ratio for obs in observations]
    gt_positive_frames = [obs.gt_anomaly for obs in observations]

    path_length = sum(
        _distance(centers[idx - 1], centers[idx])
        for idx in range(1, len(centers))
    )
    displacement = _distance(centers[0], centers[-1]) if len(centers) >= 2 else 0.0
    xs = [point[0] for point in centers]
    ys = [point[1] for point in centers]
    movement_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    centroid_radius = max((_distance(point, centroid) for point in centers), default=0.0)

    frame_count = len(observations)
    support_frames = sum(1 for flag in support_flags if flag)
    support_ratio = _safe_divide(support_frames, frame_count)
    max_support_run = _longest_true_run(support_flags)
    gt_anomaly_frames = sum(1 for flag in gt_positive_frames if flag)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_speed = max(speeds, default=0.0)
    p90_speed = float(np.percentile(speeds, 90)) if speeds else 0.0
    high_speed_frames = sum(1 for speed in speeds if speed >= cfg.running_speed)
    high_speed_ratio = _safe_divide(high_speed_frames, len(speeds))
    derived_feature_dict = features_from_trajectory_payload(
        {
            "centers": [[point[0], point[1]] for point in centers],
            "speeds": speeds,
        },
        high_speed_threshold=cfg.running_speed,
        low_speed_threshold=cfg.loiter_speed,
    ) or {}

    gt_supported = (
        frame_count >= cfg.min_track_frames
        and support_frames >= cfg.min_mask_overlap_frames
        and support_ratio >= cfg.min_mask_overlap_ratio
    )
    running_candidate = (
        gt_supported
        and len(speeds) >= cfg.running_frames
        and (
            high_speed_ratio >= cfg.running_min_high_speed_ratio
            or avg_speed >= cfg.running_speed * 0.8
        )
        and p90_speed >= cfg.running_speed
        and movement_extent >= cfg.loiter_radius * cfg.running_extent_multiplier
    )
    loitering_candidate = (
        gt_supported
        and frame_count >= cfg.loiter_frames
        and movement_extent <= cfg.loiter_radius * cfg.loiter_radius_multiplier
        and centroid_radius <= cfg.loiter_radius * cfg.loiter_radius_multiplier
        and avg_speed <= cfg.loiter_speed * cfg.loiter_speed_multiplier
    )
    normal_candidate = frame_count >= cfg.min_track_frames and support_frames == 0

    pseudo_labels = []
    reasons = []
    if running_candidate:
        pseudo_labels.append("running")
        reasons.append(
            f"high_speed_ratio={high_speed_ratio:.3f}, p90_speed={p90_speed:.3f}, movement_extent={movement_extent:.3f}"
        )
    if loitering_candidate:
        pseudo_labels.append("loitering")
        reasons.append(
            f"avg_speed={avg_speed:.3f}, centroid_radius={centroid_radius:.3f}, frame_count={frame_count}"
        )
    if normal_candidate:
        pseudo_labels.append("normal")
        reasons.append("no_anomaly_mask_overlap")

    if len([label for label in pseudo_labels if label != "normal"]) > 1:
        primary_label = "ambiguous_abnormal"
    elif "running" in pseudo_labels:
        primary_label = "running"
    elif "loitering" in pseudo_labels:
        primary_label = "loitering"
    elif "normal" in pseudo_labels:
        primary_label = "normal"
    else:
        primary_label = "unknown"

    return {
        "sample_id": f"{sequence_id}_track_{track_id}",
        "sequence_id": sequence_id,
        "track_id": int(track_id),
        "start_frame": int(frame_indices[0]),
        "end_frame": int(frame_indices[-1]),
        "frame_count": frame_count,
        "primary_label": primary_label,
        "pseudo_labels": pseudo_labels,
        "label_reasons": reasons,
        "features": {
            "mean_confidence": round(sum(confs) / len(confs), 4),
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
            "support_frames": support_frames,
            "support_ratio": round(support_ratio, 4),
            "max_support_run": max_support_run,
            "gt_anomaly_frames": gt_anomaly_frames,
            "mean_overlap_ratio": round(sum(overlap_ratios) / len(overlap_ratios), 4),
            "max_overlap_ratio": round(max(overlap_ratios, default=0.0), 4),
        },
        "trajectory": {
            "frames": frame_indices,
            "boxes": bboxes,
            "centers": [[round(point[0], 2), round(point[1], 2)] for point in centers],
            "speeds": _round_list(speeds),
            "mask_overlap_flags": support_flags,
            "mask_overlap_ratios": _round_list(overlap_ratios, digits=4),
        },
    }


def generate_avenue_pseudo_labels(config: AvenuePseudoLabelConfig) -> dict[str, object]:
    cfg = config.resolved()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        model_path=str(cfg.model),
        conf=cfg.conf_threshold,
        classes=[0],
        device=cfg.device,
    )
    sequences = _discover_sequences(cfg)

    all_samples: list[dict[str, object]] = []
    label_counts = {
        "running": 0,
        "loitering": 0,
        "normal": 0,
        "unknown": 0,
        "ambiguous_abnormal": 0,
    }
    total_frames = 0

    for sequence in sequences:
        tracker = ByteTrackerLite(
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            match_thresh=cfg.match_thresh,
            max_time_lost=cfg.max_time_lost,
            min_box_area=cfg.min_box_area,
        )

        track_observations: dict[int, list[TrackObservation]] = {}
        cap = cv2.VideoCapture(str(sequence.video_path))
        masks = _load_avenue_labels(sequence.label_path)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= len(masks):
                    break

                mask = masks[frame_idx]
                gt_anomaly = _mask_positive(mask)
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)

                for track in tracks:
                    x1, y1, x2, y2, track_id, conf = track
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    center = _bbox_center(bbox)
                    observation_list = track_observations.setdefault(int(track_id), [])
                    prev_center = observation_list[-1].center if observation_list else None
                    speed = _distance(center, prev_center) if prev_center is not None else 0.0
                    overlap_ratio, center_on_mask = _bbox_mask_support(bbox, mask)
                    observation_list.append(
                        TrackObservation(
                            frame_index=frame_idx + 1,
                            bbox=bbox,
                            center=center,
                            conf=float(conf),
                            speed=speed,
                            mask_overlap_ratio=overlap_ratio,
                            center_on_mask=center_on_mask,
                            gt_anomaly=gt_anomaly,
                        )
                    )

                frame_idx += 1
        finally:
            cap.release()

        total_frames += frame_idx
        for track_id, observations in track_observations.items():
            sample = _build_track_sample(cfg, sequence.sequence_id, track_id, observations)
            all_samples.append(sample)
            label_counts[sample["primary_label"]] = label_counts.get(sample["primary_label"], 0) + 1

    all_samples.sort(key=lambda item: (item["sequence_id"], item["track_id"]))

    all_path = cfg.output_dir / "tracks.jsonl"
    label_paths = {
        "running": cfg.output_dir / "running_tracks.jsonl",
        "loitering": cfg.output_dir / "loitering_tracks.jsonl",
        "normal": cfg.output_dir / "normal_tracks.jsonl",
        "unknown": cfg.output_dir / "unknown_tracks.jsonl",
        "ambiguous_abnormal": cfg.output_dir / "ambiguous_tracks.jsonl",
    }

    with all_path.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    for label, path in label_paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for sample in all_samples:
                if sample["primary_label"] == label:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            "avenue_root": str(cfg.avenue_root),
            "ground_truth_root": str(cfg.ground_truth_root),
            "model": str(cfg.model),
            "output_dir": str(cfg.output_dir),
            "conf_threshold": cfg.conf_threshold,
            "track_high_thresh": cfg.track_high_thresh,
            "track_low_thresh": cfg.track_low_thresh,
            "new_track_thresh": cfg.new_track_thresh,
            "match_thresh": cfg.match_thresh,
            "max_time_lost": cfg.max_time_lost,
            "min_box_area": cfg.min_box_area,
            "loiter_frames": cfg.loiter_frames,
            "loiter_radius": cfg.loiter_radius,
            "loiter_speed": cfg.loiter_speed,
            "running_speed": cfg.running_speed,
            "running_frames": cfg.running_frames,
            "min_track_frames": cfg.min_track_frames,
            "min_mask_overlap_frames": cfg.min_mask_overlap_frames,
            "min_mask_overlap_ratio": cfg.min_mask_overlap_ratio,
            "running_min_high_speed_ratio": cfg.running_min_high_speed_ratio,
            "loiter_radius_multiplier": cfg.loiter_radius_multiplier,
            "loiter_speed_multiplier": cfg.loiter_speed_multiplier,
            "running_extent_multiplier": cfg.running_extent_multiplier,
            "max_videos": cfg.max_videos,
            "sequence_ids": list(cfg.sequence_ids),
        },
        "summary": {
            "sequences": len(sequences),
            "frames": total_frames,
            "tracks": len(all_samples),
            "label_counts": label_counts,
        },
        "notes": [
            "Pseudo labels are generated from Avenue testing videos with ground-truth anomaly masks.",
            "A track is considered a positive abnormal candidate only when it overlaps anomaly masks often enough.",
            "These labels are intended for later behavior-model improvement, not as human-verified final annotations.",
        ],
        "artifacts": {
            "all_tracks": str(all_path),
            "running_tracks": str(label_paths["running"]),
            "loitering_tracks": str(label_paths["loitering"]),
            "normal_tracks": str(label_paths["normal"]),
            "unknown_tracks": str(label_paths["unknown"]),
            "ambiguous_tracks": str(label_paths["ambiguous_abnormal"]),
        },
    }

    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import features_from_trajectory_payload
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from training.config import resolve_path


@dataclass(slots=True)
class AvenueSequence:
    video_path: Path
    label_path: Path
    sequence_id: str


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _load_avenue_labels(label_path: Path) -> list[np.ndarray]:
    mat = loadmat(label_path)
    vol = mat["volLabel"]
    return [np.asarray(vol[0, idx], dtype=np.uint8) for idx in range(vol.shape[1])]


def _discover_sequences(avenue_root: Path, ground_truth_root: Path, sequence_ids: tuple[str, ...]) -> list[AvenueSequence]:
    video_dir = avenue_root / "testing_videos"
    selected_ids = {str(item).zfill(2) for item in sequence_ids}
    sequences = []
    for video_path in sorted(video_dir.glob("*.avi")):
        stem = video_path.stem
        if selected_ids and stem not in selected_ids:
            continue
        label_path = ground_truth_root / f"{int(stem)}_label.mat"
        if not label_path.exists():
            continue
        sequences.append(AvenueSequence(video_path=video_path, label_path=label_path, sequence_id=stem))
    if not sequences:
        raise FileNotFoundError("No matched Avenue sequences found for hard-negative mining.")
    return sequences


def _mask_overlap_stats(bbox: list[int], mask: np.ndarray) -> tuple[float, bool]:
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
    overlap_ratio = _safe_divide(float(np.count_nonzero(region > 0)), float(region.size))
    center_x = max(0, min(int((x1 + x2) / 2), mask.shape[1] - 1))
    center_y = max(0, min(int((y1 + y2) / 2), mask.shape[0] - 1))
    return overlap_ratio, bool(mask[center_y, center_x] > 0)


def _build_negative_sample(track_id: int, sequence_id: str, state: dict[str, object]) -> dict[str, object]:
    centers = state["centers"]
    speeds = state["speeds"]
    features = features_from_trajectory_payload(
        {
            "centers": centers,
            "speeds": speeds,
        },
        high_speed_threshold=15.57,
        low_speed_threshold=2.2,
    )
    if features is None:
        raise ValueError(f"Could not derive features for hard negative track {sequence_id}:{track_id}")

    overlap_ratios = state["overlap_ratios"]
    support_flags = state["support_flags"]
    mean_confidence = _safe_divide(sum(state["confs"]), len(state["confs"]))

    return {
        "sample_id": f"hardneg_{sequence_id}_track_{track_id}",
        "source_track_id": f"hardneg_{sequence_id}_track_{track_id}",
        "sequence_id": sequence_id,
        "track_id": int(track_id),
        "start_frame": int(state["frames"][0]),
        "end_frame": int(state["frames"][-1]),
        "frame_count": len(state["frames"]),
        "primary_label": "normal",
        "pseudo_labels": ["normal"],
        "label_reasons": [
            (
                "hard_negative_loitering_false_positive "
                f"alarm_frames={state['loiter_alarm_frames']} "
                f"model_loiter_frames={state['model_loiter_frames']} "
                f"support_frames={sum(1 for flag in support_flags if flag)}"
            )
        ],
        "features": {
            "mean_confidence": round(mean_confidence, 4),
            "avg_speed": round(float(features["avg_speed"]), 4),
            "speed_std": round(float(features["speed_std"]), 4),
            "max_speed": round(float(features["max_speed"]), 4),
            "p90_speed": round(float(features["p90_speed"]), 4),
            "high_speed_ratio": round(float(features["high_speed_ratio"]), 4),
            "stationary_ratio": round(float(features["stationary_ratio"]), 4),
            "path_length": round(float(features["path_length"]), 4),
            "displacement": round(float(features["displacement"]), 4),
            "movement_extent": round(float(features["movement_extent"]), 4),
            "centroid_radius": round(float(features["centroid_radius"]), 4),
            "straightness": round(float(features["straightness"]), 4),
            "direction_change_rate": round(float(features["direction_change_rate"]), 4),
            "mean_turn_angle": round(float(features["mean_turn_angle"]), 4),
            "support_frames": int(sum(1 for flag in support_flags if flag)),
            "support_ratio": round(_safe_divide(sum(1 for flag in support_flags if flag), len(support_flags)), 4),
            "max_support_run": int(state["max_support_run"]),
            "gt_anomaly_frames": int(state["gt_positive_frames"]),
            "mean_overlap_ratio": round(_safe_divide(sum(overlap_ratios), len(overlap_ratios)), 4),
            "max_overlap_ratio": round(max(overlap_ratios, default=0.0), 4),
            "hard_negative_score": round(
                state["loiter_alarm_frames"] + state["model_loiter_frames"] * 0.5,
                4,
            ),
        },
        "trajectory": {
            "frames": state["frames"],
            "boxes": state["boxes"],
            "centers": [[round(float(x), 2), round(float(y), 2)] for x, y in centers],
            "speeds": [round(float(value), 3) for value in speeds],
            "mask_overlap_flags": support_flags,
            "mask_overlap_ratios": [round(float(value), 4) for value in overlap_ratios],
        },
    }


def _merge_unique_rows(base_rows: list[dict[str, object]], extra_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged = []
    seen = set()
    for row in [*base_rows, *extra_rows]:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        merged.append(row)
    return merged


def mine_behavior_hard_negatives(
    *,
    avenue_root: str | Path = "data/CUHK_Avenue/Avenue Dataset",
    ground_truth_root: str | Path = "data/CUHK_Avenue/ground_truth_demo/testing_label_mask",
    detector_model: str | Path = "output/training/mot17_person_gpu_40e_960_pretrained_w2/weights/best.pt",
    behavior_model_path: str | Path = "output/behavior_training/avenue_behavior_mlp_seed2026/best.pt",
    input_dataset: str | Path = "output/avenue_pseudo_labels_filtered/tracks_filtered.jsonl",
    output_dir: str | Path = "output/behavior_hard_negatives",
    sequence_ids: tuple[str, ...] = ("01", "03", "04", "05", "17", "18"),
    device: str | int | None = 0,
    conf_threshold: float = 0.1,
    min_track_frames: int = 72,
    min_loiter_alarm_frames: int = 8,
    min_model_loiter_frames: int = 8,
    max_support_ratio: float = 0.05,
    max_overlap_ratio: float = 0.05,
) -> dict[str, object]:
    avenue_root = resolve_path(avenue_root)
    ground_truth_root = resolve_path(ground_truth_root)
    detector_model = resolve_path(detector_model)
    behavior_model_path = resolve_path(behavior_model_path)
    input_dataset = resolve_path(input_dataset)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _discover_sequences(avenue_root, ground_truth_root, sequence_ids)
    detector = YOLODetector(model_path=str(detector_model), conf=conf_threshold, classes=[0], device=device)
    tracker_kwargs = dict(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        match_thresh=0.8,
        max_time_lost=30,
        min_box_area=10.0,
    )
    behavior_kwargs = dict(
        roi=None,
        behavior_mode="hybrid",
        enable_intrusion=False,
        enable_cross_line=False,
        enable_loitering=True,
        enable_running=True,
        loiter_frames=168,
        loiter_radius=40.13,
        loiter_speed=1.45,
        running_speed=15.57,
        running_frames=3,
    )

    from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier

    classifier = TrajectoryBehaviorClassifier(checkpoint_path=behavior_model_path)

    hard_negative_rows: list[dict[str, object]] = []

    for sequence in sequences:
        tracker = ByteTrackerLite(**tracker_kwargs)
        behavior = AbnormalDetector(
            behavior_classifier=classifier,
            behavior_model_score_thresh=0.90,
            behavior_model_min_frames=24,
            loitering_model_score_thresh=0.97,
            running_model_score_thresh=0.85,
            loitering_model_min_frames=72,
            loitering_model_max_avg_speed=2.2,
            loitering_model_max_movement_extent=55.0,
            loitering_model_max_centroid_radius=28.0,
            running_model_min_avg_speed=8.0,
            running_model_min_p90_speed=16.0,
            running_model_min_movement_extent=120.0,
            **behavior_kwargs,
        )
        cap = cv2.VideoCapture(str(sequence.video_path))
        masks = _load_avenue_labels(sequence.label_path)
        track_states: dict[int, dict[str, object]] = {}
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= len(masks):
                    break
                mask = masks[frame_idx]
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                alarms, track_infos = behavior.update(tracks)
                loiter_alarm_ids = {int(track_id) for track_id, alarm_type in alarms if str(alarm_type) == "loitering"}

                for track in tracks:
                    x1, y1, x2, y2, track_id, conf = track
                    track_id = int(track_id)
                    info = track_infos.get(track_id, {})
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    overlap_ratio, support_flag = _mask_overlap_stats(bbox, mask)
                    state = track_states.setdefault(
                        track_id,
                        {
                            "frames": [],
                            "boxes": [],
                            "centers": [],
                            "speeds": [],
                            "confs": [],
                            "overlap_ratios": [],
                            "support_flags": [],
                            "gt_positive_frames": 0,
                            "loiter_alarm_frames": 0,
                            "model_loiter_frames": 0,
                            "max_support_run": 0,
                            "_support_run": 0,
                        },
                    )
                    state["frames"].append(frame_idx + 1)
                    state["boxes"].append(bbox)
                    trajectory = info.get("trajectory", [])
                    center = trajectory[-1] if trajectory else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    state["centers"].append((float(center[0]), float(center[1])))
                    speed_history = info.get("speed_history", [])
                    speed = float(speed_history[-1]) if speed_history else 0.0
                    state["speeds"].append(speed)
                    state["confs"].append(float(conf))
                    state["overlap_ratios"].append(float(overlap_ratio))
                    state["support_flags"].append(bool(support_flag))
                    if support_flag:
                        state["gt_positive_frames"] += 1
                        state["_support_run"] += 1
                        state["max_support_run"] = max(state["max_support_run"], state["_support_run"])
                    else:
                        state["_support_run"] = 0
                    if track_id in loiter_alarm_ids:
                        state["loiter_alarm_frames"] += 1
                    if info.get("model_behavior_label") == "loitering":
                        state["model_loiter_frames"] += 1

                frame_idx += 1
        finally:
            cap.release()

        for track_id, state in track_states.items():
            frame_count = len(state["frames"])
            support_ratio = _safe_divide(sum(1 for flag in state["support_flags"] if flag), frame_count)
            mean_overlap_ratio = _safe_divide(sum(state["overlap_ratios"]), len(state["overlap_ratios"]))
            if frame_count < min_track_frames:
                continue
            if state["loiter_alarm_frames"] < min_loiter_alarm_frames and state["model_loiter_frames"] < min_model_loiter_frames:
                continue
            if support_ratio > max_support_ratio or mean_overlap_ratio > max_overlap_ratio:
                continue
            hard_negative_rows.append(_build_negative_sample(track_id, sequence.sequence_id, state))

    base_rows = [
        json.loads(line)
        for line in input_dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    merged_rows = _merge_unique_rows(base_rows, hard_negative_rows)

    hard_negatives_path = output_dir / "hard_negative_tracks.jsonl"
    merged_path = output_dir / "tracks_with_hard_negatives.jsonl"
    summary_path = output_dir / "summary.json"

    with hard_negatives_path.open("w", encoding="utf-8") as handle:
        for row in hard_negative_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with merged_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = {}
    for row in merged_rows:
        label = str(row["primary_label"])
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "config": {
            "avenue_root": str(avenue_root),
            "ground_truth_root": str(ground_truth_root),
            "detector_model": str(detector_model),
            "behavior_model_path": str(behavior_model_path),
            "input_dataset": str(input_dataset),
            "sequence_ids": [str(item) for item in sequence_ids],
            "min_track_frames": min_track_frames,
            "min_loiter_alarm_frames": min_loiter_alarm_frames,
            "min_model_loiter_frames": min_model_loiter_frames,
            "max_support_ratio": max_support_ratio,
            "max_overlap_ratio": max_overlap_ratio,
        },
        "summary": {
            "hard_negative_rows": len(hard_negative_rows),
            "base_rows": len(base_rows),
            "merged_rows": len(merged_rows),
            "merged_label_counts": label_counts,
        },
        "artifacts": {
            "hard_negatives": str(hard_negatives_path),
            "merged_dataset": str(merged_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

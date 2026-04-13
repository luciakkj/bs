from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import (
    TrajectoryBehaviorClassifier,
    features_from_trajectory_payload,
)
from config import get_config
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _load_avenue_labels(label_path: Path) -> list[np.ndarray]:
    mat = loadmat(label_path)
    vol = mat["volLabel"]
    return [np.asarray(vol[0, idx], dtype=np.uint8) for idx in range(vol.shape[1])]


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


def _build_running_sample(
    track_id: int,
    sequence_id: str,
    state: dict[str, object],
    start_idx: int,
    end_idx: int,
) -> dict[str, object] | None:
    frames = state["frames"][start_idx:end_idx]
    boxes = state["boxes"][start_idx:end_idx]
    centers = state["centers"][start_idx:end_idx]
    speeds = state["speeds"][start_idx:end_idx]
    confs = state["confs"][start_idx:end_idx]
    overlap_ratios = state["overlap_ratios"][start_idx:end_idx]
    support_flags = state["support_flags"][start_idx:end_idx]
    running_alarm_flags = state["running_alarm_flags"][start_idx:end_idx]

    features = features_from_trajectory_payload(
        {
            "centers": centers,
            "speeds": speeds,
        },
        high_speed_threshold=15.57,
        low_speed_threshold=2.2,
    )
    if features is None:
        return None

    frame_count = len(frames)
    support_frames = int(sum(1 for flag in support_flags if flag))
    running_alarm_frames = int(sum(1 for flag in running_alarm_flags if flag))
    gt_positive_frames = support_frames
    mean_confidence = _safe_divide(sum(confs), len(confs))
    return {
        "sample_id": f"runpos_{sequence_id}_track_{track_id}_w_{frames[0]}_{frames[-1]}",
        "source_track_id": f"runpos_{sequence_id}_track_{track_id}",
        "sequence_id": str(sequence_id),
        "track_id": int(track_id),
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "frame_count": frame_count,
        "primary_label": "running",
        "pseudo_labels": ["running"],
        "label_reasons": [
            (
                "running_positive_mined "
                f"running_alarm_frames={running_alarm_frames} "
                f"gt_positive_frames={gt_positive_frames} "
                f"support_ratio={_safe_divide(support_frames, frame_count):.3f}"
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
            "revisit_ratio": round(float(features.get("revisit_ratio", 0.0)), 4),
            "unique_cell_ratio": round(float(features.get("unique_cell_ratio", 0.0)), 4),
            "max_cell_occupancy_ratio": round(float(features.get("max_cell_occupancy_ratio", 0.0)), 4),
            "support_frames": support_frames,
            "support_ratio": round(_safe_divide(support_frames, frame_count), 4),
            "max_support_run": support_frames,
            "running_alarm_frames": running_alarm_frames,
            "running_alarm_ratio": round(_safe_divide(running_alarm_frames, frame_count), 4),
            "gt_anomaly_frames": gt_positive_frames,
            "mean_overlap_ratio": round(_safe_divide(sum(overlap_ratios), len(overlap_ratios)), 4),
            "max_overlap_ratio": round(max(overlap_ratios, default=0.0), 4),
            "max_running_score": round(max((float(v) for v in state["running_scores"][start_idx:end_idx]), default=0.0), 4),
            "hard_positive_score": round(
                float(features["p90_speed"]) * 4.0
                + float(features["movement_extent"]) * 0.25
                + running_alarm_frames * 2.0
                + gt_positive_frames * 0.5,
                4,
            ),
        },
        "trajectory": {
            "frames": frames,
            "boxes": boxes,
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


def mine_running_positives(
    *,
    config_path: str = "config.yaml",
    input_dataset: str = "output/behavior_reconstructed/avenue_train_mask_windows_v2/tracks_reconstructed.jsonl",
    output_dir: str = "output/behavior_hard_negatives/runningpos_20260405",
    sequence_ids: tuple[str, ...] = ("01", "03", "04", "21"),
    min_track_frames: int = 24,
    min_support_ratio: float = 0.25,
    min_gt_positive_frames: int = 8,
    min_avg_speed: float = 6.0,
    min_p90_speed: float = 12.0,
    min_movement_extent: float = 70.0,
    min_high_speed_ratio: float = 0.10,
    min_running_alarm_frames: int = 2,
    min_running_score: float = 0.90,
    window_size: int = 24,
    window_stride: int = 6,
    top_k: int = 10,
) -> dict[str, object]:
    cfg = get_config(config_path)
    output_root = (PROJECT_ROOT / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = (PROJECT_ROOT / input_dataset).resolve()

    detector = YOLODetector(
        model_path=str(cfg.model.model_path),
        conf=cfg.model.conf_threshold,
        classes=list(cfg.model.classes),
        device=cfg.model.device,
        imgsz=cfg.model.imgsz,
        max_det=cfg.model.max_det,
        half=cfg.model.half,
        augment=cfg.model.augment,
    )
    behavior_classifier = None
    if cfg.behavior.behavior_model_path:
        behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=cfg.behavior.behavior_model_path,
            min_frames_override=cfg.behavior.behavior_model_min_frames,
        )
    secondary_behavior_classifier = None
    if cfg.behavior.behavior_secondary_model_path:
        secondary_behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=cfg.behavior.behavior_secondary_model_path,
            min_frames_override=cfg.behavior.behavior_model_min_frames,
        )

    selected_ids = {str(item).zfill(2) for item in sequence_ids}
    testing_dir = PROJECT_ROOT / "data" / "CUHK_Avenue" / "Avenue Dataset" / "testing_videos"
    gt_dir = PROJECT_ROOT / "data" / "CUHK_Avenue" / "ground_truth_demo" / "testing_label_mask"
    video_paths = [path for path in sorted(testing_dir.glob("*.avi")) if path.stem in selected_ids]
    if not video_paths:
        raise FileNotFoundError("No matching Avenue testing videos found.")

    all_candidates: list[dict[str, object]] = []
    track_summaries: list[dict[str, object]] = []

    for video_path in video_paths:
        sequence_id = video_path.stem
        label_path = gt_dir / f"{int(sequence_id)}_label.mat"
        if not label_path.exists():
            continue
        masks = _load_avenue_labels(label_path)

        tracker = ByteTrackerLite(
            track_high_thresh=cfg.tracker.track_high_thresh,
            track_low_thresh=cfg.tracker.track_low_thresh,
            new_track_thresh=cfg.tracker.new_track_thresh,
            match_thresh=cfg.tracker.match_thresh,
            low_match_thresh=cfg.tracker.low_match_thresh,
            unconfirmed_match_thresh=cfg.tracker.unconfirmed_match_thresh,
            score_fusion_weight=cfg.tracker.score_fusion_weight,
            max_time_lost=cfg.tracker.max_time_lost,
            min_box_area=cfg.tracker.min_box_area,
            appearance_enabled=cfg.tracker.appearance_enabled,
            appearance_weight=cfg.tracker.appearance_weight,
            appearance_ambiguity_margin=cfg.tracker.appearance_ambiguity_margin,
            appearance_feature_mode=cfg.tracker.appearance_feature_mode,
            appearance_hist_bins=cfg.tracker.appearance_hist_bins,
            appearance_min_box_size=cfg.tracker.appearance_min_box_size,
            appearance_reid_model=cfg.tracker.appearance_reid_model,
            appearance_reid_weights=cfg.tracker.appearance_reid_weights,
            appearance_reid_device=cfg.tracker.appearance_reid_device,
            appearance_reid_input_size=cfg.tracker.appearance_reid_input_size,
        )
        behavior = AbnormalDetector(
            roi=cfg.behavior.roi,
            behavior_mode=cfg.behavior.behavior_mode,
            enable_intrusion=cfg.behavior.enable_intrusion,
            enable_cross_line=cfg.behavior.enable_cross_line,
            enable_loitering=cfg.behavior.enable_loitering,
            enable_running=cfg.behavior.enable_running,
            intrusion_frames=cfg.behavior.intrusion_frames,
            cross_line=cfg.behavior.cross_line,
            loiter_frames=cfg.behavior.loiter_frames,
            loiter_radius=cfg.behavior.loiter_radius,
            loiter_speed=cfg.behavior.loiter_speed,
            running_speed=cfg.behavior.running_speed,
            running_frames=cfg.behavior.running_frames,
            behavior_classifier=behavior_classifier,
            secondary_behavior_classifier=secondary_behavior_classifier,
            behavior_ensemble_primary_weight=cfg.behavior.behavior_ensemble_primary_weight,
            behavior_ensemble_mode=cfg.behavior.behavior_ensemble_mode,
            behavior_model_score_thresh=cfg.behavior.behavior_model_score_thresh,
            behavior_model_min_frames=cfg.behavior.behavior_model_min_frames,
            behavior_model_max_tracks=cfg.behavior.behavior_model_max_tracks,
            behavior_model_resume_tracks=cfg.behavior.behavior_model_resume_tracks,
            behavior_secondary_max_tracks=cfg.behavior.behavior_secondary_max_tracks,
            behavior_secondary_resume_tracks=cfg.behavior.behavior_secondary_resume_tracks,
            behavior_secondary_loitering_only=cfg.behavior.behavior_secondary_loitering_only,
            behavior_model_eval_interval=cfg.behavior.behavior_model_eval_interval,
            loitering_hybrid_mode=cfg.behavior.loitering_hybrid_mode,
            loitering_model_support_thresh=cfg.behavior.loitering_model_support_thresh,
            loitering_model_score_thresh=cfg.behavior.loitering_model_score_thresh,
            running_model_score_thresh=cfg.behavior.running_model_score_thresh,
            loitering_model_min_frames=cfg.behavior.loitering_model_min_frames,
            loitering_activate_frames=cfg.behavior.loitering_activate_frames,
            loitering_support_activate_frames=cfg.behavior.loitering_support_activate_frames,
            loitering_support_block_running=cfg.behavior.loitering_support_block_running,
            loitering_context_gate_support_only=cfg.behavior.loitering_context_gate_support_only,
            loitering_release_frames=cfg.behavior.loitering_release_frames,
            loitering_model_max_avg_speed=cfg.behavior.loitering_model_max_avg_speed,
            loitering_model_min_movement_extent=cfg.behavior.loitering_model_min_movement_extent,
            loitering_model_min_centroid_radius=cfg.behavior.loitering_model_min_centroid_radius,
            loitering_model_max_movement_extent=cfg.behavior.loitering_model_max_movement_extent,
            loitering_model_max_centroid_radius=cfg.behavior.loitering_model_max_centroid_radius,
            loitering_model_min_stationary_ratio=cfg.behavior.loitering_model_min_stationary_ratio,
            loitering_model_min_revisit_ratio=cfg.behavior.loitering_model_min_revisit_ratio,
            loitering_model_max_unique_cell_ratio=cfg.behavior.loitering_model_max_unique_cell_ratio,
            loitering_model_min_max_cell_occupancy_ratio=cfg.behavior.loitering_model_min_max_cell_occupancy_ratio,
            loitering_model_max_straightness=cfg.behavior.loitering_model_max_straightness,
            loitering_rule_min_stationary_ratio=cfg.behavior.loitering_rule_min_stationary_ratio,
            loitering_rule_min_revisit_ratio=cfg.behavior.loitering_rule_min_revisit_ratio,
            loitering_rule_min_movement_extent=cfg.behavior.loitering_rule_min_movement_extent,
            loitering_rule_min_centroid_radius=cfg.behavior.loitering_rule_min_centroid_radius,
            loitering_rule_max_straightness=cfg.behavior.loitering_rule_max_straightness,
            loitering_rule_max_displacement_ratio=cfg.behavior.loitering_rule_max_displacement_ratio,
            loitering_max_neighbor_count=cfg.behavior.loitering_max_neighbor_count,
            loitering_neighbor_radius=cfg.behavior.loitering_neighbor_radius,
            running_model_min_avg_speed=cfg.behavior.running_model_min_avg_speed,
            running_model_min_p90_speed=cfg.behavior.running_model_min_p90_speed,
            running_model_min_movement_extent=cfg.behavior.running_model_min_movement_extent,
        )

        cap = cv2.VideoCapture(str(video_path))
        states: dict[int, dict[str, object]] = {}
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
                running_alarm_ids = {
                    int(track_id) for track_id, alarm_type in alarms if str(alarm_type) == "running"
                }
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf = track
                    track_id = int(track_id)
                    info = track_infos.get(track_id, {})
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    overlap_ratio, support_flag = _mask_overlap_stats(bbox, mask)
                    state = states.setdefault(
                        track_id,
                        {
                            "frames": [],
                            "boxes": [],
                            "centers": [],
                            "speeds": [],
                            "confs": [],
                            "overlap_ratios": [],
                            "support_flags": [],
                            "running_alarm_flags": [],
                            "running_scores": [],
                        },
                    )
                    state["frames"].append(int(frame_idx))
                    state["boxes"].append(bbox)
                    state["centers"].append(((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0))
                    state["speeds"].append(float(info.get("speed", 0.0)))
                    state["confs"].append(float(conf))
                    state["overlap_ratios"].append(float(overlap_ratio))
                    state["support_flags"].append(bool(support_flag))
                    state["running_alarm_flags"].append(
                        bool(track_id in running_alarm_ids or bool(info.get("running_active", False)))
                    )
                    state["running_scores"].append(
                        max(
                            float(info.get("model_running_score", 0.0)),
                            float(info.get("ensemble_running_score", 0.0)),
                        )
                    )
                frame_idx += 1
        finally:
            cap.release()

        per_sequence_candidates = []
        for track_id, state in states.items():
            frame_count = len(state["frames"])
            features = features_from_trajectory_payload(
                {"centers": state["centers"], "speeds": state["speeds"]},
                high_speed_threshold=15.57,
                low_speed_threshold=2.2,
            )
            if features is None:
                continue
            summary = {
                "sequence_id": sequence_id,
                "track_id": int(track_id),
                "frame_count": frame_count,
                "running_alarm_frames": int(sum(1 for flag in state["running_alarm_flags"] if flag)),
                "gt_positive_frames": int(sum(1 for flag in state["support_flags"] if flag)),
                "support_ratio": round(_safe_divide(sum(1 for flag in state["support_flags"] if flag), frame_count), 4),
                "avg_speed": round(float(features["avg_speed"]), 4),
                "p90_speed": round(float(features["p90_speed"]), 4),
                "movement_extent": round(float(features["movement_extent"]), 4),
                "max_running_score": round(max((float(v) for v in state["running_scores"]), default=0.0), 4),
            }
            track_summaries.append(summary)
            if frame_count < max(int(min_track_frames), int(window_size)):
                continue
            last_start = frame_count - int(window_size)
            for start_idx in range(0, last_start + 1, int(window_stride)):
                end_idx = start_idx + int(window_size)
                window_support_frames = sum(1 for flag in state["support_flags"][start_idx:end_idx] if flag)
                window_support_ratio = _safe_divide(window_support_frames, int(window_size))
                if window_support_frames < int(min_gt_positive_frames):
                    continue
                if window_support_ratio < float(min_support_ratio):
                    continue
                sample = _build_running_sample(track_id, sequence_id, state, start_idx, end_idx)
                if sample is None:
                    continue
                sample_features = sample["features"]
                if float(sample_features["avg_speed"]) < float(min_avg_speed):
                    continue
                if float(sample_features["p90_speed"]) < float(min_p90_speed):
                    continue
                if float(sample_features["movement_extent"]) < float(min_movement_extent):
                    continue
                if float(sample_features["high_speed_ratio"]) < float(min_high_speed_ratio):
                    continue
                if int(sample_features["running_alarm_frames"]) < int(min_running_alarm_frames):
                    continue
                if float(sample_features["max_running_score"]) < float(min_running_score):
                    continue
                per_sequence_candidates.append(sample)

        per_sequence_candidates.sort(key=lambda row: float(row["features"]["hard_positive_score"]), reverse=True)
        all_candidates.extend(per_sequence_candidates)

    base_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_candidates.sort(key=lambda row: float(row["features"]["hard_positive_score"]), reverse=True)
    selected_candidates = all_candidates[: int(top_k)]
    merged_rows = _merge_unique_rows(base_rows, selected_candidates)

    positives_path = output_root / "running_positive_tracks.jsonl"
    merged_path = output_root / "tracks_v2_plus_runningpos.jsonl"
    summary_path = output_root / "summary.json"

    positives_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in selected_candidates) + ("\n" if selected_candidates else ""),
        encoding="utf-8",
    )
    merged_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in merged_rows) + ("\n" if merged_rows else ""),
        encoding="utf-8",
    )

    label_counts: dict[str, int] = {}
    for row in merged_rows:
        label = str(row.get("primary_label", "unknown"))
        label_counts[label] = label_counts.get(label, 0) + 1

    track_summaries.sort(
        key=lambda item: (float(item["avg_speed"]) * 2.0 + float(item["movement_extent"]) * 0.05),
        reverse=True,
    )
    summary = {
        "config": {
            "config_path": str((PROJECT_ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)),
            "input_dataset": str(dataset_path),
            "sequence_ids": sorted(selected_ids),
            "min_track_frames": int(min_track_frames),
            "min_support_ratio": float(min_support_ratio),
            "min_gt_positive_frames": int(min_gt_positive_frames),
            "min_avg_speed": float(min_avg_speed),
            "min_p90_speed": float(min_p90_speed),
            "min_movement_extent": float(min_movement_extent),
            "min_high_speed_ratio": float(min_high_speed_ratio),
            "min_running_alarm_frames": int(min_running_alarm_frames),
            "min_running_score": float(min_running_score),
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "top_k": int(top_k),
        },
        "summary": {
            "base_rows": len(base_rows),
            "running_positive_rows": len(selected_candidates),
            "merged_rows": len(merged_rows),
            "merged_label_counts": label_counts,
        },
        "selected_tracks": [
            f"{row['sequence_id']}:{row['track_id']}" for row in selected_candidates
        ],
        "top_tracks": track_summaries[:20],
        "artifacts": {
            "running_positives": str(positives_path),
            "merged_dataset": str(merged_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine running-positive tracks from Avenue testing videos.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--input-dataset", default="output/behavior_reconstructed/avenue_train_mask_windows_v2/tracks_reconstructed.jsonl")
    parser.add_argument("--output-dir", default="output/behavior_hard_negatives/runningpos_20260405")
    parser.add_argument("--sequence-ids", nargs="*", default=["01", "03", "04", "21"])
    parser.add_argument("--min-track-frames", type=int, default=24)
    parser.add_argument("--min-support-ratio", type=float, default=0.25)
    parser.add_argument("--min-gt-positive-frames", type=int, default=8)
    parser.add_argument("--min-avg-speed", type=float, default=6.0)
    parser.add_argument("--min-p90-speed", type=float, default=12.0)
    parser.add_argument("--min-movement-extent", type=float, default=70.0)
    parser.add_argument("--min-high-speed-ratio", type=float, default=0.10)
    parser.add_argument("--min-running-alarm-frames", type=int, default=2)
    parser.add_argument("--min-running-score", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    result = mine_running_positives(
        config_path=args.config_path,
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        sequence_ids=tuple(args.sequence_ids),
        min_track_frames=args.min_track_frames,
        min_support_ratio=args.min_support_ratio,
        min_gt_positive_frames=args.min_gt_positive_frames,
        min_avg_speed=args.min_avg_speed,
        min_p90_speed=args.min_p90_speed,
        min_movement_extent=args.min_movement_extent,
        min_high_speed_ratio=args.min_high_speed_ratio,
        min_running_alarm_frames=args.min_running_alarm_frames,
        min_running_score=args.min_running_score,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

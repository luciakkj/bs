from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from config import get_config
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from training.config import UBnormalValidationConfig
from utils.visualization import draw


@dataclass(slots=True)
class UBnormalSequence:
    video_name: str
    video_path: Path
    annotation_dir: Path | None
    track_path: Path | None
    split: str
    label: str
    scene: str


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _mask_positive(mask: np.ndarray) -> bool:
    return bool(np.any(mask > 0))


def _track_overlaps_mask(track, mask: np.ndarray) -> bool:
    x1, y1, x2, y2, *_ = track
    x1 = max(0, min(int(x1), mask.shape[1] - 1))
    x2 = max(0, min(int(x2), mask.shape[1]))
    y1 = max(0, min(int(y1), mask.shape[0] - 1))
    y2 = max(0, min(int(y2), mask.shape[0]))

    if x2 <= x1 or y2 <= y1:
        return False

    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return False
    if np.any(region > 0):
        return True

    cx = max(0, min(int((x1 + x2) / 2), mask.shape[1] - 1))
    cy = max(0, min(int((y1 + y2) / 2), mask.shape[0] - 1))
    return bool(mask[cy, cx] > 0)


def _append_unique_alarm(
    alarms: list[tuple[int, str]],
    seen: set[tuple[int, str]],
    track_id: int,
    alarm_type: str,
) -> None:
    key = (int(track_id), str(alarm_type))
    if key in seen:
        return
    seen.add(key)
    alarms.append(key)


def _effective_alarms(
    alarms: list[tuple[int, str]],
    track_infos: dict[int, dict[str, object]],
) -> list[tuple[int, str]]:
    merged: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for track_id, alarm_type in alarms:
        _append_unique_alarm(merged, seen, int(track_id), str(alarm_type))

    for track_id, track_info in track_infos.items():
        if bool(track_info.get("intrusion_active", False)):
            _append_unique_alarm(merged, seen, int(track_id), "intrusion")
        if bool(track_info.get("loitering_active", False)):
            _append_unique_alarm(merged, seen, int(track_id), "loitering")
        if bool(track_info.get("running_active", False)):
            _append_unique_alarm(merged, seen, int(track_id), "running")
    return merged


def _load_manifest(manifest_path: Path, sequence_ids: tuple[str, ...], max_videos: int | None) -> list[UBnormalSequence]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = set(sequence_ids)
    sequences: list[UBnormalSequence] = []
    for row in rows:
        video_name = str(row["video_name"])
        if selected and video_name not in selected:
            continue
        annotation_dir = Path(row["annotation_dir"]) if row.get("annotation_dir") else None
        track_path = Path(row["track_path"]) if row.get("track_path") else None
        sequences.append(
            UBnormalSequence(
                video_name=video_name,
                video_path=Path(row["video_path"]),
                annotation_dir=annotation_dir,
                track_path=track_path,
                split=str(row["split"]),
                label=str(row["label"]),
                scene=str(row["scene"]),
            )
        )

    if max_videos is not None:
        sequences = sequences[:max_videos]
    if not sequences:
        raise FileNotFoundError("No matched UBnormal sequences were found in the manifest.")
    return sequences


def _load_ubnormal_masks(annotation_dir: Path | None) -> list[np.ndarray]:
    if annotation_dir is None or not annotation_dir.exists():
        return []
    masks: list[np.ndarray] = []
    for mask_path in sorted(annotation_dir.glob("*_gt.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        masks.append(mask.astype(np.uint8))
    return masks


def _load_track_intervals(track_path: Path | None) -> list[tuple[int, int, int]]:
    if track_path is None or not track_path.exists():
        return []
    intervals: list[tuple[int, int, int]] = []
    for line in track_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        object_id, start_frame, end_frame = (int(float(value)) for value in parts)
        intervals.append((object_id, start_frame, end_frame))
    return intervals


def _active_anomaly_ids(intervals: list[tuple[int, int, int]], frame_idx: int) -> set[int]:
    return {
        object_id
        for object_id, start_frame, end_frame in intervals
        if start_frame <= frame_idx <= end_frame
    }


def validate_on_ubnormal(config: UBnormalValidationConfig) -> dict[str, object]:
    cfg = config.resolved()
    sequences = _load_manifest(cfg.manifest_path, cfg.sequence_ids, cfg.max_videos)
    app_cfg = get_config(str(cfg.config_path))

    detector = YOLODetector(
        model_path=str(app_cfg.model.model_path),
        conf=app_cfg.model.conf_threshold,
        classes=list(app_cfg.model.classes),
        device=app_cfg.model.device,
        imgsz=app_cfg.model.imgsz,
        max_det=app_cfg.model.max_det,
        half=app_cfg.model.half,
        augment=app_cfg.model.augment,
    )
    behavior_classifier = None
    if app_cfg.behavior.behavior_model_path:
        behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=app_cfg.behavior.behavior_model_path,
            min_frames_override=app_cfg.behavior.behavior_model_min_frames,
        )
    secondary_behavior_classifier = None
    if app_cfg.behavior.behavior_secondary_model_path:
        secondary_behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=app_cfg.behavior.behavior_secondary_model_path,
            min_frames_override=app_cfg.behavior.behavior_model_min_frames,
        )

    if cfg.save_demo_frames:
        cfg.demo_dir.mkdir(parents=True, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_frames = 0
    total_predicted_anomaly_frames = 0
    total_gt_anomaly_frames = 0
    total_tracks = 0
    total_alarm_events = 0
    total_alarm_types: dict[str, int] = {}
    sequence_reports = []

    for sequence in sequences:
        tracker = ByteTrackerLite(
            track_high_thresh=app_cfg.tracker.track_high_thresh,
            track_low_thresh=app_cfg.tracker.track_low_thresh,
            new_track_thresh=app_cfg.tracker.new_track_thresh,
            match_thresh=app_cfg.tracker.match_thresh,
            low_match_thresh=app_cfg.tracker.low_match_thresh,
            unconfirmed_match_thresh=app_cfg.tracker.unconfirmed_match_thresh,
            score_fusion_weight=app_cfg.tracker.score_fusion_weight,
            max_time_lost=app_cfg.tracker.max_time_lost,
            min_box_area=app_cfg.tracker.min_box_area,
            appearance_enabled=app_cfg.tracker.appearance_enabled,
            appearance_weight=app_cfg.tracker.appearance_weight,
            appearance_ambiguity_margin=app_cfg.tracker.appearance_ambiguity_margin,
            appearance_feature_mode=app_cfg.tracker.appearance_feature_mode,
            appearance_hist_bins=app_cfg.tracker.appearance_hist_bins,
            appearance_min_box_size=app_cfg.tracker.appearance_min_box_size,
            appearance_reid_model=app_cfg.tracker.appearance_reid_model,
            appearance_reid_weights=app_cfg.tracker.appearance_reid_weights,
            appearance_reid_device=app_cfg.tracker.appearance_reid_device,
            appearance_reid_input_size=app_cfg.tracker.appearance_reid_input_size,
            crowd_boost_enabled=app_cfg.tracker.crowd_boost_enabled,
            crowd_boost_det_count=app_cfg.tracker.crowd_boost_det_count,
            crowd_match_thresh=app_cfg.tracker.crowd_match_thresh,
            crowd_low_match_thresh=app_cfg.tracker.crowd_low_match_thresh,
            crowd_appearance_weight=app_cfg.tracker.crowd_appearance_weight,
            crowd_boost_min_small_ratio=app_cfg.tracker.crowd_boost_min_small_ratio,
            crowd_boost_max_median_area_ratio=app_cfg.tracker.crowd_boost_max_median_area_ratio,
            crowd_boost_small_area_ratio_thresh=app_cfg.tracker.crowd_boost_small_area_ratio_thresh,
        )

        behavior = AbnormalDetector(
            roi=app_cfg.behavior.roi,
            behavior_mode=app_cfg.behavior.behavior_mode,
            enable_intrusion=app_cfg.behavior.enable_intrusion,
            enable_cross_line=app_cfg.behavior.enable_cross_line,
            enable_loitering=app_cfg.behavior.enable_loitering,
            enable_running=app_cfg.behavior.enable_running,
            intrusion_frames=app_cfg.behavior.intrusion_frames,
            cross_line=app_cfg.behavior.cross_line,
            loiter_frames=app_cfg.behavior.loiter_frames,
            loiter_radius=app_cfg.behavior.loiter_radius,
            loiter_speed=app_cfg.behavior.loiter_speed,
            running_speed=app_cfg.behavior.running_speed,
            running_frames=app_cfg.behavior.running_frames,
            behavior_classifier=behavior_classifier,
            secondary_behavior_classifier=secondary_behavior_classifier,
            behavior_ensemble_primary_weight=app_cfg.behavior.behavior_ensemble_primary_weight,
            behavior_ensemble_mode=app_cfg.behavior.behavior_ensemble_mode,
            behavior_model_score_thresh=app_cfg.behavior.behavior_model_score_thresh,
            behavior_model_min_frames=app_cfg.behavior.behavior_model_min_frames,
            behavior_model_max_tracks=app_cfg.behavior.behavior_model_max_tracks,
            behavior_model_resume_tracks=app_cfg.behavior.behavior_model_resume_tracks,
            behavior_secondary_max_tracks=app_cfg.behavior.behavior_secondary_max_tracks,
            behavior_secondary_resume_tracks=app_cfg.behavior.behavior_secondary_resume_tracks,
            behavior_secondary_loitering_only=app_cfg.behavior.behavior_secondary_loitering_only,
            behavior_model_eval_interval=app_cfg.behavior.behavior_model_eval_interval,
            loitering_hybrid_mode=app_cfg.behavior.loitering_hybrid_mode,
            loitering_model_support_thresh=app_cfg.behavior.loitering_model_support_thresh,
            loitering_model_score_thresh=app_cfg.behavior.loitering_model_score_thresh,
            running_model_score_thresh=app_cfg.behavior.running_model_score_thresh,
            loitering_model_min_frames=app_cfg.behavior.loitering_model_min_frames,
            loitering_activate_frames=app_cfg.behavior.loitering_activate_frames,
            loitering_support_activate_frames=app_cfg.behavior.loitering_support_activate_frames,
            loitering_support_block_running=app_cfg.behavior.loitering_support_block_running,
            loitering_context_gate_support_only=app_cfg.behavior.loitering_context_gate_support_only,
            loitering_release_frames=app_cfg.behavior.loitering_release_frames,
            loitering_model_max_avg_speed=app_cfg.behavior.loitering_model_max_avg_speed,
            loitering_model_min_movement_extent=app_cfg.behavior.loitering_model_min_movement_extent,
            loitering_model_min_centroid_radius=app_cfg.behavior.loitering_model_min_centroid_radius,
            loitering_model_max_movement_extent=app_cfg.behavior.loitering_model_max_movement_extent,
            loitering_model_max_centroid_radius=app_cfg.behavior.loitering_model_max_centroid_radius,
            loitering_model_min_stationary_ratio=app_cfg.behavior.loitering_model_min_stationary_ratio,
            loitering_model_min_revisit_ratio=app_cfg.behavior.loitering_model_min_revisit_ratio,
            loitering_model_max_unique_cell_ratio=app_cfg.behavior.loitering_model_max_unique_cell_ratio,
            loitering_model_min_max_cell_occupancy_ratio=app_cfg.behavior.loitering_model_min_max_cell_occupancy_ratio,
            loitering_model_max_straightness=app_cfg.behavior.loitering_model_max_straightness,
            loitering_rule_min_stationary_ratio=app_cfg.behavior.loitering_rule_min_stationary_ratio,
            loitering_rule_min_revisit_ratio=app_cfg.behavior.loitering_rule_min_revisit_ratio,
            loitering_rule_min_movement_extent=app_cfg.behavior.loitering_rule_min_movement_extent,
            loitering_rule_min_centroid_radius=app_cfg.behavior.loitering_rule_min_centroid_radius,
            loitering_rule_max_straightness=app_cfg.behavior.loitering_rule_max_straightness,
            loitering_rule_max_displacement_ratio=app_cfg.behavior.loitering_rule_max_displacement_ratio,
            loitering_max_neighbor_count=app_cfg.behavior.loitering_max_neighbor_count,
            loitering_neighbor_radius=app_cfg.behavior.loitering_neighbor_radius,
            running_model_min_avg_speed=app_cfg.behavior.running_model_min_avg_speed,
            running_model_min_p90_speed=app_cfg.behavior.running_model_min_p90_speed,
            running_model_min_movement_extent=app_cfg.behavior.running_model_min_movement_extent,
        )

        cap = cv2.VideoCapture(str(sequence.video_path))
        masks = _load_ubnormal_masks(sequence.annotation_dir)
        track_intervals = _load_track_intervals(sequence.track_path)
        frame_idx = 0

        seq_tp = 0
        seq_fp = 0
        seq_fn = 0
        seq_tn = 0
        seq_pred_frames = 0
        seq_gt_frames = 0
        seq_tracks = 0
        seq_alarm_events = 0
        seq_alarm_types: dict[str, int] = {}
        saved_demo_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < len(masks):
                    mask = masks[frame_idx]
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                alarms, track_infos = behavior.update(tracks)
                effective_alarms = _effective_alarms(alarms, track_infos)

                active_ids = _active_anomaly_ids(track_intervals, frame_idx)
                gt_anomaly = bool(active_ids)
                pred_anomaly = bool(effective_alarms)

                if gt_anomaly:
                    seq_gt_frames += 1
                if pred_anomaly:
                    seq_pred_frames += 1
                    seq_alarm_events += len(effective_alarms)
                    for _, alarm_type in effective_alarms:
                        seq_alarm_types[alarm_type] = seq_alarm_types.get(alarm_type, 0) + 1

                if pred_anomaly and gt_anomaly:
                    seq_tp += 1
                elif pred_anomaly and not gt_anomaly:
                    seq_fp += 1
                elif (not pred_anomaly) and gt_anomaly:
                    seq_fn += 1
                else:
                    seq_tn += 1

                seq_tracks += len(tracks)

                if cfg.save_demo_frames and pred_anomaly and saved_demo_frames < 8:
                    vis = draw(
                        frame=frame,
                        tracks=tracks,
                        alarms=effective_alarms,
                        fps=0.0,
                        roi=app_cfg.behavior.roi,
                        cross_line=app_cfg.behavior.cross_line,
                        track_infos=track_infos,
                    )
                    heat = np.zeros_like(vis)
                    heat[:, :, 2] = (mask > 0).astype(np.uint8) * 255
                    blended = cv2.addWeighted(vis, 1.0, heat, 0.35, 0.0)
                    output_path = cfg.demo_dir / f"{sequence.video_name}_{frame_idx + 1:04d}.jpg"
                    cv2.imwrite(str(output_path), blended)
                    saved_demo_frames += 1

                frame_idx += 1
        finally:
            cap.release()

        precision = _safe_divide(seq_tp, seq_tp + seq_fp)
        recall = _safe_divide(seq_tp, seq_tp + seq_fn)
        sequence_reports.append(
            {
                "video_name": sequence.video_name,
                "scene": sequence.scene,
                "split": sequence.split,
                "label": sequence.label,
                "video_path": str(sequence.video_path),
                "annotation_dir": str(sequence.annotation_dir) if sequence.annotation_dir else None,
                "track_path": str(sequence.track_path) if sequence.track_path else None,
                "frames": frame_idx,
                "predicted_anomaly_frames": seq_pred_frames,
                "gt_anomaly_frames": seq_gt_frames,
                "tp": seq_tp,
                "fp": seq_fp,
                "fn": seq_fn,
                "tn": seq_tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(_f1(precision, recall), 4),
                "alarm_events": seq_alarm_events,
                "avg_tracks_per_frame": round(_safe_divide(seq_tracks, frame_idx), 4),
                "alarm_type_counts": seq_alarm_types,
            }
        )

        total_tp += seq_tp
        total_fp += seq_fp
        total_fn += seq_fn
        total_tn += seq_tn
        total_frames += frame_idx
        total_predicted_anomaly_frames += seq_pred_frames
        total_gt_anomaly_frames += seq_gt_frames
        total_tracks += seq_tracks
        total_alarm_events += seq_alarm_events
        for alarm_type, count in seq_alarm_types.items():
            total_alarm_types[alarm_type] = total_alarm_types.get(alarm_type, 0) + count

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)
    report = {
        "config": {
            "manifest_path": str(cfg.manifest_path),
            "config_path": str(cfg.config_path),
            "model_path": str(app_cfg.model.model_path),
            "behavior_mode": app_cfg.behavior.behavior_mode,
            "save_demo_frames": bool(cfg.save_demo_frames),
            "sequence_ids": list(cfg.sequence_ids),
        },
        "summary": {
            "videos": len(sequence_reports),
            "frames": total_frames,
            "predicted_anomaly_frames": total_predicted_anomaly_frames,
            "gt_anomaly_frames": total_gt_anomaly_frames,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(precision, recall), 4),
            "alarm_events": total_alarm_events,
            "alarm_type_counts": total_alarm_types,
            "avg_tracks_per_frame": round(_safe_divide(total_tracks, total_frames), 4),
        },
        "notes": [
            "UBnormal uses official train/val/test video lists downloaded from the public repository.",
            "Metrics here are frame-level anomaly metrics derived from the current detection-tracking-behavior pipeline.",
            "Ground truth comes from UBnormal per-video anomaly track intervals (*_tracks.txt); normal videos contribute only TN/FP frames.",
        ],
        "videos": sequence_reports,
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from training.config import AvenueValidationConfig
from utils.visualization import draw


@dataclass(slots=True)
class AvenueSequence:
    video_path: Path
    label_path: Path
    sequence_id: str


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _load_avenue_labels(label_path: Path) -> list[np.ndarray]:
    mat = loadmat(label_path)
    vol = mat["volLabel"]
    frames = [np.asarray(vol[0, idx], dtype=np.uint8) for idx in range(vol.shape[1])]
    return frames


def _discover_sequences(cfg: AvenueValidationConfig) -> list[AvenueSequence]:
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


def validate_on_avenue(config: AvenueValidationConfig) -> dict[str, object]:
    cfg = config.resolved()
    sequences = _discover_sequences(cfg)

    detector = YOLODetector(
        model_path=str(cfg.model),
        conf=cfg.conf_threshold,
        classes=[0],
        device=cfg.device,
        imgsz=cfg.imgsz,
        max_det=cfg.max_det,
        half=cfg.half,
        augment=cfg.augment,
    )
    behavior_classifier = None
    if cfg.behavior_model_path:
        behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=cfg.behavior_model_path,
            min_frames_override=cfg.behavior_model_min_frames,
        )
    secondary_behavior_classifier = None
    if cfg.behavior_secondary_model_path:
        secondary_behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=cfg.behavior_secondary_model_path,
            min_frames_override=cfg.behavior_model_min_frames,
        )
    tracker = ByteTrackerLite(
        track_high_thresh=cfg.track_high_thresh,
        track_low_thresh=cfg.track_low_thresh,
        new_track_thresh=cfg.new_track_thresh,
        match_thresh=cfg.match_thresh,
        low_match_thresh=cfg.low_match_thresh,
        unconfirmed_match_thresh=cfg.unconfirmed_match_thresh,
        score_fusion_weight=cfg.score_fusion_weight,
        max_time_lost=cfg.max_time_lost,
        min_box_area=cfg.min_box_area,
        appearance_enabled=cfg.appearance_enabled,
        appearance_weight=cfg.appearance_weight,
        appearance_ambiguity_margin=cfg.appearance_ambiguity_margin,
        appearance_feature_mode=cfg.appearance_feature_mode,
        appearance_hist_bins=cfg.appearance_hist_bins,
        appearance_min_box_size=cfg.appearance_min_box_size,
        appearance_reid_model=cfg.appearance_reid_model,
        appearance_reid_weights=str(cfg.appearance_reid_weights) if cfg.appearance_reid_weights else "",
        appearance_reid_device=cfg.appearance_reid_device,
        appearance_reid_input_size=cfg.appearance_reid_input_size,
        crowd_boost_enabled=cfg.crowd_boost_enabled,
        crowd_boost_det_count=cfg.crowd_boost_det_count,
        crowd_match_thresh=cfg.crowd_match_thresh,
        crowd_low_match_thresh=cfg.crowd_low_match_thresh,
        crowd_appearance_weight=cfg.crowd_appearance_weight,
        crowd_boost_min_small_ratio=cfg.crowd_boost_min_small_ratio,
        crowd_boost_max_median_area_ratio=cfg.crowd_boost_max_median_area_ratio,
        crowd_boost_small_area_ratio_thresh=cfg.crowd_boost_small_area_ratio_thresh,
    )
    behavior = AbnormalDetector(
        roi=cfg.roi,
        behavior_mode=cfg.behavior_mode,
        enable_intrusion=cfg.enable_intrusion,
        enable_cross_line=cfg.enable_cross_line,
        enable_loitering=cfg.enable_loitering,
        enable_running=cfg.enable_running,
        intrusion_frames=1,
        cross_line=cfg.cross_line,
        loiter_frames=cfg.loiter_frames,
        loiter_radius=cfg.loiter_radius,
        loiter_speed=cfg.loiter_speed,
        running_speed=cfg.running_speed,
        running_frames=cfg.running_frames,
        behavior_classifier=behavior_classifier,
        secondary_behavior_classifier=secondary_behavior_classifier,
        behavior_ensemble_primary_weight=cfg.behavior_ensemble_primary_weight,
        behavior_ensemble_mode=cfg.behavior_ensemble_mode,
        behavior_model_score_thresh=cfg.behavior_model_score_thresh,
        behavior_model_min_frames=cfg.behavior_model_min_frames,
        behavior_model_max_tracks=cfg.behavior_model_max_tracks,
        behavior_model_resume_tracks=cfg.behavior_model_resume_tracks,
        behavior_secondary_max_tracks=cfg.behavior_secondary_max_tracks,
        behavior_secondary_resume_tracks=cfg.behavior_secondary_resume_tracks,
        behavior_secondary_loitering_only=cfg.behavior_secondary_loitering_only,
        behavior_model_eval_interval=cfg.behavior_model_eval_interval,
        loitering_hybrid_mode=cfg.loitering_hybrid_mode,
        loitering_model_support_thresh=cfg.loitering_model_support_thresh,
        loitering_model_score_thresh=cfg.loitering_model_score_thresh,
        running_model_score_thresh=cfg.running_model_score_thresh,
        loitering_model_min_frames=cfg.loitering_model_min_frames,
        loitering_activate_frames=cfg.loitering_activate_frames,
        loitering_support_activate_frames=cfg.loitering_support_activate_frames,
        loitering_support_block_running=cfg.loitering_support_block_running,
        loitering_context_gate_support_only=cfg.loitering_context_gate_support_only,
        running_loitering_arb_enabled=cfg.running_loitering_arb_enabled,
        running_loitering_min_loitering_score=cfg.running_loitering_min_loitering_score,
        running_loitering_min_stationary_ratio=cfg.running_loitering_min_stationary_ratio,
        running_loitering_max_movement_extent=cfg.running_loitering_max_movement_extent,
        running_loitering_max_p90_speed=cfg.running_loitering_max_p90_speed,
        loitering_release_frames=cfg.loitering_release_frames,
        loitering_model_max_avg_speed=cfg.loitering_model_max_avg_speed,
        loitering_model_max_movement_extent=cfg.loitering_model_max_movement_extent,
        loitering_model_max_centroid_radius=cfg.loitering_model_max_centroid_radius,
        loitering_model_min_stationary_ratio=cfg.loitering_model_min_stationary_ratio,
        loitering_model_min_revisit_ratio=cfg.loitering_model_min_revisit_ratio,
        loitering_model_max_unique_cell_ratio=cfg.loitering_model_max_unique_cell_ratio,
        loitering_model_min_max_cell_occupancy_ratio=cfg.loitering_model_min_max_cell_occupancy_ratio,
        loitering_model_max_straightness=cfg.loitering_model_max_straightness,
        loitering_rule_min_stationary_ratio=cfg.loitering_rule_min_stationary_ratio,
        loitering_rule_min_revisit_ratio=cfg.loitering_rule_min_revisit_ratio,
        loitering_rule_min_movement_extent=cfg.loitering_rule_min_movement_extent,
        loitering_rule_min_centroid_radius=cfg.loitering_rule_min_centroid_radius,
        loitering_rule_max_straightness=cfg.loitering_rule_max_straightness,
        loitering_rule_max_displacement_ratio=cfg.loitering_rule_max_displacement_ratio,
        loitering_max_neighbor_count=cfg.loitering_max_neighbor_count,
        loitering_neighbor_radius=cfg.loitering_neighbor_radius,
        running_model_min_avg_speed=cfg.running_model_min_avg_speed,
        running_model_min_p90_speed=cfg.running_model_min_p90_speed,
        running_model_min_movement_extent=cfg.running_model_min_movement_extent,
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
    total_alarm_tracks_overlapping_gt = 0
    total_alarm_events = 0
    total_alarm_types: dict[str, int] = {}
    sequence_reports = []

    for sequence in sequences:
        tracker = ByteTrackerLite(
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            match_thresh=cfg.match_thresh,
            low_match_thresh=cfg.low_match_thresh,
            unconfirmed_match_thresh=cfg.unconfirmed_match_thresh,
        score_fusion_weight=cfg.score_fusion_weight,
        max_time_lost=cfg.max_time_lost,
        min_box_area=cfg.min_box_area,
        appearance_enabled=cfg.appearance_enabled,
        appearance_weight=cfg.appearance_weight,
        appearance_ambiguity_margin=cfg.appearance_ambiguity_margin,
        appearance_feature_mode=cfg.appearance_feature_mode,
        appearance_hist_bins=cfg.appearance_hist_bins,
        appearance_min_box_size=cfg.appearance_min_box_size,
            appearance_reid_model=cfg.appearance_reid_model,
            appearance_reid_weights=str(cfg.appearance_reid_weights) if cfg.appearance_reid_weights else "",
            appearance_reid_device=cfg.appearance_reid_device,
            appearance_reid_input_size=cfg.appearance_reid_input_size,
            crowd_boost_enabled=cfg.crowd_boost_enabled,
            crowd_boost_det_count=cfg.crowd_boost_det_count,
            crowd_match_thresh=cfg.crowd_match_thresh,
            crowd_low_match_thresh=cfg.crowd_low_match_thresh,
            crowd_appearance_weight=cfg.crowd_appearance_weight,
            crowd_boost_min_small_ratio=cfg.crowd_boost_min_small_ratio,
            crowd_boost_max_median_area_ratio=cfg.crowd_boost_max_median_area_ratio,
            crowd_boost_small_area_ratio_thresh=cfg.crowd_boost_small_area_ratio_thresh,
        )
        behavior = AbnormalDetector(
            roi=cfg.roi,
            behavior_mode=cfg.behavior_mode,
            enable_intrusion=cfg.enable_intrusion,
            enable_cross_line=cfg.enable_cross_line,
            enable_loitering=cfg.enable_loitering,
            enable_running=cfg.enable_running,
            intrusion_frames=1,
            cross_line=cfg.cross_line,
            loiter_frames=cfg.loiter_frames,
            loiter_radius=cfg.loiter_radius,
            loiter_speed=cfg.loiter_speed,
            running_speed=cfg.running_speed,
            running_frames=cfg.running_frames,
            behavior_classifier=behavior_classifier,
            secondary_behavior_classifier=secondary_behavior_classifier,
            behavior_ensemble_primary_weight=cfg.behavior_ensemble_primary_weight,
            behavior_ensemble_mode=cfg.behavior_ensemble_mode,
            behavior_model_score_thresh=cfg.behavior_model_score_thresh,
            behavior_model_min_frames=cfg.behavior_model_min_frames,
            behavior_model_max_tracks=cfg.behavior_model_max_tracks,
            behavior_model_resume_tracks=cfg.behavior_model_resume_tracks,
            behavior_secondary_max_tracks=cfg.behavior_secondary_max_tracks,
            behavior_secondary_resume_tracks=cfg.behavior_secondary_resume_tracks,
            behavior_secondary_loitering_only=cfg.behavior_secondary_loitering_only,
            behavior_model_eval_interval=cfg.behavior_model_eval_interval,
            loitering_hybrid_mode=cfg.loitering_hybrid_mode,
            loitering_model_support_thresh=cfg.loitering_model_support_thresh,
            loitering_model_score_thresh=cfg.loitering_model_score_thresh,
            running_model_score_thresh=cfg.running_model_score_thresh,
            loitering_model_min_frames=cfg.loitering_model_min_frames,
            loitering_activate_frames=cfg.loitering_activate_frames,
            loitering_support_activate_frames=cfg.loitering_support_activate_frames,
            loitering_support_block_running=cfg.loitering_support_block_running,
            loitering_context_gate_support_only=cfg.loitering_context_gate_support_only,
            loitering_release_frames=cfg.loitering_release_frames,
            loitering_model_max_avg_speed=cfg.loitering_model_max_avg_speed,
            loitering_model_max_movement_extent=cfg.loitering_model_max_movement_extent,
            loitering_model_max_centroid_radius=cfg.loitering_model_max_centroid_radius,
            loitering_model_min_stationary_ratio=cfg.loitering_model_min_stationary_ratio,
            loitering_model_min_revisit_ratio=cfg.loitering_model_min_revisit_ratio,
            loitering_model_max_unique_cell_ratio=cfg.loitering_model_max_unique_cell_ratio,
            loitering_model_min_max_cell_occupancy_ratio=cfg.loitering_model_min_max_cell_occupancy_ratio,
            loitering_model_max_straightness=cfg.loitering_model_max_straightness,
            loitering_rule_min_stationary_ratio=cfg.loitering_rule_min_stationary_ratio,
            loitering_rule_min_revisit_ratio=cfg.loitering_rule_min_revisit_ratio,
            loitering_rule_min_movement_extent=cfg.loitering_rule_min_movement_extent,
            loitering_rule_min_centroid_radius=cfg.loitering_rule_min_centroid_radius,
            loitering_rule_max_straightness=cfg.loitering_rule_max_straightness,
            loitering_rule_max_displacement_ratio=cfg.loitering_rule_max_displacement_ratio,
            loitering_max_neighbor_count=cfg.loitering_max_neighbor_count,
            loitering_neighbor_radius=cfg.loitering_neighbor_radius,
            running_model_min_avg_speed=cfg.running_model_min_avg_speed,
            running_model_min_p90_speed=cfg.running_model_min_p90_speed,
            running_model_min_movement_extent=cfg.running_model_min_movement_extent,
        )

        cap = cv2.VideoCapture(str(sequence.video_path))
        masks = _load_avenue_labels(sequence.label_path)
        frame_idx = 0

        seq_tp = 0
        seq_fp = 0
        seq_fn = 0
        seq_tn = 0
        seq_pred_frames = 0
        seq_gt_frames = 0
        seq_tracks = 0
        seq_alarm_overlap = 0
        seq_alarm_events = 0
        seq_alarm_types: dict[str, int] = {}
        saved_demo_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= len(masks):
                    break

                mask = masks[frame_idx]
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                alarms, track_infos = behavior.update(tracks)
                effective_alarms = _effective_alarms(alarms, track_infos)

                gt_anomaly = _mask_positive(mask)
                pred_anomaly = bool(effective_alarms)
                overlapping_alarm = any(
                    _track_overlaps_mask(track, mask)
                    for track in tracks
                    if any(int(track[4]) == int(alarm_track_id) for alarm_track_id, _ in effective_alarms)
                )

                if gt_anomaly:
                    seq_gt_frames += 1
                if pred_anomaly:
                    seq_pred_frames += 1
                    seq_alarm_events += len(effective_alarms)
                    for _, alarm_type in effective_alarms:
                        seq_alarm_types[alarm_type] = seq_alarm_types.get(alarm_type, 0) + 1
                    if overlapping_alarm:
                        seq_alarm_overlap += 1

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
                        roi=cfg.roi,
                        cross_line=cfg.cross_line,
                        track_infos=track_infos,
                    )
                    heat = np.zeros_like(vis)
                    heat[:, :, 2] = (mask > 0).astype(np.uint8) * 255
                    blended = cv2.addWeighted(vis, 1.0, heat, 0.35, 0.0)
                    output_path = cfg.demo_dir / f"{sequence.sequence_id}_{frame_idx + 1:04d}.jpg"
                    cv2.imwrite(str(output_path), blended)
                    saved_demo_frames += 1

                frame_idx += 1
        finally:
            cap.release()

        precision = _safe_divide(seq_tp, seq_tp + seq_fp)
        recall = _safe_divide(seq_tp, seq_tp + seq_fn)
        sequence_reports.append(
            {
                "sequence_id": sequence.sequence_id,
                "video_path": str(sequence.video_path),
                "label_path": str(sequence.label_path),
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
                "alarm_frames_overlapping_gt": seq_alarm_overlap,
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
        total_alarm_tracks_overlapping_gt += seq_alarm_overlap
        total_alarm_events += seq_alarm_events
        for alarm_type, count in seq_alarm_types.items():
            total_alarm_types[alarm_type] = total_alarm_types.get(alarm_type, 0) + count

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)
    report = {
        "config": {
            "avenue_root": str(cfg.avenue_root),
            "ground_truth_root": str(cfg.ground_truth_root),
            "model": str(cfg.model),
            "imgsz": cfg.imgsz,
            "max_det": cfg.max_det,
            "half": bool(cfg.half),
            "augment": bool(cfg.augment),
            "conf_threshold": cfg.conf_threshold,
            "track_high_thresh": cfg.track_high_thresh,
            "track_low_thresh": cfg.track_low_thresh,
            "new_track_thresh": cfg.new_track_thresh,
            "match_thresh": cfg.match_thresh,
            "low_match_thresh": cfg.low_match_thresh,
            "unconfirmed_match_thresh": cfg.unconfirmed_match_thresh,
            "score_fusion_weight": cfg.score_fusion_weight,
            "max_time_lost": cfg.max_time_lost,
            "min_box_area": cfg.min_box_area,
            "appearance_enabled": bool(cfg.appearance_enabled),
            "appearance_weight": float(cfg.appearance_weight),
            "appearance_ambiguity_margin": float(cfg.appearance_ambiguity_margin),
            "appearance_feature_mode": str(cfg.appearance_feature_mode),
            "appearance_hist_bins": list(cfg.appearance_hist_bins),
            "appearance_min_box_size": cfg.appearance_min_box_size,
            "appearance_reid_model": str(cfg.appearance_reid_model),
            "appearance_reid_weights": str(cfg.appearance_reid_weights) if cfg.appearance_reid_weights else "",
            "appearance_reid_device": cfg.appearance_reid_device,
            "appearance_reid_input_size": list(cfg.appearance_reid_input_size),
            "behavior_mode": cfg.behavior_mode,
            "behavior_model_path": str(cfg.behavior_model_path) if cfg.behavior_model_path else None,
            "behavior_secondary_model_path": str(cfg.behavior_secondary_model_path) if cfg.behavior_secondary_model_path else None,
            "behavior_ensemble_primary_weight": cfg.behavior_ensemble_primary_weight,
            "behavior_ensemble_mode": cfg.behavior_ensemble_mode,
            "behavior_model_score_thresh": cfg.behavior_model_score_thresh,
            "behavior_model_min_frames": cfg.behavior_model_min_frames,
            "behavior_secondary_loitering_only": bool(cfg.behavior_secondary_loitering_only),
            "loitering_hybrid_mode": cfg.loitering_hybrid_mode,
            "loitering_model_support_thresh": cfg.loitering_model_support_thresh,
            "loitering_model_score_thresh": cfg.loitering_model_score_thresh,
            "running_model_score_thresh": cfg.running_model_score_thresh,
            "loitering_model_min_frames": cfg.loitering_model_min_frames,
            "loitering_activate_frames": cfg.loitering_activate_frames,
            "loitering_support_activate_frames": cfg.loitering_support_activate_frames,
            "loitering_support_block_running": bool(cfg.loitering_support_block_running),
            "loitering_context_gate_support_only": bool(cfg.loitering_context_gate_support_only),
            "loitering_release_frames": cfg.loitering_release_frames,
            "loitering_model_max_avg_speed": cfg.loitering_model_max_avg_speed,
            "loitering_model_max_movement_extent": cfg.loitering_model_max_movement_extent,
            "loitering_model_max_centroid_radius": cfg.loitering_model_max_centroid_radius,
            "loitering_model_min_stationary_ratio": cfg.loitering_model_min_stationary_ratio,
            "loitering_model_min_revisit_ratio": cfg.loitering_model_min_revisit_ratio,
            "loitering_model_max_unique_cell_ratio": cfg.loitering_model_max_unique_cell_ratio,
            "loitering_model_min_max_cell_occupancy_ratio": cfg.loitering_model_min_max_cell_occupancy_ratio,
            "loitering_model_max_straightness": cfg.loitering_model_max_straightness,
            "loitering_rule_min_stationary_ratio": cfg.loitering_rule_min_stationary_ratio,
            "loitering_rule_min_revisit_ratio": cfg.loitering_rule_min_revisit_ratio,
            "loitering_rule_min_movement_extent": cfg.loitering_rule_min_movement_extent,
            "loitering_rule_min_centroid_radius": cfg.loitering_rule_min_centroid_radius,
            "loitering_rule_max_straightness": cfg.loitering_rule_max_straightness,
            "loitering_max_neighbor_count": cfg.loitering_max_neighbor_count,
            "loitering_neighbor_radius": cfg.loitering_neighbor_radius,
            "running_model_min_avg_speed": cfg.running_model_min_avg_speed,
            "running_model_min_p90_speed": cfg.running_model_min_p90_speed,
            "running_model_min_movement_extent": cfg.running_model_min_movement_extent,
            "enable_loitering": cfg.enable_loitering,
            "enable_running": cfg.enable_running,
            "enable_intrusion": cfg.enable_intrusion,
            "enable_cross_line": cfg.enable_cross_line,
            "loiter_frames": cfg.loiter_frames,
            "loiter_radius": cfg.loiter_radius,
            "loiter_speed": cfg.loiter_speed,
            "running_speed": cfg.running_speed,
            "running_frames": cfg.running_frames,
            "sequence_ids": list(cfg.sequence_ids),
        },
        "summary": {
            "sequences": len(sequence_reports),
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
            "alarm_frames_overlapping_gt": total_alarm_tracks_overlapping_gt,
            "alarm_type_counts": total_alarm_types,
            "avg_tracks_per_frame": round(_safe_divide(total_tracks, total_frames), 4),
        },
        "notes": [
            "CUHK Avenue is used here as a weakly supervised anomaly validation set, not as a direct running/loitering class-label training set.",
            "Metrics are frame-level anomaly metrics: a frame is predicted anomalous when at least one configured behavior stays active or a one-shot alarm is triggered.",
            "alarm_frames_overlapping_gt counts anomalous frames where at least one active-or-triggered alarm track overlaps the Avenue anomaly mask.",
        ],
        "sequences": sequence_reports,
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

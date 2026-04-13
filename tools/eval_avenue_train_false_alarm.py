from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from config import get_config
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite


def _build_components(app_config):
    detector = YOLODetector(
        model_path=str(app_config.model.model_path),
        conf=app_config.model.conf_threshold,
        classes=list(app_config.model.classes),
        device=app_config.model.device,
        imgsz=app_config.model.imgsz,
        max_det=app_config.model.max_det,
        half=app_config.model.half,
        augment=app_config.model.augment,
    )

    behavior_classifier = None
    if app_config.behavior.behavior_model_path:
        behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=app_config.behavior.behavior_model_path,
            min_frames_override=app_config.behavior.behavior_model_min_frames,
        )

    secondary_behavior_classifier = None
    if app_config.behavior.behavior_secondary_model_path:
        secondary_behavior_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=app_config.behavior.behavior_secondary_model_path,
            min_frames_override=app_config.behavior.behavior_model_min_frames,
        )

    return detector, behavior_classifier, secondary_behavior_classifier


def evaluate_train_false_alarm(config_path: str = "config.yaml", output_path: str | None = None) -> dict[str, object]:
    app = get_config(config_path)
    detector, behavior_classifier, secondary_behavior_classifier = _build_components(app)
    training_dir = PROJECT_ROOT / "data" / "CUHK_Avenue" / "Avenue Dataset" / "training_videos"
    if not training_dir.exists():
        raise FileNotFoundError(f"Avenue training videos not found: {training_dir}")

    total_frames = 0
    total_predicted_frames = 0
    total_alarm_events = 0
    total_tracks = 0
    total_alarm_type_counts: dict[str, int] = {}
    details: list[dict[str, object]] = []

    for video_path in sorted(training_dir.glob("*.avi")):
        tracker = ByteTrackerLite(
            track_high_thresh=app.tracker.track_high_thresh,
            track_low_thresh=app.tracker.track_low_thresh,
            new_track_thresh=app.tracker.new_track_thresh,
            match_thresh=app.tracker.match_thresh,
            low_match_thresh=app.tracker.low_match_thresh,
            unconfirmed_match_thresh=app.tracker.unconfirmed_match_thresh,
            score_fusion_weight=app.tracker.score_fusion_weight,
            max_time_lost=app.tracker.max_time_lost,
            min_box_area=app.tracker.min_box_area,
            appearance_enabled=app.tracker.appearance_enabled,
            appearance_weight=app.tracker.appearance_weight,
            appearance_ambiguity_margin=app.tracker.appearance_ambiguity_margin,
            appearance_feature_mode=app.tracker.appearance_feature_mode,
            appearance_hist_bins=app.tracker.appearance_hist_bins,
            appearance_min_box_size=app.tracker.appearance_min_box_size,
            appearance_reid_model=app.tracker.appearance_reid_model,
            appearance_reid_weights=app.tracker.appearance_reid_weights,
            appearance_reid_device=app.tracker.appearance_reid_device,
            appearance_reid_input_size=app.tracker.appearance_reid_input_size,
        )
        behavior = AbnormalDetector(
            roi=app.behavior.roi,
            behavior_mode=app.behavior.behavior_mode,
            enable_intrusion=app.behavior.enable_intrusion,
            enable_cross_line=app.behavior.enable_cross_line,
            enable_loitering=app.behavior.enable_loitering,
            enable_running=app.behavior.enable_running,
            intrusion_frames=app.behavior.intrusion_frames,
            cross_line=app.behavior.cross_line,
            loiter_frames=app.behavior.loiter_frames,
            loiter_radius=app.behavior.loiter_radius,
            loiter_speed=app.behavior.loiter_speed,
            running_speed=app.behavior.running_speed,
            running_frames=app.behavior.running_frames,
            behavior_classifier=behavior_classifier,
            secondary_behavior_classifier=secondary_behavior_classifier,
            behavior_ensemble_primary_weight=app.behavior.behavior_ensemble_primary_weight,
            behavior_ensemble_mode=app.behavior.behavior_ensemble_mode,
            behavior_model_score_thresh=app.behavior.behavior_model_score_thresh,
            behavior_model_min_frames=app.behavior.behavior_model_min_frames,
            behavior_model_max_tracks=app.behavior.behavior_model_max_tracks,
            behavior_model_resume_tracks=app.behavior.behavior_model_resume_tracks,
            behavior_secondary_max_tracks=app.behavior.behavior_secondary_max_tracks,
            behavior_secondary_resume_tracks=app.behavior.behavior_secondary_resume_tracks,
            behavior_secondary_loitering_only=app.behavior.behavior_secondary_loitering_only,
            behavior_model_eval_interval=app.behavior.behavior_model_eval_interval,
            loitering_hybrid_mode=app.behavior.loitering_hybrid_mode,
            loitering_model_support_thresh=app.behavior.loitering_model_support_thresh,
            loitering_model_score_thresh=app.behavior.loitering_model_score_thresh,
            running_model_score_thresh=app.behavior.running_model_score_thresh,
            loitering_model_min_frames=app.behavior.loitering_model_min_frames,
            loitering_activate_frames=app.behavior.loitering_activate_frames,
            loitering_support_activate_frames=app.behavior.loitering_support_activate_frames,
            loitering_support_block_running=app.behavior.loitering_support_block_running,
            loitering_context_gate_support_only=app.behavior.loitering_context_gate_support_only,
            loitering_release_frames=app.behavior.loitering_release_frames,
            loitering_model_max_avg_speed=app.behavior.loitering_model_max_avg_speed,
            loitering_model_min_movement_extent=app.behavior.loitering_model_min_movement_extent,
            loitering_model_min_centroid_radius=app.behavior.loitering_model_min_centroid_radius,
            loitering_model_max_movement_extent=app.behavior.loitering_model_max_movement_extent,
            loitering_model_max_centroid_radius=app.behavior.loitering_model_max_centroid_radius,
            loitering_model_min_stationary_ratio=app.behavior.loitering_model_min_stationary_ratio,
            loitering_model_min_revisit_ratio=app.behavior.loitering_model_min_revisit_ratio,
            loitering_model_max_unique_cell_ratio=app.behavior.loitering_model_max_unique_cell_ratio,
            loitering_model_min_max_cell_occupancy_ratio=app.behavior.loitering_model_min_max_cell_occupancy_ratio,
            loitering_model_max_straightness=app.behavior.loitering_model_max_straightness,
            loitering_rule_min_stationary_ratio=app.behavior.loitering_rule_min_stationary_ratio,
            loitering_rule_min_revisit_ratio=app.behavior.loitering_rule_min_revisit_ratio,
            loitering_rule_min_movement_extent=app.behavior.loitering_rule_min_movement_extent,
            loitering_rule_min_centroid_radius=app.behavior.loitering_rule_min_centroid_radius,
            loitering_rule_max_straightness=app.behavior.loitering_rule_max_straightness,
            loitering_rule_max_displacement_ratio=app.behavior.loitering_rule_max_displacement_ratio,
            loitering_max_neighbor_count=app.behavior.loitering_max_neighbor_count,
            loitering_neighbor_radius=app.behavior.loitering_neighbor_radius,
            running_model_min_avg_speed=app.behavior.running_model_min_avg_speed,
            running_model_min_p90_speed=app.behavior.running_model_min_p90_speed,
            running_model_min_movement_extent=app.behavior.running_model_min_movement_extent,
        )

        cap = cv2.VideoCapture(str(video_path))
        frames = 0
        predicted_frames = 0
        alarm_events = 0
        seq_tracks = 0
        seq_alarm_type_counts: dict[str, int] = {}
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames += 1
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                alarms, track_infos = behavior.update(tracks)
                effective_alarm_types: list[str] = []
                seen_pairs: set[tuple[int, str]] = set()
                for track_id, alarm_type in alarms:
                    key = (int(track_id), str(alarm_type))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    effective_alarm_types.append(str(alarm_type))
                for track_id, info in track_infos.items():
                    if bool(info.get("intrusion_active", False)):
                        key = (int(track_id), "intrusion")
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            effective_alarm_types.append("intrusion")
                    if bool(info.get("loitering_active", False)):
                        key = (int(track_id), "loitering")
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            effective_alarm_types.append("loitering")
                    if bool(info.get("running_active", False)):
                        key = (int(track_id), "running")
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            effective_alarm_types.append("running")

                if effective_alarm_types:
                    predicted_frames += 1
                    alarm_events += len(effective_alarm_types)
                    for alarm_type in effective_alarm_types:
                        seq_alarm_type_counts[alarm_type] = seq_alarm_type_counts.get(alarm_type, 0) + 1
                seq_tracks += len(tracks)
        finally:
            cap.release()

        total_frames += frames
        total_predicted_frames += predicted_frames
        total_alarm_events += alarm_events
        total_tracks += seq_tracks
        for alarm_type, count in seq_alarm_type_counts.items():
            total_alarm_type_counts[alarm_type] = total_alarm_type_counts.get(alarm_type, 0) + count

        details.append(
            {
                "sequence_id": video_path.stem,
                "video_path": str(video_path),
                "frames": frames,
                "predicted_anomaly_frames": predicted_frames,
                "false_alarm_rate": (predicted_frames / frames) if frames else 0.0,
                "alarm_events": alarm_events,
                "avg_tracks_per_frame": (seq_tracks / frames) if frames else 0.0,
                "alarm_type_counts": seq_alarm_type_counts,
            }
        )

    report = {
        "config_path": str((PROJECT_ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)),
        "videos": len(details),
        "frames": total_frames,
        "predicted_anomaly_frames": total_predicted_frames,
        "false_alarm_rate": (total_predicted_frames / total_frames) if total_frames else 0.0,
        "specificity": 1.0 - ((total_predicted_frames / total_frames) if total_frames else 0.0),
        "alarm_events": total_alarm_events,
        "avg_tracks_per_frame": (total_tracks / total_frames) if total_frames else 0.0,
        "alarm_type_counts": total_alarm_type_counts,
        "details": details,
    }
    if output_path:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = (PROJECT_ROOT / destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate false alarms on Avenue training videos.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--output-path", default="output/avenue/train_false_alarm_eval.json")
    args = parser.parse_args()

    result = evaluate_train_false_alarm(config_path=args.config_path, output_path=args.output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

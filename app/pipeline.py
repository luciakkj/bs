import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import cv2

from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from utils.visualization import draw
from services.event_logger import EventLogger
from services.snapshot_service import SnapshotService
from services.run_meta_service import RunMetaService
from services.video_export_service import VideoExportService


class VideoAnalyticsPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.started_at_ts = time.time()

        self.source_type = self._determine_source_type()
        self.source_name = self._determine_source_name()
        self.model_name = self._determine_model_name()
        self.runtime_profile = str(getattr(cfg.model, "runtime_profile", "balanced") or "balanced").lower()
        detector_predict_conf = cfg.model.predict_conf
        if detector_predict_conf is None:
            detector_predict_conf = min(
                float(cfg.model.conf_threshold),
                float(cfg.tracker.track_low_thresh),
            )
        self.detector_predict_conf = float(detector_predict_conf)
        self.runtime_augment = bool(cfg.model.augment)
        self.runtime_nms_iou = cfg.model.nms_iou
        self.runtime_track_high_thresh = float(cfg.tracker.track_high_thresh)
        self.runtime_new_track_thresh = float(cfg.tracker.new_track_thresh)
        self.runtime_match_thresh = float(cfg.tracker.match_thresh)
        if self.runtime_profile == "crowd_recall":
            self.detector_predict_conf = min(self.detector_predict_conf, 0.18)
            self.runtime_track_high_thresh = min(self.runtime_track_high_thresh, 0.42)
            self.runtime_new_track_thresh = min(self.runtime_new_track_thresh, 0.55)
        elif self.runtime_profile == "tracking_boost":
            self.runtime_augment = False
            self.detector_predict_conf = min(self.detector_predict_conf, 0.12)
            self.runtime_track_high_thresh = min(self.runtime_track_high_thresh, 0.38)
            self.runtime_new_track_thresh = min(self.runtime_new_track_thresh, 0.48)
            self.runtime_match_thresh = min(self.runtime_match_thresh, 0.78)

        self.detector = YOLODetector(
            cfg.model.model_path,
            conf=self.detector_predict_conf,
            classes=list(cfg.model.classes),
            device=cfg.model.device,
            imgsz=cfg.model.imgsz,
            max_det=cfg.model.max_det,
            nms_iou=self.runtime_nms_iou,
            half=cfg.model.half,
            augment=self.runtime_augment,
        )
        self.behavior_classifier = None
        if getattr(cfg.behavior, "behavior_model_path", ""):
            self.behavior_classifier = TrajectoryBehaviorClassifier(
                checkpoint_path=cfg.behavior.behavior_model_path,
                min_frames_override=cfg.behavior.behavior_model_min_frames,
            )
        self.secondary_behavior_classifier = None
        if getattr(cfg.behavior, "behavior_secondary_model_path", ""):
            self.secondary_behavior_classifier = TrajectoryBehaviorClassifier(
                checkpoint_path=cfg.behavior.behavior_secondary_model_path,
                min_frames_override=cfg.behavior.behavior_model_min_frames,
            )
        self.tracker = ByteTrackerLite(
            track_high_thresh=self.runtime_track_high_thresh,
            track_low_thresh=cfg.tracker.track_low_thresh,
            new_track_thresh=self.runtime_new_track_thresh,
            new_track_confirm_frames=cfg.tracker.new_track_confirm_frames,
            match_thresh=self.runtime_match_thresh,
            low_match_thresh=cfg.tracker.low_match_thresh,
            unconfirmed_match_thresh=cfg.tracker.unconfirmed_match_thresh,
            score_fusion_weight=cfg.tracker.score_fusion_weight,
            max_time_lost=cfg.tracker.max_time_lost,
            min_box_area=cfg.tracker.min_box_area,
            appearance_enabled=cfg.tracker.appearance_enabled,
            appearance_weight=cfg.tracker.appearance_weight,
            appearance_ambiguity_margin=cfg.tracker.appearance_ambiguity_margin,
            appearance_all_valid=cfg.tracker.appearance_all_valid,
            appearance_feature_mode=cfg.tracker.appearance_feature_mode,
            appearance_hist_bins=cfg.tracker.appearance_hist_bins,
            appearance_min_box_size=cfg.tracker.appearance_min_box_size,
            appearance_reid_model=cfg.tracker.appearance_reid_model,
            appearance_reid_weights=cfg.tracker.appearance_reid_weights,
            appearance_reid_device=cfg.tracker.appearance_reid_device,
            appearance_reid_input_size=cfg.tracker.appearance_reid_input_size,
            appearance_reid_flip_aug=cfg.tracker.appearance_reid_flip_aug,
            track_stability_weight=cfg.tracker.track_stability_weight,
            motion_gate_enabled=cfg.tracker.motion_gate_enabled,
            motion_gate_thresh=cfg.tracker.motion_gate_thresh,
            crowd_boost_enabled=cfg.tracker.crowd_boost_enabled,
            crowd_boost_det_count=cfg.tracker.crowd_boost_det_count,
            crowd_match_thresh=cfg.tracker.crowd_match_thresh,
            crowd_low_match_thresh=cfg.tracker.crowd_low_match_thresh,
            crowd_new_track_confirm_frames=cfg.tracker.crowd_new_track_confirm_frames,
            crowd_appearance_weight=cfg.tracker.crowd_appearance_weight,
            crowd_track_stability_weight=cfg.tracker.crowd_track_stability_weight,
            crowd_boost_min_small_ratio=cfg.tracker.crowd_boost_min_small_ratio,
            crowd_boost_max_median_area_ratio=cfg.tracker.crowd_boost_max_median_area_ratio,
            crowd_boost_small_area_ratio_thresh=cfg.tracker.crowd_boost_small_area_ratio_thresh,
            cmc_enabled=cfg.tracker.cmc_enabled,
            cmc_motion_model=cfg.tracker.cmc_motion_model,
            cmc_ecc_iterations=cfg.tracker.cmc_ecc_iterations,
            cmc_ecc_eps=cfg.tracker.cmc_ecc_eps,
            cmc_downscale=cfg.tracker.cmc_downscale,
        )
        self.behavior = AbnormalDetector(
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
            behavior_classifier=self.behavior_classifier,
            secondary_behavior_classifier=self.secondary_behavior_classifier,
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
            running_loitering_arb_enabled=cfg.behavior.running_loitering_arb_enabled,
            running_loitering_min_loitering_score=cfg.behavior.running_loitering_min_loitering_score,
            running_loitering_min_stationary_ratio=cfg.behavior.running_loitering_min_stationary_ratio,
            running_loitering_max_movement_extent=cfg.behavior.running_loitering_max_movement_extent,
            running_loitering_max_p90_speed=cfg.behavior.running_loitering_max_p90_speed,
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

        self.event_logger = None
        if cfg.output.enable_event_log:
            self.event_logger = EventLogger(
                output_dir=cfg.output.event_log_dir,
                run_id=self.run_id,
            )

        self.snapshot_service = None
        if cfg.output.enable_snapshot:
            self.snapshot_service = SnapshotService(
                output_root=cfg.output.snapshot_root,
                run_id=self.run_id,
                roi=cfg.behavior.roi,
            )

        self.video_export_service = None
        self.video_export_path = None

        self.run_meta_service = RunMetaService(
            output_dir=cfg.output.event_log_dir,
            run_id=self.run_id,
        )

        self.prev_time = time.time()
        self.smoothed_fps = 0.0
        self.last_alarm_log_time = {}

        self.total_frames = 0
        self.total_alarms = 0
        self.alarm_counts = defaultdict(int)

        self._save_run_meta()

    def _determine_source_type(self):
        if self.cfg.source.use_camera:
            return "camera"
        elif self.cfg.source.video_path:
            return "video"
        else:
            return "image_sequence"

    def _determine_source_name(self):
        if self.source_type == "camera":
            return f"camera_{self.cfg.source.camera_id}"

        if self.source_type == "video":
            return Path(self.cfg.source.video_path).name

        seq_path = Path(self.cfg.source.image_sequence_path)
        parts = seq_path.parts
        if "img1" in parts:
            img1_index = parts.index("img1")
            if img1_index - 1 >= 0:
                return parts[img1_index - 1]

        if seq_path.parent.name:
            return seq_path.parent.name

        return "image_sequence"

    def _determine_model_name(self):
        return Path(self.cfg.model.model_path).name

    def _build_run_meta(self, ended_at=None, duration_seconds=None, avg_fps=None):
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "source": {
                "source_type": self.source_type,
                "source_name": self.source_name,
                "use_camera": self.cfg.source.use_camera,
                "camera_id": self.cfg.source.camera_id,
                "video_path": self.cfg.source.video_path,
                "image_sequence_path": self.cfg.source.image_sequence_path,
            },
            "model": {
                "model_path": self.cfg.model.model_path,
                "model_name": self.model_name,
                "device": self.cfg.model.device,
                "imgsz": self.cfg.model.imgsz,
                "max_det": self.cfg.model.max_det,
                "half": self.cfg.model.half,
                "augment": self.runtime_augment,
                "runtime_profile": self.runtime_profile,
                "predict_conf": self.detector_predict_conf,
                "conf_threshold": self.cfg.model.conf_threshold,
                "classes": list(self.cfg.model.classes),
            },
            "tracker": {
                "track_high_thresh": self.runtime_track_high_thresh,
                "track_low_thresh": self.cfg.tracker.track_low_thresh,
                "new_track_thresh": self.runtime_new_track_thresh,
                "match_thresh": self.runtime_match_thresh,
                "low_match_thresh": self.cfg.tracker.low_match_thresh,
                "unconfirmed_match_thresh": self.cfg.tracker.unconfirmed_match_thresh,
                "score_fusion_weight": self.cfg.tracker.score_fusion_weight,
                "max_time_lost": self.cfg.tracker.max_time_lost,
                "min_box_area": self.cfg.tracker.min_box_area,
                "appearance_enabled": self.cfg.tracker.appearance_enabled,
                "appearance_weight": self.cfg.tracker.appearance_weight,
                "appearance_ambiguity_margin": self.cfg.tracker.appearance_ambiguity_margin,
                "appearance_feature_mode": self.cfg.tracker.appearance_feature_mode,
                "appearance_hist_bins": list(self.cfg.tracker.appearance_hist_bins),
                "appearance_min_box_size": self.cfg.tracker.appearance_min_box_size,
                "appearance_reid_model": self.cfg.tracker.appearance_reid_model,
                "appearance_reid_weights": self.cfg.tracker.appearance_reid_weights,
                "appearance_reid_device": self.cfg.tracker.appearance_reid_device,
                "appearance_reid_input_size": list(self.cfg.tracker.appearance_reid_input_size),
                "crowd_boost_enabled": self.cfg.tracker.crowd_boost_enabled,
                "crowd_boost_det_count": self.cfg.tracker.crowd_boost_det_count,
                "crowd_match_thresh": self.cfg.tracker.crowd_match_thresh,
                "crowd_low_match_thresh": self.cfg.tracker.crowd_low_match_thresh,
                "crowd_appearance_weight": self.cfg.tracker.crowd_appearance_weight,
                "crowd_track_stability_weight": self.cfg.tracker.crowd_track_stability_weight,
                "crowd_boost_min_small_ratio": self.cfg.tracker.crowd_boost_min_small_ratio,
                "crowd_boost_max_median_area_ratio": self.cfg.tracker.crowd_boost_max_median_area_ratio,
                "crowd_boost_small_area_ratio_thresh": self.cfg.tracker.crowd_boost_small_area_ratio_thresh,
            },
            "behavior": {
                "roi": list(self.cfg.behavior.roi),
                "roi_mode": getattr(self.cfg.behavior, "roi_mode", "fixed"),
                "behavior_mode": self.cfg.behavior.behavior_mode,
                "enable_intrusion": self.cfg.behavior.enable_intrusion,
                "enable_cross_line": self.cfg.behavior.enable_cross_line,
                "enable_loitering": self.cfg.behavior.enable_loitering,
                "enable_running": self.cfg.behavior.enable_running,
                "intrusion_frames": self.cfg.behavior.intrusion_frames,
                "cross_line": self.cfg.behavior.cross_line,
                "loiter_frames": self.cfg.behavior.loiter_frames,
                "loiter_radius": self.cfg.behavior.loiter_radius,
                "loiter_speed": self.cfg.behavior.loiter_speed,
                "running_speed": self.cfg.behavior.running_speed,
                "running_frames": self.cfg.behavior.running_frames,
                "behavior_model_path": self.cfg.behavior.behavior_model_path,
                "behavior_secondary_model_path": self.cfg.behavior.behavior_secondary_model_path,
                "behavior_ensemble_primary_weight": self.cfg.behavior.behavior_ensemble_primary_weight,
                "behavior_ensemble_mode": self.cfg.behavior.behavior_ensemble_mode,
                "behavior_model_score_thresh": self.cfg.behavior.behavior_model_score_thresh,
                "behavior_model_min_frames": self.cfg.behavior.behavior_model_min_frames,
                "behavior_model_max_tracks": self.cfg.behavior.behavior_model_max_tracks,
                "behavior_model_resume_tracks": self.cfg.behavior.behavior_model_resume_tracks,
                "behavior_secondary_max_tracks": self.cfg.behavior.behavior_secondary_max_tracks,
                "behavior_secondary_resume_tracks": self.cfg.behavior.behavior_secondary_resume_tracks,
                "behavior_secondary_loitering_only": self.cfg.behavior.behavior_secondary_loitering_only,
                "behavior_model_eval_interval": self.cfg.behavior.behavior_model_eval_interval,
                "loitering_hybrid_mode": self.cfg.behavior.loitering_hybrid_mode,
                "loitering_model_support_thresh": self.cfg.behavior.loitering_model_support_thresh,
                "loitering_model_score_thresh": self.cfg.behavior.loitering_model_score_thresh,
                "running_model_score_thresh": self.cfg.behavior.running_model_score_thresh,
                "loitering_model_min_frames": self.cfg.behavior.loitering_model_min_frames,
                "loitering_activate_frames": self.cfg.behavior.loitering_activate_frames,
                "loitering_support_activate_frames": self.cfg.behavior.loitering_support_activate_frames,
                "loitering_context_gate_support_only": self.cfg.behavior.loitering_context_gate_support_only,
                "loitering_release_frames": self.cfg.behavior.loitering_release_frames,
                "loitering_model_max_avg_speed": self.cfg.behavior.loitering_model_max_avg_speed,
                "loitering_model_min_movement_extent": self.cfg.behavior.loitering_model_min_movement_extent,
                "loitering_model_min_centroid_radius": self.cfg.behavior.loitering_model_min_centroid_radius,
                "loitering_model_max_movement_extent": self.cfg.behavior.loitering_model_max_movement_extent,
                "loitering_model_max_centroid_radius": self.cfg.behavior.loitering_model_max_centroid_radius,
                "loitering_model_min_stationary_ratio": self.cfg.behavior.loitering_model_min_stationary_ratio,
                "loitering_model_min_revisit_ratio": self.cfg.behavior.loitering_model_min_revisit_ratio,
                "loitering_model_max_unique_cell_ratio": self.cfg.behavior.loitering_model_max_unique_cell_ratio,
                "loitering_model_min_max_cell_occupancy_ratio": self.cfg.behavior.loitering_model_min_max_cell_occupancy_ratio,
                "loitering_model_max_straightness": self.cfg.behavior.loitering_model_max_straightness,
                "loitering_rule_min_stationary_ratio": self.cfg.behavior.loitering_rule_min_stationary_ratio,
                "loitering_rule_min_revisit_ratio": self.cfg.behavior.loitering_rule_min_revisit_ratio,
                "loitering_rule_min_movement_extent": self.cfg.behavior.loitering_rule_min_movement_extent,
                "loitering_rule_min_centroid_radius": self.cfg.behavior.loitering_rule_min_centroid_radius,
                "loitering_rule_max_straightness": self.cfg.behavior.loitering_rule_max_straightness,
                "loitering_rule_max_displacement_ratio": self.cfg.behavior.loitering_rule_max_displacement_ratio,
                "loitering_max_neighbor_count": self.cfg.behavior.loitering_max_neighbor_count,
                "loitering_neighbor_radius": self.cfg.behavior.loitering_neighbor_radius,
                "running_model_min_avg_speed": self.cfg.behavior.running_model_min_avg_speed,
                "running_model_min_p90_speed": self.cfg.behavior.running_model_min_p90_speed,
                "running_model_min_movement_extent": self.cfg.behavior.running_model_min_movement_extent,
            },
            "display": {
                "window_name": self.cfg.display.window_name,
                "quit_key": self.cfg.display.quit_key,
                "fps_smoothing": self.cfg.display.fps_smoothing,
            },
            "log": {
                "alarm_cooldown_seconds": self.cfg.log.alarm_cooldown_seconds,
            },
            "output": {
                "enable_event_log": self.cfg.output.enable_event_log,
                "enable_snapshot": self.cfg.output.enable_snapshot,
                "enable_video_export": self.cfg.output.enable_video_export,
                "event_log_dir": self.cfg.output.event_log_dir,
                "snapshot_root": self.cfg.output.snapshot_root,
                "video_output_dir": self.cfg.output.video_output_dir,
                "video_export_fps": self.cfg.output.video_export_fps,
                "video_codec": self.cfg.output.video_codec,
                "video_export_path": self.video_export_path,
            },
            "summary": {
                "total_frames": self.total_frames,
                "total_alarms": self.total_alarms,
                "alarm_counts": dict(self.alarm_counts),
                "duration_seconds": duration_seconds,
                "avg_fps": avg_fps,
            },
        }

    def _save_run_meta(self, ended_at=None, duration_seconds=None, avg_fps=None):
        meta = self._build_run_meta(
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            avg_fps=avg_fps,
        )
        self.run_meta_service.save(meta)

    def _build_capture(self):
        if self.cfg.source.use_camera:
            return cv2.VideoCapture(self.cfg.source.camera_id)
        elif self.cfg.source.video_path:
            return cv2.VideoCapture(self.cfg.source.video_path)
        else:
            return cv2.VideoCapture(self.cfg.source.image_sequence_path)

    def _resolve_export_fps(self, cap):
        configured_fps = float(getattr(self.cfg.output, "video_export_fps", 0.0) or 0.0)
        if configured_fps > 0:
            return configured_fps

        source_fps = 0.0
        if cap is not None:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if 1.0 <= source_fps <= 240.0:
            return source_fps
        return 25.0

    def _ensure_video_export_service(self, frame, cap=None):
        if not getattr(self.cfg.output, "enable_video_export", False):
            return
        if self.video_export_service is not None:
            return

        export_fps = self._resolve_export_fps(cap)
        self.video_export_service = VideoExportService(
            output_root=self.cfg.output.video_output_dir,
            run_id=self.run_id,
            source_name=self.source_name,
            fps=export_fps,
            codec=self.cfg.output.video_codec,
        )
        self.video_export_path = self.video_export_service.output_path
        self.video_export_service.open(frame)

    def _select_roi_if_needed(self, frame):
        roi_mode = getattr(self.cfg.behavior, "roi_mode", "fixed")

        if roi_mode != "interactive":
            return

        preview = frame.copy()
        tip = "Select ROI, then press ENTER/SPACE. Press C to keep current ROI."
        cv2.putText(
            preview,
            tip,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        roi = cv2.selectROI("Select ROI", preview, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return

        new_roi = (int(x), int(y), int(x + w), int(y + h))

        self.cfg.behavior.roi = new_roi
        self.behavior.roi = new_roi
        self.cfg.behavior.enable_intrusion = True
        self.behavior.enable_intrusion = True

        if self.snapshot_service is not None:
            self.snapshot_service.roi = new_roi

        self._save_run_meta()

    def _calc_fps(self):
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        alpha = min(0.99, max(0.0, float(getattr(self.cfg.display, "fps_smoothing", 0.9))))
        if self.smoothed_fps <= 0:
            self.smoothed_fps = fps
        else:
            self.smoothed_fps = alpha * self.smoothed_fps + (1.0 - alpha) * fps
        return self.smoothed_fps

    def _should_log_alarm(self, track_id, alarm_type):
        now = time.time()
        key = (int(track_id), str(alarm_type))
        cooldown = self.cfg.log.alarm_cooldown_seconds

        last_time = self.last_alarm_log_time.get(key)
        if last_time is None:
            self.last_alarm_log_time[key] = now
            return True

        if now - last_time >= cooldown:
            self.last_alarm_log_time[key] = now
            return True

        return False

    def _record_alarm_stats(self, alarm_type):
        self.total_alarms += 1
        self.alarm_counts[str(alarm_type)] += 1

    def _build_event(self, track_id, alarm_type, track_infos):
        info = track_infos.get(track_id, {})

        return {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "track_id": int(track_id),
            "alarm_type": str(alarm_type),
            "bbox": info.get("bbox"),
            "speed": info.get("speed"),
            "inside_roi": info.get("inside_roi"),
            "dwell_frames": info.get("dwell_frames"),
        }

    def _log_alarms(self, alarms, track_infos, frame):
        for track_id, alarm_type in alarms:
            if not self._should_log_alarm(track_id, alarm_type):
                continue

            self._record_alarm_stats(alarm_type)

            event = self._build_event(track_id, alarm_type, track_infos)

            if self.snapshot_service is not None:
                info = track_infos.get(track_id, {})
                bbox = info.get("bbox")
                frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                snapshot_path = self.snapshot_service.save(
                    frame=frame,
                    alarm_type=alarm_type,
                    track_id=track_id,
                    bbox=bbox,
                    frame_timestamp=frame_timestamp,
                )
                event["snapshot_path"] = snapshot_path
            else:
                event["snapshot_path"] = None

            if self.event_logger is not None:
                self.event_logger.log(event)

    def _process_frame(self, frame, cap=None):
        self.total_frames += 1

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame=frame)
        alarms, track_infos = self.behavior.update(tracks)

        self._log_alarms(alarms, track_infos, frame)

        fps = self._calc_fps()

        vis = draw(
            frame=frame,
            tracks=tracks,
            alarms=alarms,
            fps=fps,
            roi=self.cfg.behavior.roi,
            cross_line=self.cfg.behavior.cross_line,
            track_infos=track_infos,
        )
        if getattr(self.cfg.output, "enable_video_export", False):
            self._ensure_video_export_service(vis, cap=cap)
            if self.video_export_service is not None:
                self.video_export_service.write(vis)
        return vis

    def _handle_key(self):
        key = cv2.waitKey(1) & 0xFF
        return key == ord(self.cfg.display.quit_key)

    def run(self):
        cap = self._build_capture()
        if not cap.isOpened():
            print("无法打开视频源")
            return

        try:
            ret, first_frame = cap.read()
            if not ret:
                print("无法读取首帧")
                return

            self._select_roi_if_needed(first_frame)

            first_vis = self._process_frame(first_frame, cap=cap)
            cv2.imshow(self.cfg.display.window_name, first_vis)
            if self._handle_key():
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                vis = self._process_frame(frame, cap=cap)
                cv2.imshow(self.cfg.display.window_name, vis)

                if self._handle_key():
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

            if self.snapshot_service is not None:
                self.snapshot_service.close()
            if self.video_export_service is not None:
                self.video_export_service.close()

            ended_at = datetime.now().isoformat(timespec="seconds")
            duration_seconds = max(time.time() - self.started_at_ts, 1e-6)
            avg_fps = self.total_frames / duration_seconds if self.total_frames > 0 else 0.0

            self._save_run_meta(
                ended_at=ended_at,
                duration_seconds=round(duration_seconds, 3),
                avg_fps=round(avg_fps, 3),
            )

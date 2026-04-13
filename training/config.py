from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(slots=True)
class PrepareConfig:
    mot_root: Path = field(default_factory=lambda: resolve_path("data/MOT17"))
    output_dir: Path = field(default_factory=lambda: resolve_path("data/processed/mot17_person"))
    split_name: str = "train"
    detector_filter: str | None = "FRCNN"
    include_classes: tuple[int, ...] = (1,)
    min_visibility: float = 0.25
    val_ratio: float = 0.25
    val_sequences: tuple[str, ...] = ()
    overwrite: bool = False
    dense_small_repeat_factor: int = 1
    dense_small_max_repeat_frames: int | None = None
    dense_small_min_gt_count: int = 12
    dense_small_min_small_ratio: float = 0.7
    dense_small_max_median_area_ratio: float = 0.002
    small_box_area_ratio_thresh: float = 0.002
    dense_small_crop_enable: bool = False
    dense_small_crop_max_frames: int | None = None
    dense_small_crop_width_ratio: float = 0.5
    dense_small_crop_height_ratio: float = 0.5
    dense_small_crop_min_boxes: int = 6

    def resolved(self) -> "PrepareConfig":
        return PrepareConfig(
            mot_root=resolve_path(self.mot_root),
            output_dir=resolve_path(self.output_dir),
            split_name=self.split_name,
            detector_filter=self.detector_filter,
            include_classes=tuple(self.include_classes),
            min_visibility=float(self.min_visibility),
            val_ratio=float(self.val_ratio),
            val_sequences=tuple(self.val_sequences),
            overwrite=bool(self.overwrite),
            dense_small_repeat_factor=max(1, int(self.dense_small_repeat_factor)),
            dense_small_max_repeat_frames=(
                max(1, int(self.dense_small_max_repeat_frames))
                if self.dense_small_max_repeat_frames is not None
                else None
            ),
            dense_small_min_gt_count=max(1, int(self.dense_small_min_gt_count)),
            dense_small_min_small_ratio=float(self.dense_small_min_small_ratio),
            dense_small_max_median_area_ratio=float(self.dense_small_max_median_area_ratio),
            small_box_area_ratio_thresh=float(self.small_box_area_ratio_thresh),
            dense_small_crop_enable=bool(self.dense_small_crop_enable),
            dense_small_crop_max_frames=(
                max(1, int(self.dense_small_crop_max_frames))
                if self.dense_small_crop_max_frames is not None
                else None
            ),
            dense_small_crop_width_ratio=float(self.dense_small_crop_width_ratio),
            dense_small_crop_height_ratio=float(self.dense_small_crop_height_ratio),
            dense_small_crop_min_boxes=max(1, int(self.dense_small_crop_min_boxes)),
        )


@dataclass(slots=True)
class TrainConfig:
    data: Path = field(default_factory=lambda: resolve_path("data/processed/mot17_person/mot17_person.yaml"))
    model: Path = field(default_factory=lambda: resolve_path("models/yolov8n.pt"))
    project: Path = field(default_factory=lambda: resolve_path("output/training"))
    name: str = "mot17_person"
    epochs: int = 50
    imgsz: int = 960
    batch: int = 8
    device: str | int | None = None
    workers: int = 4
    patience: int = 20

    def resolved(self) -> "TrainConfig":
        return TrainConfig(
            data=resolve_path(self.data),
            model=resolve_path(self.model),
            project=resolve_path(self.project),
            name=self.name,
            epochs=int(self.epochs),
            imgsz=int(self.imgsz),
            batch=int(self.batch),
            device=self.device,
            workers=int(self.workers),
            patience=int(self.patience),
        )


@dataclass(slots=True)
class ValidateConfig:
    data: Path = field(default_factory=lambda: resolve_path("data/processed/mot17_person/mot17_person.yaml"))
    weights: Path = field(default_factory=lambda: resolve_path("models/yolov8n.pt"))
    imgsz: int = 960
    batch: int = 8
    device: str | int | None = None
    split: str = "val"
    workers: int = 0

    def resolved(self) -> "ValidateConfig":
        return ValidateConfig(
            data=resolve_path(self.data),
            weights=resolve_path(self.weights),
            imgsz=int(self.imgsz),
            batch=int(self.batch),
            device=self.device,
            split=self.split,
            workers=int(self.workers),
        )


@dataclass(slots=True)
class MOTTrackingEvalConfig:
    config_path: Path = field(default_factory=lambda: resolve_path("config.yaml"))
    mot_root: Path = field(default_factory=lambda: resolve_path("data/MOT17"))
    output_path: Path = field(default_factory=lambda: resolve_path("output/tracking/mot17_tracking_eval.json"))
    save_mot_dir: Path | None = field(default_factory=lambda: resolve_path("output/tracking/mot17_tracking_eval_tracks"))
    split_name: str = "train"
    detector_filter: str | None = "FRCNN"
    sequence_names: tuple[str, ...] = ()
    sequence_runtime_overrides_path: Path | None = None
    include_classes: tuple[int, ...] = (1,)
    min_visibility: float = 0.0
    min_iou: float = 0.5
    max_frames: int | None = None
    apply_runtime_profile: bool = True
    model_path: Path | None = None
    device: str | int | None = None
    imgsz: int | None = None
    max_det: int | None = None
    nms_iou: float | None = None
    half: bool | None = None
    augment: bool | None = None
    predict_conf: float | None = None
    track_high_thresh: float | None = None
    track_low_thresh: float | None = None
    new_track_thresh: float | None = None
    new_track_confirm_frames: int | None = None
    match_thresh: float | None = None
    low_match_thresh: float | None = None
    unconfirmed_match_thresh: float | None = None
    score_fusion_weight: float | None = None
    max_time_lost: int | None = None
    min_box_area: float | None = None
    appearance_enabled: bool | None = None
    appearance_weight: float | None = None
    appearance_ambiguity_margin: float | None = None
    appearance_all_valid: bool | None = None
    appearance_feature_mode: str | None = None
    appearance_hist_bins: tuple[int, int, int] | None = None
    appearance_min_box_size: int | None = None
    appearance_reid_model: str | None = None
    appearance_reid_weights: Path | None = None
    appearance_reid_device: str | None = None
    appearance_reid_input_size: tuple[int, int] | None = None
    appearance_reid_flip_aug: bool | None = None
    track_stability_weight: float | None = None
    motion_gate_enabled: bool | None = None
    motion_gate_thresh: float | None = None
    crowd_boost_enabled: bool | None = None
    crowd_boost_det_count: int | None = None
    crowd_match_thresh: float | None = None
    crowd_low_match_thresh: float | None = None
    crowd_new_track_confirm_frames: int | None = None
    crowd_appearance_weight: float | None = None
    crowd_track_stability_weight: float | None = None
    crowd_boost_min_small_ratio: float | None = None
    crowd_boost_max_median_area_ratio: float | None = None
    crowd_boost_small_area_ratio_thresh: float | None = None

    def resolved(self) -> "MOTTrackingEvalConfig":
        return MOTTrackingEvalConfig(
            config_path=resolve_path(self.config_path),
            mot_root=resolve_path(self.mot_root),
            output_path=resolve_path(self.output_path),
            save_mot_dir=resolve_path(self.save_mot_dir) if self.save_mot_dir else None,
            split_name=str(self.split_name),
            detector_filter=str(self.detector_filter) if self.detector_filter else None,
            sequence_names=tuple(str(item) for item in self.sequence_names),
            sequence_runtime_overrides_path=(
                resolve_path(self.sequence_runtime_overrides_path)
                if self.sequence_runtime_overrides_path
                else None
            ),
            include_classes=tuple(int(item) for item in self.include_classes),
            min_visibility=float(self.min_visibility),
            min_iou=float(self.min_iou),
            max_frames=int(self.max_frames) if self.max_frames is not None else None,
            apply_runtime_profile=bool(self.apply_runtime_profile),
            model_path=resolve_path(self.model_path) if self.model_path else None,
            device=self.device,
            imgsz=int(self.imgsz) if self.imgsz is not None else None,
            max_det=int(self.max_det) if self.max_det is not None else None,
            nms_iou=float(self.nms_iou) if self.nms_iou is not None else None,
            half=bool(self.half) if self.half is not None else None,
            augment=bool(self.augment) if self.augment is not None else None,
            predict_conf=float(self.predict_conf) if self.predict_conf is not None else None,
            track_high_thresh=float(self.track_high_thresh) if self.track_high_thresh is not None else None,
            track_low_thresh=float(self.track_low_thresh) if self.track_low_thresh is not None else None,
            new_track_thresh=float(self.new_track_thresh) if self.new_track_thresh is not None else None,
            new_track_confirm_frames=int(self.new_track_confirm_frames) if self.new_track_confirm_frames is not None else None,
            match_thresh=float(self.match_thresh) if self.match_thresh is not None else None,
            low_match_thresh=float(self.low_match_thresh) if self.low_match_thresh is not None else None,
            unconfirmed_match_thresh=float(self.unconfirmed_match_thresh) if self.unconfirmed_match_thresh is not None else None,
            score_fusion_weight=float(self.score_fusion_weight) if self.score_fusion_weight is not None else None,
            max_time_lost=int(self.max_time_lost) if self.max_time_lost is not None else None,
            min_box_area=float(self.min_box_area) if self.min_box_area is not None else None,
            appearance_enabled=bool(self.appearance_enabled) if self.appearance_enabled is not None else None,
            appearance_weight=float(self.appearance_weight) if self.appearance_weight is not None else None,
            appearance_ambiguity_margin=float(self.appearance_ambiguity_margin) if self.appearance_ambiguity_margin is not None else None,
            appearance_all_valid=bool(self.appearance_all_valid) if self.appearance_all_valid is not None else None,
            appearance_feature_mode=str(self.appearance_feature_mode) if self.appearance_feature_mode is not None else None,
            appearance_hist_bins=tuple(int(value) for value in self.appearance_hist_bins) if self.appearance_hist_bins is not None else None,
            appearance_min_box_size=int(self.appearance_min_box_size) if self.appearance_min_box_size is not None else None,
            appearance_reid_model=str(self.appearance_reid_model) if self.appearance_reid_model is not None else None,
            appearance_reid_weights=resolve_path(self.appearance_reid_weights) if self.appearance_reid_weights else None,
            appearance_reid_device=str(self.appearance_reid_device) if self.appearance_reid_device is not None else None,
            appearance_reid_input_size=tuple(int(value) for value in self.appearance_reid_input_size) if self.appearance_reid_input_size is not None else None,
            appearance_reid_flip_aug=bool(self.appearance_reid_flip_aug) if self.appearance_reid_flip_aug is not None else None,
            track_stability_weight=float(self.track_stability_weight) if self.track_stability_weight is not None else None,
            motion_gate_enabled=bool(self.motion_gate_enabled) if self.motion_gate_enabled is not None else None,
            motion_gate_thresh=float(self.motion_gate_thresh) if self.motion_gate_thresh is not None else None,
            crowd_boost_enabled=bool(self.crowd_boost_enabled) if self.crowd_boost_enabled is not None else None,
            crowd_boost_det_count=int(self.crowd_boost_det_count) if self.crowd_boost_det_count is not None else None,
            crowd_match_thresh=float(self.crowd_match_thresh) if self.crowd_match_thresh is not None else None,
            crowd_low_match_thresh=float(self.crowd_low_match_thresh) if self.crowd_low_match_thresh is not None else None,
            crowd_new_track_confirm_frames=int(self.crowd_new_track_confirm_frames) if self.crowd_new_track_confirm_frames is not None else None,
            crowd_appearance_weight=float(self.crowd_appearance_weight) if self.crowd_appearance_weight is not None else None,
            crowd_track_stability_weight=float(self.crowd_track_stability_weight) if self.crowd_track_stability_weight is not None else None,
            crowd_boost_min_small_ratio=float(self.crowd_boost_min_small_ratio) if self.crowd_boost_min_small_ratio is not None else None,
            crowd_boost_max_median_area_ratio=float(self.crowd_boost_max_median_area_ratio) if self.crowd_boost_max_median_area_ratio is not None else None,
            crowd_boost_small_area_ratio_thresh=float(self.crowd_boost_small_area_ratio_thresh) if self.crowd_boost_small_area_ratio_thresh is not None else None,
        )


@dataclass(slots=True)
class CalibrationConfig:
    mot_root: Path = field(default_factory=lambda: resolve_path("data/MOT17"))
    output_path: Path = field(default_factory=lambda: resolve_path("output/calibration/behavior_thresholds.json"))
    split_name: str = "train"
    detector_filter: str | None = "FRCNN"
    include_classes: tuple[int, ...] = (1,)
    min_visibility: float = 0.25

    def resolved(self) -> "CalibrationConfig":
        return CalibrationConfig(
            mot_root=resolve_path(self.mot_root),
            output_path=resolve_path(self.output_path),
            split_name=self.split_name,
            detector_filter=self.detector_filter,
            include_classes=tuple(self.include_classes),
            min_visibility=float(self.min_visibility),
        )


@dataclass(slots=True)
class AvenueValidationConfig:
    avenue_root: Path = field(default_factory=lambda: resolve_path("data/CUHK_Avenue/Avenue Dataset"))
    ground_truth_root: Path = field(default_factory=lambda: resolve_path("data/CUHK_Avenue/ground_truth_demo/testing_label_mask"))
    model: Path = field(default_factory=lambda: resolve_path("models/yolov8n.pt"))
    output_path: Path = field(default_factory=lambda: resolve_path("output/avenue/validation_report.json"))
    device: str | int | None = None
    imgsz: int | None = None
    max_det: int | None = None
    half: bool = False
    augment: bool = False
    conf_threshold: float = 0.1
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    low_match_thresh: float = 0.5
    unconfirmed_match_thresh: float = 0.7
    score_fusion_weight: float = 0.0
    max_time_lost: int = 30
    min_box_area: float = 10.0
    appearance_enabled: bool = False
    appearance_weight: float = 0.25
    appearance_ambiguity_margin: float = 0.05
    appearance_all_valid: bool = False
    appearance_feature_mode: str = "hsv"
    appearance_hist_bins: tuple[int, int, int] = (8, 4, 4)
    appearance_min_box_size: int = 16
    appearance_reid_model: str = "mobilenet_v3_small"
    appearance_reid_weights: Path | None = None
    appearance_reid_device: str | None = None
    appearance_reid_input_size: tuple[int, int] = (256, 128)
    appearance_reid_flip_aug: bool = False
    motion_gate_enabled: bool = False
    motion_gate_thresh: float = 9.4877
    crowd_boost_enabled: bool = False
    crowd_boost_det_count: int = 12
    crowd_match_thresh: float | None = None
    crowd_low_match_thresh: float | None = None
    crowd_appearance_weight: float | None = None
    crowd_track_stability_weight: float | None = None
    crowd_boost_min_small_ratio: float | None = None
    crowd_boost_max_median_area_ratio: float | None = None
    crowd_boost_small_area_ratio_thresh: float = 0.002
    behavior_mode: str = "rules"
    behavior_model_path: Path | None = None
    behavior_secondary_model_path: Path | None = None
    behavior_ensemble_primary_weight: float = 1.0
    behavior_ensemble_mode: str = "weighted"
    behavior_model_score_thresh: float = 0.55
    behavior_model_min_frames: int = 24
    behavior_model_max_tracks: int = 32
    behavior_model_resume_tracks: int = 26
    behavior_secondary_max_tracks: int = 24
    behavior_secondary_resume_tracks: int = 18
    behavior_secondary_loitering_only: bool = True
    behavior_model_eval_interval: int = 3
    loitering_hybrid_mode: str = "union"
    loitering_model_support_thresh: float = 0.0
    loitering_model_score_thresh: float = 0.97
    running_model_score_thresh: float = 0.85
    loitering_model_min_frames: int = 72
    loitering_activate_frames: int = 1
    loitering_support_activate_frames: int = 0
    loitering_support_block_running: bool = False
    loitering_context_gate_support_only: bool = False
    running_loitering_arb_enabled: bool = False
    running_loitering_min_loitering_score: float = 0.72
    running_loitering_min_stationary_ratio: float = 0.90
    running_loitering_max_movement_extent: float = 50.0
    running_loitering_max_p90_speed: float = 3.0
    loitering_release_frames: int = 1
    loitering_model_max_avg_speed: float = 2.2
    loitering_model_min_movement_extent: float = 0.0
    loitering_model_min_centroid_radius: float = 0.0
    loitering_model_max_movement_extent: float = 55.0
    loitering_model_max_centroid_radius: float = 28.0
    loitering_model_min_stationary_ratio: float = 0.0
    loitering_model_min_revisit_ratio: float = 0.0
    loitering_model_max_unique_cell_ratio: float = 1.0
    loitering_model_min_max_cell_occupancy_ratio: float = 0.0
    loitering_model_max_straightness: float = 1.0
    loitering_rule_min_stationary_ratio: float = 0.0
    loitering_rule_min_revisit_ratio: float = 0.0
    loitering_rule_min_movement_extent: float = 0.0
    loitering_rule_min_centroid_radius: float = 0.0
    loitering_rule_max_straightness: float = 1.0
    loitering_rule_max_displacement_ratio: float = 1.0
    loitering_max_neighbor_count: int = -1
    loitering_neighbor_radius: float = 80.0
    running_model_min_avg_speed: float = 8.0
    running_model_min_p90_speed: float = 16.0
    running_model_min_movement_extent: float = 120.0
    enable_loitering: bool = True
    enable_running: bool = True
    enable_intrusion: bool = False
    enable_cross_line: bool = False
    roi: tuple[int, int, int, int] | None = None
    cross_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    loiter_frames: int = 168
    loiter_radius: float = 40.13
    loiter_speed: float = 1.45
    running_speed: float = 15.57
    running_frames: int = 3
    max_videos: int | None = None
    sequence_ids: tuple[str, ...] = ()
    save_demo_frames: bool = False
    demo_dir: Path = field(default_factory=lambda: resolve_path("output/avenue/demo"))

    def resolved(self) -> "AvenueValidationConfig":
        return AvenueValidationConfig(
            avenue_root=resolve_path(self.avenue_root),
            ground_truth_root=resolve_path(self.ground_truth_root),
            model=resolve_path(self.model),
            output_path=resolve_path(self.output_path),
            device=self.device,
            imgsz=int(self.imgsz) if self.imgsz is not None else None,
            max_det=int(self.max_det) if self.max_det is not None else None,
            half=bool(self.half),
            augment=bool(self.augment),
            conf_threshold=float(self.conf_threshold),
            track_high_thresh=float(self.track_high_thresh),
            track_low_thresh=float(self.track_low_thresh),
            new_track_thresh=float(self.new_track_thresh),
            match_thresh=float(self.match_thresh),
            low_match_thresh=float(self.low_match_thresh),
            unconfirmed_match_thresh=float(self.unconfirmed_match_thresh),
            score_fusion_weight=float(self.score_fusion_weight),
            max_time_lost=int(self.max_time_lost),
            min_box_area=float(self.min_box_area),
            appearance_enabled=bool(self.appearance_enabled),
            appearance_weight=float(self.appearance_weight),
            appearance_ambiguity_margin=float(self.appearance_ambiguity_margin),
            appearance_all_valid=bool(self.appearance_all_valid),
            appearance_feature_mode=str(self.appearance_feature_mode),
            appearance_hist_bins=tuple(int(value) for value in self.appearance_hist_bins),
            appearance_min_box_size=int(self.appearance_min_box_size),
            appearance_reid_model=str(self.appearance_reid_model),
            appearance_reid_weights=resolve_path(self.appearance_reid_weights) if self.appearance_reid_weights else None,
            appearance_reid_device=str(self.appearance_reid_device) if self.appearance_reid_device is not None else None,
            appearance_reid_input_size=tuple(int(value) for value in self.appearance_reid_input_size),
            appearance_reid_flip_aug=bool(self.appearance_reid_flip_aug),
            motion_gate_enabled=bool(self.motion_gate_enabled),
            motion_gate_thresh=float(self.motion_gate_thresh),
            crowd_boost_enabled=bool(self.crowd_boost_enabled),
            crowd_boost_det_count=int(self.crowd_boost_det_count),
            crowd_match_thresh=float(self.crowd_match_thresh) if self.crowd_match_thresh is not None else None,
            crowd_low_match_thresh=float(self.crowd_low_match_thresh) if self.crowd_low_match_thresh is not None else None,
            crowd_appearance_weight=float(self.crowd_appearance_weight) if self.crowd_appearance_weight is not None else None,
            crowd_track_stability_weight=float(self.crowd_track_stability_weight) if self.crowd_track_stability_weight is not None else None,
            crowd_boost_min_small_ratio=float(self.crowd_boost_min_small_ratio) if self.crowd_boost_min_small_ratio is not None else None,
            crowd_boost_max_median_area_ratio=float(self.crowd_boost_max_median_area_ratio) if self.crowd_boost_max_median_area_ratio is not None else None,
            crowd_boost_small_area_ratio_thresh=float(self.crowd_boost_small_area_ratio_thresh),
            behavior_mode=str(self.behavior_mode),
            behavior_model_path=resolve_path(self.behavior_model_path) if self.behavior_model_path else None,
            behavior_secondary_model_path=resolve_path(self.behavior_secondary_model_path) if self.behavior_secondary_model_path else None,
            behavior_ensemble_primary_weight=float(self.behavior_ensemble_primary_weight),
            behavior_ensemble_mode=str(self.behavior_ensemble_mode),
            behavior_model_score_thresh=float(self.behavior_model_score_thresh),
            behavior_model_min_frames=int(self.behavior_model_min_frames),
            behavior_model_max_tracks=int(self.behavior_model_max_tracks),
            behavior_model_resume_tracks=int(self.behavior_model_resume_tracks),
            behavior_secondary_max_tracks=int(self.behavior_secondary_max_tracks),
            behavior_secondary_resume_tracks=int(self.behavior_secondary_resume_tracks),
            behavior_secondary_loitering_only=bool(self.behavior_secondary_loitering_only),
            behavior_model_eval_interval=int(self.behavior_model_eval_interval),
            loitering_hybrid_mode=str(self.loitering_hybrid_mode),
            loitering_model_support_thresh=float(self.loitering_model_support_thresh),
            loitering_model_score_thresh=float(self.loitering_model_score_thresh),
            running_model_score_thresh=float(self.running_model_score_thresh),
            loitering_model_min_frames=int(self.loitering_model_min_frames),
            loitering_activate_frames=int(self.loitering_activate_frames),
            loitering_support_activate_frames=int(self.loitering_support_activate_frames),
            loitering_support_block_running=bool(self.loitering_support_block_running),
            loitering_context_gate_support_only=bool(self.loitering_context_gate_support_only),
            running_loitering_arb_enabled=bool(self.running_loitering_arb_enabled),
            running_loitering_min_loitering_score=float(self.running_loitering_min_loitering_score),
            running_loitering_min_stationary_ratio=float(self.running_loitering_min_stationary_ratio),
            running_loitering_max_movement_extent=float(self.running_loitering_max_movement_extent),
            running_loitering_max_p90_speed=float(self.running_loitering_max_p90_speed),
            loitering_release_frames=int(self.loitering_release_frames),
            loitering_model_max_avg_speed=float(self.loitering_model_max_avg_speed),
            loitering_model_min_movement_extent=float(self.loitering_model_min_movement_extent),
            loitering_model_min_centroid_radius=float(self.loitering_model_min_centroid_radius),
            loitering_model_max_movement_extent=float(self.loitering_model_max_movement_extent),
            loitering_model_max_centroid_radius=float(self.loitering_model_max_centroid_radius),
            loitering_model_min_stationary_ratio=float(self.loitering_model_min_stationary_ratio),
            loitering_model_min_revisit_ratio=float(self.loitering_model_min_revisit_ratio),
            loitering_model_max_unique_cell_ratio=float(self.loitering_model_max_unique_cell_ratio),
            loitering_model_min_max_cell_occupancy_ratio=float(self.loitering_model_min_max_cell_occupancy_ratio),
            loitering_model_max_straightness=float(self.loitering_model_max_straightness),
            loitering_rule_min_stationary_ratio=float(self.loitering_rule_min_stationary_ratio),
            loitering_rule_min_revisit_ratio=float(self.loitering_rule_min_revisit_ratio),
            loitering_rule_min_movement_extent=float(self.loitering_rule_min_movement_extent),
            loitering_rule_min_centroid_radius=float(self.loitering_rule_min_centroid_radius),
            loitering_rule_max_straightness=float(self.loitering_rule_max_straightness),
            loitering_rule_max_displacement_ratio=float(self.loitering_rule_max_displacement_ratio),
            loitering_max_neighbor_count=int(self.loitering_max_neighbor_count),
            loitering_neighbor_radius=float(self.loitering_neighbor_radius),
            running_model_min_avg_speed=float(self.running_model_min_avg_speed),
            running_model_min_p90_speed=float(self.running_model_min_p90_speed),
            running_model_min_movement_extent=float(self.running_model_min_movement_extent),
            enable_loitering=bool(self.enable_loitering),
            enable_running=bool(self.enable_running),
            enable_intrusion=bool(self.enable_intrusion),
            enable_cross_line=bool(self.enable_cross_line),
            roi=tuple(self.roi) if self.roi else None,
            cross_line=tuple(tuple(point) for point in self.cross_line) if self.cross_line else None,
            loiter_frames=int(self.loiter_frames),
            loiter_radius=float(self.loiter_radius),
            loiter_speed=float(self.loiter_speed),
            running_speed=float(self.running_speed),
            running_frames=int(self.running_frames),
            max_videos=int(self.max_videos) if self.max_videos is not None else None,
            sequence_ids=tuple(str(item).zfill(2) for item in self.sequence_ids),
            save_demo_frames=bool(self.save_demo_frames),
            demo_dir=resolve_path(self.demo_dir),
        )


@dataclass(slots=True)
class UBnormalValidationConfig:
    manifest_path: Path = field(default_factory=lambda: resolve_path("data/processed/ubnormal_official/manifests/test.jsonl"))
    config_path: Path = field(default_factory=lambda: resolve_path("config.yaml"))
    output_path: Path = field(default_factory=lambda: resolve_path("output/ubnormal/validation_report.json"))
    max_videos: int | None = None
    sequence_ids: tuple[str, ...] = ()
    save_demo_frames: bool = False
    demo_dir: Path = field(default_factory=lambda: resolve_path("output/ubnormal/demo"))

    def resolved(self) -> "UBnormalValidationConfig":
        return UBnormalValidationConfig(
            manifest_path=resolve_path(self.manifest_path),
            config_path=resolve_path(self.config_path),
            output_path=resolve_path(self.output_path),
            max_videos=int(self.max_videos) if self.max_videos is not None else None,
            sequence_ids=tuple(str(item) for item in self.sequence_ids),
            save_demo_frames=bool(self.save_demo_frames),
            demo_dir=resolve_path(self.demo_dir),
        )


@dataclass(slots=True)
class AvenuePseudoLabelConfig:
    avenue_root: Path = field(default_factory=lambda: resolve_path("data/CUHK_Avenue/Avenue Dataset"))
    ground_truth_root: Path = field(default_factory=lambda: resolve_path("data/CUHK_Avenue/ground_truth_demo/testing_label_mask"))
    model: Path = field(default_factory=lambda: resolve_path("models/yolov8n.pt"))
    output_dir: Path = field(default_factory=lambda: resolve_path("output/avenue_pseudo_labels"))
    device: str | int | None = None
    conf_threshold: float = 0.1
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    max_time_lost: int = 30
    min_box_area: float = 10.0
    loiter_frames: int = 96
    loiter_radius: float = 40.13
    loiter_speed: float = 1.45
    running_speed: float = 15.57
    running_frames: int = 3
    min_track_frames: int = 24
    min_mask_overlap_frames: int = 3
    min_mask_overlap_ratio: float = 0.1
    running_min_high_speed_ratio: float = 0.3
    loiter_radius_multiplier: float = 1.25
    loiter_speed_multiplier: float = 1.2
    running_extent_multiplier: float = 1.5
    max_videos: int | None = None
    sequence_ids: tuple[str, ...] = ()

    def resolved(self) -> "AvenuePseudoLabelConfig":
        return AvenuePseudoLabelConfig(
            avenue_root=resolve_path(self.avenue_root),
            ground_truth_root=resolve_path(self.ground_truth_root),
            model=resolve_path(self.model),
            output_dir=resolve_path(self.output_dir),
            device=self.device,
            conf_threshold=float(self.conf_threshold),
            track_high_thresh=float(self.track_high_thresh),
            track_low_thresh=float(self.track_low_thresh),
            new_track_thresh=float(self.new_track_thresh),
            match_thresh=float(self.match_thresh),
            max_time_lost=int(self.max_time_lost),
            min_box_area=float(self.min_box_area),
            loiter_frames=int(self.loiter_frames),
            loiter_radius=float(self.loiter_radius),
            loiter_speed=float(self.loiter_speed),
            running_speed=float(self.running_speed),
            running_frames=int(self.running_frames),
            min_track_frames=int(self.min_track_frames),
            min_mask_overlap_frames=int(self.min_mask_overlap_frames),
            min_mask_overlap_ratio=float(self.min_mask_overlap_ratio),
            running_min_high_speed_ratio=float(self.running_min_high_speed_ratio),
            loiter_radius_multiplier=float(self.loiter_radius_multiplier),
            loiter_speed_multiplier=float(self.loiter_speed_multiplier),
            running_extent_multiplier=float(self.running_extent_multiplier),
            max_videos=int(self.max_videos) if self.max_videos is not None else None,
            sequence_ids=tuple(str(item).zfill(2) for item in self.sequence_ids),
        )


@dataclass(slots=True)
class AvenueBehaviorWindowConfig:
    input_path: Path = field(default_factory=lambda: resolve_path("output/avenue_pseudo_labels/tracks.jsonl"))
    output_dir: Path = field(default_factory=lambda: resolve_path("output/behavior_reconstructed/avenue_mask_windows_v1"))
    loiter_window: int = 72
    running_window: int = 24
    normal_window: int = 48
    loiter_stride: int = 12
    running_stride: int = 6
    normal_stride: int = 24
    min_track_frames: int = 24
    min_loiter_support_frames: int = 24
    min_loiter_support_ratio: float = 0.45
    min_loiter_stationary_ratio: float = 0.15
    min_loiter_revisit_ratio: float = 0.02
    max_loiter_straightness: float = 0.65
    min_running_support_frames: int = 8
    min_running_support_ratio: float = 0.30
    min_running_high_speed_ratio: float = 0.20
    min_running_avg_speed: float = 8.0
    min_running_p90_speed: float = 15.57
    min_running_movement_extent: float = 120.0
    max_normal_support_frames: int = 1
    max_normal_support_ratio: float = 0.02
    max_normal_windows_per_track: int = 2
    normal_to_positive_ratio: float = 1.5
    seed: int = 42

    def resolved(self) -> "AvenueBehaviorWindowConfig":
        return AvenueBehaviorWindowConfig(
            input_path=resolve_path(self.input_path),
            output_dir=resolve_path(self.output_dir),
            loiter_window=int(self.loiter_window),
            running_window=int(self.running_window),
            normal_window=int(self.normal_window),
            loiter_stride=int(self.loiter_stride),
            running_stride=int(self.running_stride),
            normal_stride=int(self.normal_stride),
            min_track_frames=int(self.min_track_frames),
            min_loiter_support_frames=int(self.min_loiter_support_frames),
            min_loiter_support_ratio=float(self.min_loiter_support_ratio),
            min_loiter_stationary_ratio=float(self.min_loiter_stationary_ratio),
            min_loiter_revisit_ratio=float(self.min_loiter_revisit_ratio),
            max_loiter_straightness=float(self.max_loiter_straightness),
            min_running_support_frames=int(self.min_running_support_frames),
            min_running_support_ratio=float(self.min_running_support_ratio),
            min_running_high_speed_ratio=float(self.min_running_high_speed_ratio),
            min_running_avg_speed=float(self.min_running_avg_speed),
            min_running_p90_speed=float(self.min_running_p90_speed),
            min_running_movement_extent=float(self.min_running_movement_extent),
            max_normal_support_frames=int(self.max_normal_support_frames),
            max_normal_support_ratio=float(self.max_normal_support_ratio),
            max_normal_windows_per_track=int(self.max_normal_windows_per_track),
            normal_to_positive_ratio=float(self.normal_to_positive_ratio),
            seed=int(self.seed),
        )


@dataclass(slots=True)
class BehaviorClassifierTrainConfig:
    dataset_path: Path = field(default_factory=lambda: resolve_path("output/avenue_pseudo_labels/tracks.jsonl"))
    output_dir: Path = field(default_factory=lambda: resolve_path("output/behavior_training"))
    run_name: str = "avenue_behavior_mlp"
    device: str | int | None = None
    model_type: str = "mlp"
    sequence_length: int = 72
    labels: tuple[str, ...] = ("normal", "running", "loitering")
    val_ratio: float = 0.2
    val_split_mode: str = "auto"
    seed: int = 42
    epochs: int = 150
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss_type: str = "ce"
    focal_gamma: float = 2.0
    class_balance_beta: float = 0.999
    hidden_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.1
    patience: int = 25

    def resolved(self) -> "BehaviorClassifierTrainConfig":
        return BehaviorClassifierTrainConfig(
            dataset_path=resolve_path(self.dataset_path),
            output_dir=resolve_path(self.output_dir),
            run_name=self.run_name,
            device=self.device,
            model_type=str(self.model_type),
            sequence_length=int(self.sequence_length),
            labels=tuple(self.labels),
            val_ratio=float(self.val_ratio),
            val_split_mode=str(self.val_split_mode),
            seed=int(self.seed),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            loss_type=str(self.loss_type),
            focal_gamma=float(self.focal_gamma),
            class_balance_beta=float(self.class_balance_beta),
            hidden_dims=tuple(int(value) for value in self.hidden_dims),
            dropout=float(self.dropout),
            patience=int(self.patience),
        )


@dataclass(slots=True)
class BehaviorModelEvalConfig:
    checkpoint_path: Path = field(default_factory=lambda: resolve_path("output/behavior_training/best.pt"))
    dataset_path: Path = field(default_factory=lambda: resolve_path("output/avenue_pseudo_labels/tracks.jsonl"))
    output_path: Path = field(default_factory=lambda: resolve_path("output/behavior_eval/report.json"))
    device: str | int | None = None
    loitering_min_score: float | None = 0.595
    running_min_score: float | None = None
    running_loitering_arb_enabled: bool = True
    running_loitering_min_loitering_score: float = 0.695
    running_loitering_min_stationary_ratio: float = 0.88
    running_loitering_max_movement_extent: float = 55.0
    running_loitering_max_p90_speed: float = 3.0
    loitering_borderline_gate_enabled: bool = True
    loitering_borderline_gate_max_score: float = 0.76
    loitering_borderline_gate_min_stationary_ratio: float = 0.70
    loitering_borderline_gate_max_movement_extent: float = 70.0
    loitering_borderline_gate_max_p90_speed: float = 4.0
    loitering_borderline_gate_min_revisit_ratio: float = 0.88
    running_borderline_gate_enabled: bool = True
    running_borderline_gate_max_score: float = 0.75
    running_borderline_gate_min_stationary_ratio: float = 0.80
    running_borderline_gate_max_movement_extent: float = 20.0
    running_borderline_gate_max_p90_speed: float = 2.0
    quality_adaptive_loitering_enabled: bool = True
    quality_adaptive_loitering_long_track_frames: float = 90.0
    quality_adaptive_loitering_long_track_max_score: float = 0.82
    quality_adaptive_loitering_long_track_min_revisit_ratio: float = 0.88
    source_aware_running_gate_enabled: bool = False
    source_aware_rswacv_running_max_score: float = 0.68
    source_aware_rswacv_running_max_movement_extent: float = 10.0
    source_aware_rswacv_running_max_p90_speed: float = 1.5

    def resolved(self) -> "BehaviorModelEvalConfig":
        return BehaviorModelEvalConfig(
            checkpoint_path=resolve_path(self.checkpoint_path),
            dataset_path=resolve_path(self.dataset_path),
            output_path=resolve_path(self.output_path),
            device=self.device,
            loitering_min_score=(
                float(self.loitering_min_score)
                if self.loitering_min_score is not None
                else None
            ),
            running_min_score=(
                float(self.running_min_score)
                if self.running_min_score is not None
                else None
            ),
            running_loitering_arb_enabled=bool(self.running_loitering_arb_enabled),
            running_loitering_min_loitering_score=float(self.running_loitering_min_loitering_score),
            running_loitering_min_stationary_ratio=float(self.running_loitering_min_stationary_ratio),
            running_loitering_max_movement_extent=float(self.running_loitering_max_movement_extent),
            running_loitering_max_p90_speed=float(self.running_loitering_max_p90_speed),
            loitering_borderline_gate_enabled=bool(self.loitering_borderline_gate_enabled),
            loitering_borderline_gate_max_score=float(self.loitering_borderline_gate_max_score),
            loitering_borderline_gate_min_stationary_ratio=float(self.loitering_borderline_gate_min_stationary_ratio),
            loitering_borderline_gate_max_movement_extent=float(self.loitering_borderline_gate_max_movement_extent),
            loitering_borderline_gate_max_p90_speed=float(self.loitering_borderline_gate_max_p90_speed),
            loitering_borderline_gate_min_revisit_ratio=float(self.loitering_borderline_gate_min_revisit_ratio),
            running_borderline_gate_enabled=bool(self.running_borderline_gate_enabled),
            running_borderline_gate_max_score=float(self.running_borderline_gate_max_score),
            running_borderline_gate_min_stationary_ratio=float(self.running_borderline_gate_min_stationary_ratio),
            running_borderline_gate_max_movement_extent=float(self.running_borderline_gate_max_movement_extent),
            running_borderline_gate_max_p90_speed=float(self.running_borderline_gate_max_p90_speed),
            quality_adaptive_loitering_enabled=bool(self.quality_adaptive_loitering_enabled),
            quality_adaptive_loitering_long_track_frames=float(self.quality_adaptive_loitering_long_track_frames),
            quality_adaptive_loitering_long_track_max_score=float(self.quality_adaptive_loitering_long_track_max_score),
            quality_adaptive_loitering_long_track_min_revisit_ratio=float(self.quality_adaptive_loitering_long_track_min_revisit_ratio),
            source_aware_running_gate_enabled=bool(self.source_aware_running_gate_enabled),
            source_aware_rswacv_running_max_score=float(self.source_aware_rswacv_running_max_score),
            source_aware_rswacv_running_max_movement_extent=float(self.source_aware_rswacv_running_max_movement_extent),
            source_aware_rswacv_running_max_p90_speed=float(self.source_aware_rswacv_running_max_p90_speed),
        )

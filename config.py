from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path
import yaml


@dataclass
class SourceConfig:
    use_camera: bool = False
    camera_id: int = 0
    video_path: str = ""
    mot17_seq: str = "bs/data/MOT17/train/MOT17-10-FRCNN/img1/%06d.jpg"


@dataclass
class ModelConfig:
    model_path: str = "bs/models/yolov8n.pt"
    device: str | int | None = None
    imgsz: int | None = None
    max_det: int | None = None
    nms_iou: float | None = None
    half: bool = False
    augment: bool = False
    runtime_profile: str = "balanced"
    predict_conf: float | None = None
    conf_threshold: float = 0.1
    classes: Tuple[int, ...] = (0,)


@dataclass
class TrackerConfig:
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    new_track_confirm_frames: int = 1
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
    appearance_hist_bins: Tuple[int, int, int] = (8, 4, 4)
    appearance_min_box_size: int = 16
    appearance_reid_model: str = "mobilenet_v3_small"
    appearance_reid_weights: str = ""
    appearance_reid_device: str = ""
    appearance_reid_input_size: Tuple[int, int] = (256, 128)
    appearance_reid_flip_aug: bool = False
    track_stability_weight: float = 0.0
    motion_gate_enabled: bool = False
    motion_gate_thresh: float = 9.4877
    crowd_boost_enabled: bool = False
    crowd_boost_det_count: int = 12
    crowd_match_thresh: float | None = None
    crowd_low_match_thresh: float | None = None
    crowd_new_track_confirm_frames: int | None = None
    crowd_appearance_weight: float | None = None
    crowd_track_stability_weight: float | None = None
    crowd_boost_min_small_ratio: float | None = None
    crowd_boost_max_median_area_ratio: float | None = None
    crowd_boost_small_area_ratio_thresh: float = 0.002


@dataclass
class BehaviorConfig:
    roi: Tuple[int, int, int, int] = (200, 120, 500, 420)
    roi_mode: str = "fixed"
    behavior_mode: str = "rules"
    enable_intrusion: bool = False
    enable_cross_line: bool = False
    enable_loitering: bool = False
    enable_running: bool = False
    intrusion_frames: int = 1
    cross_line: Tuple[Tuple[int, int], Tuple[int, int]] | None = None
    loiter_frames: int = 168
    loiter_radius: float = 40.13
    loiter_speed: float = 1.45
    running_speed: float = 15.57
    running_frames: int = 3
    behavior_model_path: str = ""
    behavior_secondary_model_path: str = ""
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


@dataclass
class DisplayConfig:
    window_name: str = "Intelligent Video Monitor"
    quit_key: str = "q"
    fps_smoothing: float = 0.95


@dataclass
class LogConfig:
    alarm_cooldown_seconds: float = 2.0


@dataclass
class OutputConfig:
    enable_event_log: bool = True
    enable_snapshot: bool = True
    enable_video_export: bool = False
    event_log_dir: str = "bs/output"
    snapshot_root: str = "bs/output/snaps"
    video_output_dir: str = "bs/output/videos"
    video_export_fps: float = 0.0
    video_codec: str = "mp4v"


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    log: LogConfig = field(default_factory=LogConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


BASE_DIR = Path(__file__).resolve().parent


def load_yaml_config(path="config.yaml"):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> str:
    if not path_str:
        return path_str

    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR.parent / path

    return str(path.resolve())


def get_config(path="config.yaml") -> AppConfig:
    data = load_yaml_config(path)

    source_data = data.get("source", {}).copy()
    model_data = data.get("model", {}).copy()
    tracker_data = data.get("tracker", {}).copy()
    behavior_data = data.get("behavior", {}).copy()
    output_data = data.get("output", {}).copy()

    roi_value = tuple(behavior_data.get("roi", [200, 120, 500, 420]))
    cross_line_value = behavior_data.get("cross_line")
    if cross_line_value:
        cross_line_value = tuple(
            tuple(int(v) for v in point)
            for point in cross_line_value
        )
    else:
        cross_line_value = None

    if source_data.get("video_path"):
        source_data["video_path"] = resolve_path(source_data["video_path"])

    if source_data.get("mot17_seq"):
        source_data["mot17_seq"] = resolve_path(source_data["mot17_seq"])

    if model_data.get("model_path"):
        model_data["model_path"] = resolve_path(model_data["model_path"])
    model_data["classes"] = tuple(model_data.get("classes", [0]))

    if "min_conf" in tracker_data and "track_high_thresh" not in tracker_data:
        tracker_data["track_high_thresh"] = tracker_data["min_conf"]
    if "max_lost" in tracker_data and "max_time_lost" not in tracker_data:
        tracker_data["max_time_lost"] = tracker_data["max_lost"]
    if "iou_thresh" in tracker_data and "match_thresh" not in tracker_data:
        tracker_data["match_thresh"] = max(0.5, 1.0 - float(tracker_data["iou_thresh"]))
    if "appearance_hist_bins" in tracker_data:
        tracker_data["appearance_hist_bins"] = tuple(int(v) for v in tracker_data["appearance_hist_bins"])

    if output_data.get("event_log_dir"):
        output_data["event_log_dir"] = resolve_path(output_data["event_log_dir"])

    if output_data.get("snapshot_root"):
        output_data["snapshot_root"] = resolve_path(output_data["snapshot_root"])
    if output_data.get("video_output_dir"):
        output_data["video_output_dir"] = resolve_path(output_data["video_output_dir"])

    if behavior_data.get("behavior_model_path"):
        behavior_data["behavior_model_path"] = resolve_path(behavior_data["behavior_model_path"])
    if behavior_data.get("behavior_secondary_model_path"):
        behavior_data["behavior_secondary_model_path"] = resolve_path(behavior_data["behavior_secondary_model_path"])

    return AppConfig(
        source=SourceConfig(**source_data),
        model=ModelConfig(**model_data),
        tracker=TrackerConfig(**tracker_data),
        behavior=BehaviorConfig(
            **{
                **behavior_data,
                "roi": roi_value,
                "cross_line": cross_line_value,
            }
        ),
        display=DisplayConfig(**data.get("display", {})),
        log=LogConfig(**data.get("log", {})),
        output=OutputConfig(**output_data),
    )

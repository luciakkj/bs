from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

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


def _compute_hard_negative_score(
    *,
    state: dict[str, object],
    features: dict[str, float],
    support_ratio: float,
    score_mode: str,
) -> float:
    classic_score = (
        float(state["loiter_alarm_frames"]) * 2.0
        + float(state["model_loiter_frames"]) * 0.5
        + float(state["max_support_run"]) * 0.25
    )
    if score_mode == "classic":
        return classic_score

    motion_normality = (
        min(float(features.get("movement_extent", 0.0)) / 120.0, 2.0) * 18.0
        + min(float(features.get("centroid_radius", 0.0)) / 40.0, 2.0) * 12.0
        + float(features.get("straightness", 0.0)) * 18.0
        + float(features.get("unique_cell_ratio", 0.0)) * 12.0
        - float(features.get("stationary_ratio", 0.0)) * 20.0
        - float(features.get("max_cell_occupancy_ratio", 0.0)) * 14.0
        - float(features.get("revisit_ratio", 0.0)) * 10.0
    )
    model_bias = (
        float(state["model_loiter_frames"]) * 1.0
        + float(state["loiter_alarm_frames"]) * 1.5
        - support_ratio * 40.0
        - float(state["max_support_run"]) * 0.2
    )
    return model_bias + motion_normality


def _build_negative_sample(
    track_id: int,
    sequence_id: str,
    state: dict[str, object],
    *,
    score_mode: str,
) -> dict[str, object] | None:
    features = features_from_trajectory_payload(
        {
            "centers": state["centers"],
            "speeds": state["speeds"],
        },
        high_speed_threshold=15.57,
        low_speed_threshold=2.2,
    )
    if features is None:
        return None

    support_flags = state["support_flags"]
    support_ratio = _safe_divide(sum(1 for flag in support_flags if flag), len(support_flags))
    hard_negative_score = _compute_hard_negative_score(
        state=state,
        features=features,
        support_ratio=support_ratio,
        score_mode=score_mode,
    )
    mean_confidence = _safe_divide(sum(state["confs"]), len(state["confs"]))
    return {
        "sample_id": f"trainhardneg_{sequence_id}_track_{track_id}",
        "source_track_id": f"trainhardneg_{sequence_id}_track_{track_id}",
        "sequence_id": str(sequence_id),
        "track_id": int(track_id),
        "start_frame": int(state["frames"][0]),
        "end_frame": int(state["frames"][-1]),
        "frame_count": len(state["frames"]),
        "primary_label": "normal",
        "pseudo_labels": ["normal"],
        "label_reasons": [
            (
                "train_normal_false_positive "
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
            "support_ratio": round(support_ratio, 4),
            "max_support_run": int(state["max_support_run"]),
            "hard_negative_score": round(hard_negative_score, 4),
            "hard_negative_score_mode": score_mode,
        },
        "trajectory": {
            "frames": state["frames"],
            "boxes": state["boxes"],
            "centers": [[round(float(x), 2), round(float(y), 2)] for x, y in state["centers"]],
            "speeds": [round(float(value), 3) for value in state["speeds"]],
            "support_flags": support_flags,
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


def mine_train_hardneg(
    *,
    config_path: str = "config.yaml",
    input_dataset: str = "output/behavior_reconstructed/avenue_train_mask_windows_v2/tracks_reconstructed.jsonl",
    output_dir: str = "output/behavior_hard_negatives/trainfp_20260405",
    sequence_ids: tuple[str, ...] = ("10", "13", "15", "16"),
    min_track_frames: int = 48,
    min_model_loiter_frames: int = 24,
    min_alarm_frames: int = 2,
    max_support_ratio: float = 0.20,
    top_k: int = 12,
    score_mode: str = "classic",
    min_movement_extent: float = 0.0,
    min_straightness: float = 0.0,
    max_stationary_ratio: float = 1.0,
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
    training_dir = PROJECT_ROOT / "data" / "CUHK_Avenue" / "Avenue Dataset" / "training_videos"
    video_paths = [path for path in sorted(training_dir.glob("*.avi")) if path.stem in selected_ids]
    if not video_paths:
        raise FileNotFoundError("No matching Avenue training videos found.")

    all_candidates: list[dict[str, object]] = []
    track_summaries: list[dict[str, object]] = []

    for video_path in video_paths:
        sequence_id = video_path.stem
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
                if not ret:
                    break
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame=frame)
                alarms, track_infos = behavior.update(tracks)
                loiter_alarm_ids = {
                    int(track_id) for track_id, alarm_type in alarms if str(alarm_type) == "loitering"
                }
                for track in tracks:
                    x1, y1, x2, y2, track_id, conf = track
                    track_id = int(track_id)
                    info = track_infos.get(track_id, {})
                    state = states.setdefault(
                        track_id,
                        {
                            "frames": [],
                            "boxes": [],
                            "centers": [],
                            "speeds": [],
                            "confs": [],
                            "support_flags": [],
                            "loiter_alarm_frames": 0,
                            "model_loiter_frames": 0,
                            "max_support_run": 0,
                            "current_support_run": 0,
                        },
                    )
                    state["frames"].append(int(frame_idx))
                    state["boxes"].append([int(x1), int(y1), int(x2), int(y2)])
                    center = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
                    state["centers"].append(center)
                    state["speeds"].append(float(info.get("speed", 0.0)))
                    state["confs"].append(float(conf))
                    support_flag = bool(info.get("support_only_loitering", False))
                    state["support_flags"].append(support_flag)
                    if support_flag:
                        state["current_support_run"] += 1
                        state["max_support_run"] = max(state["max_support_run"], state["current_support_run"])
                    else:
                        state["current_support_run"] = 0
                    if track_id in loiter_alarm_ids or bool(info.get("loitering_active", False)):
                        state["loiter_alarm_frames"] += 1
                    if float(info.get("model_loitering_score", 0.0)) >= float(cfg.behavior.loitering_model_support_thresh):
                        state["model_loiter_frames"] += 1
                frame_idx += 1
        finally:
            cap.release()

        per_sequence_candidates = []
        for track_id, state in states.items():
            frame_count = len(state["frames"])
            support_ratio = _safe_divide(sum(1 for flag in state["support_flags"] if flag), frame_count)
            features = features_from_trajectory_payload(
                {
                    "centers": state["centers"],
                    "speeds": state["speeds"],
                },
                high_speed_threshold=15.57,
                low_speed_threshold=2.2,
            )
            if features is None:
                continue
            hard_negative_score = _compute_hard_negative_score(
                state=state,
                features=features,
                support_ratio=support_ratio,
                score_mode=score_mode,
            )
            summary = {
                "sequence_id": sequence_id,
                "track_id": int(track_id),
                "frame_count": frame_count,
                "loiter_alarm_frames": int(state["loiter_alarm_frames"]),
                "model_loiter_frames": int(state["model_loiter_frames"]),
                "support_only_frames": int(sum(1 for flag in state["support_flags"] if flag)),
                "support_ratio": round(support_ratio, 4),
                "movement_extent": round(float(features["movement_extent"]), 4),
                "centroid_radius": round(float(features["centroid_radius"]), 4),
                "straightness": round(float(features["straightness"]), 4),
                "stationary_ratio": round(float(features["stationary_ratio"]), 4),
                "revisit_ratio": round(float(features["revisit_ratio"]), 4),
                "unique_cell_ratio": round(float(features["unique_cell_ratio"]), 4),
                "max_cell_occupancy_ratio": round(float(features["max_cell_occupancy_ratio"]), 4),
                "hard_negative_score": round(hard_negative_score, 4),
                "hard_negative_score_mode": score_mode,
            }
            track_summaries.append(summary)
            if frame_count < int(min_track_frames):
                continue
            if state["model_loiter_frames"] < int(min_model_loiter_frames):
                continue
            if state["loiter_alarm_frames"] < int(min_alarm_frames):
                continue
            if support_ratio > float(max_support_ratio):
                continue
            if float(features["movement_extent"]) < float(min_movement_extent):
                continue
            if float(features["straightness"]) < float(min_straightness):
                continue
            if float(features["stationary_ratio"]) > float(max_stationary_ratio):
                continue
            sample = _build_negative_sample(
                track_id,
                sequence_id,
                state,
                score_mode=score_mode,
            )
            if sample is None:
                continue
            per_sequence_candidates.append(sample)

        per_sequence_candidates.sort(key=lambda row: float(row["features"]["hard_negative_score"]), reverse=True)
        all_candidates.extend(per_sequence_candidates[: max(1, int(top_k))])

    base_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_candidates.sort(key=lambda row: float(row["features"]["hard_negative_score"]), reverse=True)
    selected_candidates = all_candidates[: int(top_k)]
    merged_rows = _merge_unique_rows(base_rows, selected_candidates)

    hardneg_path = output_root / "hard_negative_tracks.jsonl"
    merged_path = output_root / "tracks_v2_plus_trainfp_hardneg.jsonl"
    summary_path = output_root / "summary.json"

    hardneg_path.write_text(
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

    track_summaries.sort(key=lambda item: float(item["hard_negative_score"]), reverse=True)
    summary = {
        "config": {
            "config_path": str((PROJECT_ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path)),
            "input_dataset": str(dataset_path),
            "sequence_ids": sorted(selected_ids),
            "min_track_frames": int(min_track_frames),
            "min_model_loiter_frames": int(min_model_loiter_frames),
            "min_alarm_frames": int(min_alarm_frames),
            "max_support_ratio": float(max_support_ratio),
            "top_k": int(top_k),
            "score_mode": str(score_mode),
            "min_movement_extent": float(min_movement_extent),
            "min_straightness": float(min_straightness),
            "max_stationary_ratio": float(max_stationary_ratio),
        },
        "summary": {
            "base_rows": len(base_rows),
            "hard_negative_rows": len(selected_candidates),
            "merged_rows": len(merged_rows),
            "merged_label_counts": label_counts,
        },
        "selected_tracks": [
            f"{row['sequence_id']}:{row['track_id']}" for row in selected_candidates
        ],
        "top_tracks": track_summaries[:20],
        "artifacts": {
            "hard_negatives": str(hardneg_path),
            "merged_dataset": str(merged_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine loitering false positives from Avenue training videos.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--input-dataset", default="output/behavior_reconstructed/avenue_train_mask_windows_v2/tracks_reconstructed.jsonl")
    parser.add_argument("--output-dir", default="output/behavior_hard_negatives/trainfp_20260405")
    parser.add_argument("--sequence-ids", nargs="*", default=["10", "13", "15", "16"])
    parser.add_argument("--min-track-frames", type=int, default=48)
    parser.add_argument("--min-model-loiter-frames", type=int, default=24)
    parser.add_argument("--min-alarm-frames", type=int, default=2)
    parser.add_argument("--max-support-ratio", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--score-mode", choices=["classic", "normal_like"], default="classic")
    parser.add_argument("--min-movement-extent", type=float, default=0.0)
    parser.add_argument("--min-straightness", type=float, default=0.0)
    parser.add_argument("--max-stationary-ratio", type=float, default=1.0)
    args = parser.parse_args()

    result = mine_train_hardneg(
        config_path=args.config_path,
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        sequence_ids=tuple(args.sequence_ids),
        min_track_frames=args.min_track_frames,
        min_model_loiter_frames=args.min_model_loiter_frames,
        min_alarm_frames=args.min_alarm_frames,
        max_support_ratio=args.max_support_ratio,
        top_k=args.top_k,
        score_mode=args.score_mode,
        min_movement_extent=args.min_movement_extent,
        min_straightness=args.min_straightness,
        max_stationary_ratio=args.max_stationary_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

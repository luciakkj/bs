from __future__ import annotations

import configparser
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda array_like, dtype=float: np.asarray(array_like, dtype=dtype)

import motmetrics as mm

from config import get_config
from detector.yolo_detector import YOLODetector
from tracker.byte_tracker import ByteTrackerLite
from training.config import MOTTrackingEvalConfig


def _resolve_runtime_params(cfg: MOTTrackingEvalConfig) -> dict[str, Any]:
    app_cfg = get_config(str(cfg.config_path))

    model_path = str(cfg.model_path) if cfg.model_path else app_cfg.model.model_path
    device = cfg.device if cfg.device is not None else app_cfg.model.device
    imgsz = cfg.imgsz if cfg.imgsz is not None else app_cfg.model.imgsz
    max_det = cfg.max_det if cfg.max_det is not None else app_cfg.model.max_det
    nms_iou = cfg.nms_iou if cfg.nms_iou is not None else app_cfg.model.nms_iou
    half = cfg.half if cfg.half is not None else app_cfg.model.half
    augment = cfg.augment if cfg.augment is not None else app_cfg.model.augment

    predict_conf = cfg.predict_conf
    if predict_conf is None:
        predict_conf = app_cfg.model.predict_conf
    if predict_conf is None:
        predict_conf = min(
            float(app_cfg.model.conf_threshold),
            float(app_cfg.tracker.track_low_thresh),
        )

    track_high_thresh = cfg.track_high_thresh if cfg.track_high_thresh is not None else app_cfg.tracker.track_high_thresh
    track_low_thresh = cfg.track_low_thresh if cfg.track_low_thresh is not None else app_cfg.tracker.track_low_thresh
    new_track_thresh = cfg.new_track_thresh if cfg.new_track_thresh is not None else app_cfg.tracker.new_track_thresh
    new_track_confirm_frames = (
        cfg.new_track_confirm_frames
        if cfg.new_track_confirm_frames is not None
        else app_cfg.tracker.new_track_confirm_frames
    )
    match_thresh = cfg.match_thresh if cfg.match_thresh is not None else app_cfg.tracker.match_thresh
    low_match_thresh = cfg.low_match_thresh if cfg.low_match_thresh is not None else app_cfg.tracker.low_match_thresh
    unconfirmed_match_thresh = (
        cfg.unconfirmed_match_thresh
        if cfg.unconfirmed_match_thresh is not None
        else app_cfg.tracker.unconfirmed_match_thresh
    )
    score_fusion_weight = (
        cfg.score_fusion_weight
        if cfg.score_fusion_weight is not None
        else app_cfg.tracker.score_fusion_weight
    )
    max_time_lost = cfg.max_time_lost if cfg.max_time_lost is not None else app_cfg.tracker.max_time_lost
    min_box_area = cfg.min_box_area if cfg.min_box_area is not None else app_cfg.tracker.min_box_area

    runtime_profile = str(getattr(app_cfg.model, "runtime_profile", "balanced") or "balanced").lower()
    if cfg.apply_runtime_profile and runtime_profile == "crowd_recall":
        predict_conf = min(float(predict_conf), 0.18)
        track_high_thresh = min(float(track_high_thresh), 0.42)
        new_track_thresh = min(float(new_track_thresh), 0.55)
    elif cfg.apply_runtime_profile and runtime_profile == "tracking_boost":
        augment = False
        predict_conf = min(float(predict_conf), 0.12)
        track_high_thresh = min(float(track_high_thresh), 0.38)
        new_track_thresh = min(float(new_track_thresh), 0.48)
        match_thresh = min(float(match_thresh), 0.78)

    return {
        "config_path": str(cfg.config_path),
        "runtime_profile": runtime_profile if cfg.apply_runtime_profile else "disabled",
        "model_path": str(model_path),
        "device": device,
        "imgsz": imgsz,
        "max_det": max_det,
        "nms_iou": float(nms_iou) if nms_iou is not None else None,
        "half": bool(half),
        "augment": bool(augment),
        "predict_conf": float(predict_conf),
        "track_high_thresh": float(track_high_thresh),
        "track_low_thresh": float(track_low_thresh),
        "new_track_thresh": float(new_track_thresh),
        "new_track_confirm_frames": int(new_track_confirm_frames),
        "match_thresh": float(match_thresh),
        "low_match_thresh": float(low_match_thresh),
        "unconfirmed_match_thresh": float(unconfirmed_match_thresh),
        "score_fusion_weight": float(score_fusion_weight),
        "max_time_lost": int(max_time_lost),
        "min_box_area": float(min_box_area),
        "appearance_enabled": bool(
            cfg.appearance_enabled if cfg.appearance_enabled is not None else app_cfg.tracker.appearance_enabled
        ),
        "appearance_weight": float(
            cfg.appearance_weight if cfg.appearance_weight is not None else app_cfg.tracker.appearance_weight
        ),
        "appearance_ambiguity_margin": float(
            cfg.appearance_ambiguity_margin
            if cfg.appearance_ambiguity_margin is not None
            else app_cfg.tracker.appearance_ambiguity_margin
        ),
        "appearance_all_valid": bool(
            cfg.appearance_all_valid
            if cfg.appearance_all_valid is not None
            else app_cfg.tracker.appearance_all_valid
        ),
        "appearance_feature_mode": str(
            cfg.appearance_feature_mode
            if cfg.appearance_feature_mode is not None
            else app_cfg.tracker.appearance_feature_mode
        ),
        "appearance_hist_bins": tuple(
            int(value)
            for value in (
                cfg.appearance_hist_bins
                if cfg.appearance_hist_bins is not None
                else app_cfg.tracker.appearance_hist_bins
            )
        ),
        "appearance_min_box_size": int(
            cfg.appearance_min_box_size
            if cfg.appearance_min_box_size is not None
            else app_cfg.tracker.appearance_min_box_size
        ),
        "appearance_reid_model": str(
            cfg.appearance_reid_model
            if cfg.appearance_reid_model is not None
            else app_cfg.tracker.appearance_reid_model
        ),
        "appearance_reid_weights": str(
            cfg.appearance_reid_weights
            if cfg.appearance_reid_weights is not None
            else app_cfg.tracker.appearance_reid_weights
        ),
        "appearance_reid_device": str(
            cfg.appearance_reid_device
            if cfg.appearance_reid_device is not None
            else app_cfg.tracker.appearance_reid_device
        ),
        "appearance_reid_input_size": tuple(
            int(value)
            for value in (
                cfg.appearance_reid_input_size
                if cfg.appearance_reid_input_size is not None
                else app_cfg.tracker.appearance_reid_input_size
            )
        ),
        "appearance_reid_flip_aug": bool(
            cfg.appearance_reid_flip_aug
            if cfg.appearance_reid_flip_aug is not None
            else app_cfg.tracker.appearance_reid_flip_aug
        ),
        "track_stability_weight": float(
            cfg.track_stability_weight
            if cfg.track_stability_weight is not None
            else app_cfg.tracker.track_stability_weight
        ),
        "motion_gate_enabled": bool(
            cfg.motion_gate_enabled
            if cfg.motion_gate_enabled is not None
            else app_cfg.tracker.motion_gate_enabled
        ),
        "motion_gate_thresh": float(
            cfg.motion_gate_thresh
            if cfg.motion_gate_thresh is not None
            else app_cfg.tracker.motion_gate_thresh
        ),
        "crowd_boost_enabled": bool(
            cfg.crowd_boost_enabled if cfg.crowd_boost_enabled is not None else app_cfg.tracker.crowd_boost_enabled
        ),
        "crowd_boost_det_count": int(
            cfg.crowd_boost_det_count if cfg.crowd_boost_det_count is not None else app_cfg.tracker.crowd_boost_det_count
        ),
        "crowd_match_thresh": (
            float(cfg.crowd_match_thresh)
            if cfg.crowd_match_thresh is not None
            else (
                float(app_cfg.tracker.crowd_match_thresh)
                if app_cfg.tracker.crowd_match_thresh is not None
                else None
            )
        ),
        "crowd_low_match_thresh": (
            float(cfg.crowd_low_match_thresh)
            if cfg.crowd_low_match_thresh is not None
            else (
                float(app_cfg.tracker.crowd_low_match_thresh)
                if app_cfg.tracker.crowd_low_match_thresh is not None
                else None
            )
        ),
        "crowd_new_track_confirm_frames": (
            int(cfg.crowd_new_track_confirm_frames)
            if cfg.crowd_new_track_confirm_frames is not None
            else (
                int(app_cfg.tracker.crowd_new_track_confirm_frames)
                if app_cfg.tracker.crowd_new_track_confirm_frames is not None
                else None
            )
        ),
        "crowd_appearance_weight": (
            float(cfg.crowd_appearance_weight)
            if cfg.crowd_appearance_weight is not None
            else (
                float(app_cfg.tracker.crowd_appearance_weight)
                if app_cfg.tracker.crowd_appearance_weight is not None
                else None
            )
        ),
        "crowd_track_stability_weight": (
            float(cfg.crowd_track_stability_weight)
            if cfg.crowd_track_stability_weight is not None
            else (
                float(app_cfg.tracker.crowd_track_stability_weight)
                if app_cfg.tracker.crowd_track_stability_weight is not None
                else None
            )
        ),
        "crowd_boost_min_small_ratio": (
            float(cfg.crowd_boost_min_small_ratio)
            if cfg.crowd_boost_min_small_ratio is not None
            else (
                float(app_cfg.tracker.crowd_boost_min_small_ratio)
                if app_cfg.tracker.crowd_boost_min_small_ratio is not None
                else None
            )
        ),
        "crowd_boost_max_median_area_ratio": (
            float(cfg.crowd_boost_max_median_area_ratio)
            if cfg.crowd_boost_max_median_area_ratio is not None
            else (
                float(app_cfg.tracker.crowd_boost_max_median_area_ratio)
                if app_cfg.tracker.crowd_boost_max_median_area_ratio is not None
                else None
            )
        ),
        "crowd_boost_small_area_ratio_thresh": (
            float(cfg.crowd_boost_small_area_ratio_thresh)
            if cfg.crowd_boost_small_area_ratio_thresh is not None
            else float(app_cfg.tracker.crowd_boost_small_area_ratio_thresh)
        ),
    }


def _load_sequence_runtime_overrides(cfg: MOTTrackingEvalConfig) -> dict[str, dict[str, Any]]:
    if cfg.sequence_runtime_overrides_path is None:
        return {}
    path = cfg.sequence_runtime_overrides_path
    if not path.exists():
        raise FileNotFoundError(f"Sequence runtime overrides file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Sequence runtime overrides must be a JSON object mapping sequence names to override dicts.")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            normalized[str(key)] = dict(value)
    return normalized


def _iter_sequence_dirs(cfg: MOTTrackingEvalConfig) -> list[Path]:
    split_dir = cfg.mot_root / cfg.split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"MOT split directory not found: {split_dir}")

    requested = {str(item) for item in cfg.sequence_names}
    requested_short = {item.zfill(2) for item in requested if item.isdigit()}

    sequence_dirs: list[Path] = []
    for seq_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        name = seq_dir.name
        if cfg.detector_filter and not name.endswith(f"-{cfg.detector_filter}"):
            continue
        if requested:
            seq_id = name.split("-")[1] if "-" in name else name
            if name not in requested and seq_id not in requested_short:
                continue
        sequence_dirs.append(seq_dir)

    if not sequence_dirs:
        raise FileNotFoundError(
            f"No MOT sequences matched split={cfg.split_name!r}, detector_filter={cfg.detector_filter!r}, "
            f"sequence_names={cfg.sequence_names!r}"
        )
    return sequence_dirs


def _read_seqinfo(seq_dir: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini", encoding="utf-8")
    section = parser["Sequence"]
    return {
        "name": section.get("name", seq_dir.name),
        "frame_rate": section.getint("frameRate", fallback=30),
        "seq_length": section.getint("seqLength", fallback=0),
        "im_width": section.getint("imWidth", fallback=0),
        "im_height": section.getint("imHeight", fallback=0),
        "im_dir": section.get("imDir", "img1"),
        "im_ext": section.get("imExt", ".jpg"),
    }


def _resolve_frame_path(image_dir: Path, frame_id: int, image_ext: str) -> Path:
    candidates = (
        image_dir / f"{frame_id:06d}{image_ext}",
        image_dir / f"{frame_id:08d}{image_ext}",
        image_dir / f"{frame_id}{image_ext}",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_gt_by_frame(seq_dir: Path, include_classes: tuple[int, ...], min_visibility: float) -> dict[int, list[dict[str, float]]]:
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    gt_by_frame: dict[int, list[dict[str, float]]] = defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue
            frame_id = int(float(row[0]))
            obj_id = int(float(row[1]))
            x, y, w, h = (float(value) for value in row[2:6])
            mark = int(float(row[6]))
            cls = int(float(row[7]))
            visibility = float(row[8])
            if mark != 1:
                continue
            if cls not in include_classes:
                continue
            if visibility < min_visibility:
                continue
            gt_by_frame[frame_id].append(
                {
                    "id": obj_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "visibility": visibility,
                }
            )
    return gt_by_frame


def _mot_box_from_track(track_row: list[float]) -> list[float]:
    x1, y1, x2, y2 = track_row[:4]
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _write_mot_results(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def evaluate_mot_tracking(config: MOTTrackingEvalConfig) -> dict[str, Any]:
    cfg = config.resolved()
    runtime = _resolve_runtime_params(cfg)
    sequence_runtime_overrides = _load_sequence_runtime_overrides(cfg)
    sequence_dirs = _iter_sequence_dirs(cfg)

    detector = None
    detector_signature = None

    metric_names = [
        "num_frames",
        "idf1",
        "idp",
        "idr",
        "recall",
        "precision",
        "num_objects",
        "num_unique_objects",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "mota",
        "motp",
    ]
    summary_helper = mm.metrics.create()

    accumulators: list[mm.MOTAccumulator] = []
    accumulator_names: list[str] = []
    sequence_reports: list[dict[str, Any]] = []

    eval_started = time.perf_counter()

    for seq_dir in sequence_dirs:
        seq_runtime = dict(runtime)
        seq_override = sequence_runtime_overrides.get(seq_dir.name, {})
        for key, value in seq_override.items():
            if key in seq_runtime:
                seq_runtime[key] = value

        current_detector_signature = (
            seq_runtime["model_path"],
            seq_runtime["predict_conf"],
            seq_runtime["device"],
            seq_runtime["imgsz"],
            seq_runtime["max_det"],
            seq_runtime["half"],
            seq_runtime["augment"],
        )
        if detector is None or current_detector_signature != detector_signature:
            detector = YOLODetector(
                seq_runtime["model_path"],
                conf=seq_runtime["predict_conf"],
                classes=[0],
                device=seq_runtime["device"],
                imgsz=seq_runtime["imgsz"],
                max_det=seq_runtime["max_det"],
                nms_iou=seq_runtime["nms_iou"],
                half=seq_runtime["half"],
                augment=seq_runtime["augment"],
            )
            detector_signature = current_detector_signature

        seqinfo = _read_seqinfo(seq_dir)
        gt_by_frame = _load_gt_by_frame(seq_dir, cfg.include_classes, cfg.min_visibility)
        image_dir = seq_dir / seqinfo["im_dir"]
        sequence_length = int(seqinfo["seq_length"])
        if cfg.max_frames is not None:
            sequence_length = min(sequence_length, int(cfg.max_frames))

        tracker = ByteTrackerLite(
            track_high_thresh=seq_runtime["track_high_thresh"],
            track_low_thresh=seq_runtime["track_low_thresh"],
            new_track_thresh=seq_runtime["new_track_thresh"],
            new_track_confirm_frames=seq_runtime["new_track_confirm_frames"],
            match_thresh=seq_runtime["match_thresh"],
            low_match_thresh=seq_runtime["low_match_thresh"],
            unconfirmed_match_thresh=seq_runtime["unconfirmed_match_thresh"],
            score_fusion_weight=seq_runtime["score_fusion_weight"],
            max_time_lost=seq_runtime["max_time_lost"],
            min_box_area=seq_runtime["min_box_area"],
            appearance_enabled=seq_runtime["appearance_enabled"],
            appearance_weight=seq_runtime["appearance_weight"],
            appearance_ambiguity_margin=seq_runtime["appearance_ambiguity_margin"],
            appearance_all_valid=seq_runtime["appearance_all_valid"],
            appearance_feature_mode=seq_runtime["appearance_feature_mode"],
            appearance_hist_bins=seq_runtime["appearance_hist_bins"],
            appearance_min_box_size=seq_runtime["appearance_min_box_size"],
            appearance_reid_model=seq_runtime["appearance_reid_model"],
            appearance_reid_weights=seq_runtime["appearance_reid_weights"],
            appearance_reid_device=seq_runtime["appearance_reid_device"],
            appearance_reid_input_size=seq_runtime["appearance_reid_input_size"],
            appearance_reid_flip_aug=seq_runtime["appearance_reid_flip_aug"],
            track_stability_weight=seq_runtime["track_stability_weight"],
            motion_gate_enabled=seq_runtime["motion_gate_enabled"],
            motion_gate_thresh=seq_runtime["motion_gate_thresh"],
            crowd_boost_enabled=seq_runtime["crowd_boost_enabled"],
            crowd_boost_det_count=seq_runtime["crowd_boost_det_count"],
            crowd_match_thresh=seq_runtime["crowd_match_thresh"],
            crowd_low_match_thresh=seq_runtime["crowd_low_match_thresh"],
            crowd_new_track_confirm_frames=seq_runtime["crowd_new_track_confirm_frames"],
            crowd_appearance_weight=seq_runtime["crowd_appearance_weight"],
            crowd_track_stability_weight=seq_runtime["crowd_track_stability_weight"],
            crowd_boost_min_small_ratio=seq_runtime["crowd_boost_min_small_ratio"],
            crowd_boost_max_median_area_ratio=seq_runtime["crowd_boost_max_median_area_ratio"],
            crowd_boost_small_area_ratio_thresh=seq_runtime["crowd_boost_small_area_ratio_thresh"],
        )
        accumulator = mm.MOTAccumulator(auto_id=False)
        mot_rows: list[list[float]] = []
        detections_total = 0
        tracks_total = 0
        seq_started = time.perf_counter()

        for frame_id in range(1, sequence_length + 1):
            frame_path = _resolve_frame_path(image_dir, frame_id, seqinfo["im_ext"])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise FileNotFoundError(f"Frame not found or unreadable: {frame_path}")

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame=frame)
            detections_total += len(detections)
            tracks_total += len(tracks)

            gt_entries = gt_by_frame.get(frame_id, [])
            gt_ids = [int(item["id"]) for item in gt_entries]
            gt_boxes = np.asarray(
                [[item["x"], item["y"], item["w"], item["h"]] for item in gt_entries],
                dtype=float,
            )

            track_ids = [int(item[4]) for item in tracks]
            track_boxes = np.asarray([_mot_box_from_track(item) for item in tracks], dtype=float)
            distances = mm.distances.iou_matrix(
                gt_boxes,
                track_boxes,
                max_iou=max(0.0, 1.0 - float(cfg.min_iou)),
            )
            accumulator.update(gt_ids, track_ids, distances, frameid=frame_id)

            for item in tracks:
                x1, y1, x2, y2, track_id, score = item
                mot_rows.append(
                    [
                        frame_id,
                        int(track_id),
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                        float(score),
                        -1,
                        -1,
                        -1,
                    ]
                )

        seq_seconds = time.perf_counter() - seq_started
        accumulators.append(accumulator)
        accumulator_names.append(seq_dir.name)

        if cfg.save_mot_dir:
            _write_mot_results(cfg.save_mot_dir / f"{seq_dir.name}.txt", mot_rows)

        sequence_reports.append(
            {
                "sequence": seq_dir.name,
                "source_dir": str(seq_dir),
                "frames_evaluated": sequence_length,
                "gt_boxes_total": int(
                    sum(len(gt_by_frame.get(frame_id, ())) for frame_id in range(1, sequence_length + 1))
                ),
                "detections_total": int(detections_total),
                "tracks_total": int(tracks_total),
                "avg_detections_per_frame": float(detections_total / max(sequence_length, 1)),
                "avg_tracks_per_frame": float(tracks_total / max(sequence_length, 1)),
                "runtime_seconds": float(seq_seconds),
                "fps_tracking_eval": float(sequence_length / seq_seconds) if seq_seconds > 0 else None,
                "runtime_overrides_applied": {
                    str(key): _jsonify(value) for key, value in seq_override.items() if key in seq_runtime
                },
            }
        )

    summary = summary_helper.compute_many(
        accumulators,
        names=accumulator_names,
        metrics=metric_names,
        generate_overall=True,
    )
    summary_by_name = {
        str(index): {str(column): _jsonify(value) for column, value in row.items()}
        for index, row in summary.iterrows()
    }

    result = {
        "config": {
            "mot_root": str(cfg.mot_root),
            "split_name": cfg.split_name,
            "detector_filter": cfg.detector_filter,
            "sequence_names": list(cfg.sequence_names),
            "sequence_runtime_overrides_path": (
                str(cfg.sequence_runtime_overrides_path) if cfg.sequence_runtime_overrides_path else None
            ),
            "include_classes": list(cfg.include_classes),
            "min_visibility": float(cfg.min_visibility),
            "min_iou": float(cfg.min_iou),
            "max_frames": cfg.max_frames,
            "save_mot_dir": str(cfg.save_mot_dir) if cfg.save_mot_dir else None,
        },
        "runtime": _jsonify(runtime),
        "sequences": sequence_reports,
        "metrics": summary_by_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds_total": float(time.perf_counter() - eval_started),
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(
        json.dumps(_jsonify(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result

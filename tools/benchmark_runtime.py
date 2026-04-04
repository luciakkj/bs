from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior.abnormal_detector import AbnormalDetector
from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from tracker.byte_tracker import ByteTrackerLite
from utils.visualization import draw


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def _resolve_image_sequence_pattern(path: Path) -> Path:
    if path.is_file():
        return path

    candidate_dirs = [path]
    img1_dir = path / "img1"
    if img1_dir.is_dir():
        candidate_dirs.insert(0, img1_dir)

    for directory in candidate_dirs:
        if next(directory.glob("*.jpg"), None) is not None:
            return directory / "%06d.jpg"

    raise FileNotFoundError(f"No jpg image sequence found under: {path}")


def _resolve_capture_source(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Source path does not exist: {path}")
    if path.is_dir():
        return str(_resolve_image_sequence_pattern(path).resolve())
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return str(path.resolve())
    raise ValueError(f"Unsupported source path: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end runtime of detection + tracking + behavior.")
    parser.add_argument("--source", default="data/MOT17/train/MOT17-02-FRCNN", help="Video path or image sequence directory.")
    parser.add_argument("--weights", required=True, help="Detector weights path or model name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to benchmark.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames excluded from timing.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold.")
    parser.add_argument("--behavior-model-path", default="", help="Optional behavior classifier checkpoint.")
    parser.add_argument("--draw-vis", action="store_true", help="Include overlay drawing in timing.")
    parser.add_argument("--enable-loitering", action="store_true")
    parser.add_argument("--enable-running", action="store_true")
    parser.add_argument("--enable-intrusion", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    source = _resolve_capture_source(args.source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    model = YOLO(args.weights)
    tracker = ByteTrackerLite()

    behavior_classifier = None
    if args.behavior_model_path:
        behavior_classifier = TrajectoryBehaviorClassifier(checkpoint_path=args.behavior_model_path)

    behavior = AbnormalDetector(
        roi=(200, 120, 500, 420),
        behavior_mode="hybrid" if behavior_classifier is not None else "rules",
        enable_intrusion=args.enable_intrusion,
        enable_loitering=args.enable_loitering,
        enable_running=args.enable_running,
        behavior_classifier=behavior_classifier,
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
    )

    processed_frames = 0
    timed_frames = 0
    total_time = 0.0
    timings = {
        "detect": 0.0,
        "track": 0.0,
        "behavior": 0.0,
        "draw": 0.0,
    }

    try:
        while processed_frames < args.frames:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1

            t0 = time.perf_counter()
            results = model.predict(
                frame,
                conf=args.conf,
                classes=[0],
                device=args.device,
                imgsz=args.imgsz,
                verbose=False,
            )[0]
            t1 = time.perf_counter()

            detections = []
            if results.boxes is not None:
                xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
                confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []
                for box, score in zip(xyxy, confs):
                    x1, y1, x2, y2 = map(int, box[:4])
                    detections.append([x1, y1, x2, y2, float(score)])

            tracks = tracker.update(detections)
            t2 = time.perf_counter()
            alarms, track_infos = behavior.update(tracks)
            t3 = time.perf_counter()

            if args.draw_vis:
                _ = draw(
                    frame=frame,
                    tracks=tracks,
                    alarms=alarms,
                    fps=0.0,
                    roi=(200, 120, 500, 420),
                    cross_line=None,
                    track_infos=track_infos,
                )
            t4 = time.perf_counter()

            if processed_frames <= args.warmup:
                continue

            timed_frames += 1
            total_time += t4 - t0
            timings["detect"] += t1 - t0
            timings["track"] += t2 - t1
            timings["behavior"] += t3 - t2
            timings["draw"] += t4 - t3
    finally:
        cap.release()

    if timed_frames <= 0:
        raise RuntimeError("Not enough frames were processed for benchmarking.")

    summary = {
        "weights": str(args.weights),
        "imgsz": int(args.imgsz),
        "source": source,
        "frames_total": processed_frames,
        "frames_timed": timed_frames,
        "fps_end_to_end": round(timed_frames / total_time, 3),
        "ms_per_frame_total": round(total_time * 1000.0 / timed_frames, 3),
        "ms_per_frame_detect": round(timings["detect"] * 1000.0 / timed_frames, 3),
        "ms_per_frame_track": round(timings["track"] * 1000.0 / timed_frames, 3),
        "ms_per_frame_behavior": round(timings["behavior"] * 1000.0 / timed_frames, 3),
        "ms_per_frame_draw": round(timings["draw"] * 1000.0 / timed_frames, 3),
        "draw_enabled": bool(args.draw_vis),
        "behavior_model_enabled": bool(args.behavior_model_path),
        "enable_intrusion": bool(args.enable_intrusion),
        "enable_loitering": bool(args.enable_loitering),
        "enable_running": bool(args.enable_running),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

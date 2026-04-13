from __future__ import annotations

import argparse
import json

from training.avenue_pseudo_labels import generate_avenue_pseudo_labels
from training.avenue_validation import validate_on_avenue
from training.ubnormal_validation import validate_on_ubnormal
from training.avenue_behavior_windows import build_avenue_behavior_windows
from training.behavior_calibration import calibrate_behavior_thresholds
from training.behavior_classifier import train_behavior_classifier
from training.behavior_dataset_expansion import expand_behavior_dataset
from training.behavior_hard_negative_mining import mine_behavior_hard_negatives
from training.behavior_model_eval import evaluate_behavior_model
from training.behavior_window_reconstruction import reconstruct_behavior_windows
from training.mot_tracking_eval import evaluate_mot_tracking
from training.config import (
    AvenueBehaviorWindowConfig,
    AvenuePseudoLabelConfig,
    AvenueValidationConfig,
    BehaviorModelEvalConfig,
    UBnormalValidationConfig,
    BehaviorClassifierTrainConfig,
    CalibrationConfig,
    MOTTrackingEvalConfig,
    PrepareConfig,
    TrainConfig,
    ValidateConfig,
)
from training.mot_dataset import MOT17ToYOLOConverter
from training.yolo_workflow import train_detector, validate_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Training utilities for the intelligent video surveillance project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert MOT17 annotations to YOLO training format.")
    prepare.add_argument("--mot-root", default="data/MOT17")
    prepare.add_argument("--output-dir", default="data/processed/mot17_person")
    prepare.add_argument("--split-name", default="train")
    prepare.add_argument("--detector", default="FRCNN")
    prepare.add_argument("--min-visibility", type=float, default=0.25)
    prepare.add_argument("--val-ratio", type=float, default=0.25)
    prepare.add_argument("--val-sequences", nargs="*", default=[])
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--dense-small-repeat-factor", type=int, default=1)
    prepare.add_argument("--dense-small-max-repeat-frames", type=int, default=None)
    prepare.add_argument("--dense-small-min-gt-count", type=int, default=12)
    prepare.add_argument("--dense-small-min-small-ratio", type=float, default=0.7)
    prepare.add_argument("--dense-small-max-median-area-ratio", type=float, default=0.002)
    prepare.add_argument("--small-box-area-ratio-thresh", type=float, default=0.002)
    prepare.add_argument("--dense-small-crop-enable", action=argparse.BooleanOptionalAction, default=False)
    prepare.add_argument("--dense-small-crop-max-frames", type=int, default=None)
    prepare.add_argument("--dense-small-crop-width-ratio", type=float, default=0.5)
    prepare.add_argument("--dense-small-crop-height-ratio", type=float, default=0.5)
    prepare.add_argument("--dense-small-crop-min-boxes", type=int, default=6)

    calibrate = subparsers.add_parser("calibrate", help="Estimate behavior thresholds from MOT17 trajectories.")
    calibrate.add_argument("--mot-root", default="data/MOT17")
    calibrate.add_argument("--output-path", default="output/calibration/behavior_thresholds.json")
    calibrate.add_argument("--split-name", default="train")
    calibrate.add_argument("--detector", default="FRCNN")
    calibrate.add_argument("--min-visibility", type=float, default=0.25)

    train = subparsers.add_parser("train", help="Train a YOLO person detector.")
    train.add_argument("--data", default="data/processed/mot17_person/mot17_person.yaml")
    train.add_argument("--model", default="models/yolov8n.pt")
    train.add_argument("--project", default="output/training")
    train.add_argument("--name", default="mot17_person")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=960)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--device", default=None)

    validate = subparsers.add_parser("validate", help="Validate YOLO weights on the prepared dataset.")
    validate.add_argument("--data", default="data/processed/mot17_person/mot17_person.yaml")
    validate.add_argument("--weights", default="models/yolov8n.pt")
    validate.add_argument("--imgsz", type=int, default=960)
    validate.add_argument("--batch", type=int, default=8)
    validate.add_argument("--device", default=None)
    validate.add_argument("--split", default="val")
    validate.add_argument("--workers", type=int, default=0)

    tracking_eval = subparsers.add_parser(
        "eval-mot-tracking",
        help="Evaluate detector + ByteTrackLite on MOT17 with MOT-style tracking metrics.",
    )
    tracking_eval.add_argument("--config-path", default="config.yaml")
    tracking_eval.add_argument("--mot-root", default="data/MOT17")
    tracking_eval.add_argument("--output-path", default="output/tracking/mot17_tracking_eval.json")
    tracking_eval.add_argument("--save-mot-dir", default="output/tracking/mot17_tracking_eval_tracks")
    tracking_eval.add_argument("--split-name", default="train")
    tracking_eval.add_argument("--detector", default="FRCNN")
    tracking_eval.add_argument("--sequence-names", nargs="*", default=[])
    tracking_eval.add_argument("--sequence-runtime-overrides-path", default=None)
    tracking_eval.add_argument("--include-classes", nargs="*", type=int, default=[1])
    tracking_eval.add_argument("--min-visibility", type=float, default=0.0)
    tracking_eval.add_argument("--min-iou", type=float, default=0.5)
    tracking_eval.add_argument("--max-frames", type=int, default=None)
    tracking_eval.add_argument(
        "--apply-runtime-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    tracking_eval.add_argument("--model-path", default=None)
    tracking_eval.add_argument("--device", default=None)
    tracking_eval.add_argument("--imgsz", type=int, default=None)
    tracking_eval.add_argument("--max-det", type=int, default=None)
    tracking_eval.add_argument("--nms-iou", type=float, default=None)
    tracking_eval.add_argument("--half", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--predict-conf", type=float, default=None)
    tracking_eval.add_argument("--track-high-thresh", type=float, default=None)
    tracking_eval.add_argument("--track-low-thresh", type=float, default=None)
    tracking_eval.add_argument("--new-track-thresh", type=float, default=None)
    tracking_eval.add_argument("--match-thresh", type=float, default=None)
    tracking_eval.add_argument("--low-match-thresh", type=float, default=None)
    tracking_eval.add_argument("--unconfirmed-match-thresh", type=float, default=None)
    tracking_eval.add_argument("--score-fusion-weight", type=float, default=None)
    tracking_eval.add_argument("--max-time-lost", type=int, default=None)
    tracking_eval.add_argument("--min-box-area", type=float, default=None)
    tracking_eval.add_argument("--appearance-enabled", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--appearance-weight", type=float, default=None)
    tracking_eval.add_argument("--appearance-ambiguity-margin", type=float, default=None)
    tracking_eval.add_argument("--appearance-feature-mode", default=None)
    tracking_eval.add_argument("--appearance-hist-bins", nargs=3, type=int, default=None)
    tracking_eval.add_argument("--appearance-min-box-size", type=int, default=None)
    tracking_eval.add_argument("--appearance-reid-model", default=None)
    tracking_eval.add_argument("--appearance-reid-weights", default=None)
    tracking_eval.add_argument("--appearance-reid-device", default=None)
    tracking_eval.add_argument("--appearance-reid-input-size", nargs=2, type=int, default=None)
    tracking_eval.add_argument("--appearance-reid-flip-aug", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--appearance-all-valid", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--motion-gate-enabled", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--motion-gate-thresh", type=float, default=None)
    tracking_eval.add_argument("--crowd-boost-enabled", action=argparse.BooleanOptionalAction, default=None)
    tracking_eval.add_argument("--crowd-boost-det-count", type=int, default=None)
    tracking_eval.add_argument("--crowd-match-thresh", type=float, default=None)
    tracking_eval.add_argument("--crowd-low-match-thresh", type=float, default=None)
    tracking_eval.add_argument("--crowd-appearance-weight", type=float, default=None)
    tracking_eval.add_argument("--crowd-boost-min-small-ratio", type=float, default=None)
    tracking_eval.add_argument("--crowd-boost-max-median-area-ratio", type=float, default=None)
    tracking_eval.add_argument("--crowd-boost-small-area-ratio-thresh", type=float, default=None)

    avenue = subparsers.add_parser("validate-avenue", help="Run anomaly validation on CUHK Avenue using detection, tracking, and rule-based behavior analysis.")
    avenue.add_argument("--avenue-root", default="data/CUHK_Avenue/Avenue Dataset")
    avenue.add_argument("--ground-truth-root", default="data/CUHK_Avenue/ground_truth_demo/testing_label_mask")
    avenue.add_argument("--model", default="models/yolov8n.pt")
    avenue.add_argument("--output-path", default="output/avenue/validation_report.json")
    avenue.add_argument("--device", default=None)
    avenue.add_argument("--imgsz", type=int, default=None)
    avenue.add_argument("--max-det", type=int, default=None)
    avenue.add_argument("--half", action="store_true")
    avenue.add_argument("--augment", action="store_true")
    avenue.add_argument("--conf-threshold", type=float, default=0.1)
    avenue.add_argument("--track-high-thresh", type=float, default=0.5)
    avenue.add_argument("--track-low-thresh", type=float, default=0.1)
    avenue.add_argument("--new-track-thresh", type=float, default=0.6)
    avenue.add_argument("--match-thresh", type=float, default=0.8)
    avenue.add_argument("--low-match-thresh", type=float, default=0.5)
    avenue.add_argument("--unconfirmed-match-thresh", type=float, default=0.7)
    avenue.add_argument("--score-fusion-weight", type=float, default=0.0)
    avenue.add_argument("--max-time-lost", type=int, default=30)
    avenue.add_argument("--min-box-area", type=float, default=10.0)
    avenue.add_argument("--appearance-enabled", action=argparse.BooleanOptionalAction, default=False)
    avenue.add_argument("--appearance-weight", type=float, default=0.25)
    avenue.add_argument("--appearance-ambiguity-margin", type=float, default=0.05)
    avenue.add_argument("--appearance-feature-mode", default="hsv")
    avenue.add_argument("--appearance-hist-bins", nargs=3, type=int, default=[8, 4, 4])
    avenue.add_argument("--appearance-min-box-size", type=int, default=16)
    avenue.add_argument("--appearance-reid-model", default="mobilenet_v3_small")
    avenue.add_argument("--appearance-reid-weights", default=None)
    avenue.add_argument("--appearance-reid-device", default=None)
    avenue.add_argument("--appearance-reid-input-size", nargs=2, type=int, default=[256, 128])
    avenue.add_argument("--behavior-mode", default="rules")
    avenue.add_argument("--behavior-model-path", default=None)
    avenue.add_argument("--behavior-secondary-model-path", default=None)
    avenue.add_argument("--behavior-ensemble-primary-weight", type=float, default=1.0)
    avenue.add_argument("--behavior-ensemble-mode", default="weighted")
    avenue.add_argument("--behavior-model-score-thresh", type=float, default=0.55)
    avenue.add_argument("--behavior-model-min-frames", type=int, default=24)
    avenue.add_argument("--behavior-model-max-tracks", type=int, default=32)
    avenue.add_argument("--behavior-model-resume-tracks", type=int, default=26)
    avenue.add_argument("--behavior-secondary-max-tracks", type=int, default=24)
    avenue.add_argument("--behavior-secondary-resume-tracks", type=int, default=18)
    avenue.add_argument(
        "--behavior-secondary-loitering-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    avenue.add_argument("--behavior-model-eval-interval", type=int, default=3)
    avenue.add_argument("--loitering-hybrid-mode", choices=["union", "model_only", "model_support", "rules_only"], default="union")
    avenue.add_argument("--loitering-model-support-thresh", type=float, default=0.0)
    avenue.add_argument("--loitering-model-score-thresh", type=float, default=0.97)
    avenue.add_argument("--running-model-score-thresh", type=float, default=0.85)
    avenue.add_argument("--loitering-model-min-frames", type=int, default=72)
    avenue.add_argument("--loitering-activate-frames", type=int, default=1)
    avenue.add_argument("--loitering-support-activate-frames", type=int, default=0)
    avenue.add_argument("--loitering-support-block-running", action="store_true")
    avenue.add_argument("--loitering-context-gate-support-only", action=argparse.BooleanOptionalAction, default=False)
    avenue.add_argument("--running-loitering-arb-enabled", action=argparse.BooleanOptionalAction, default=False)
    avenue.add_argument("--running-loitering-min-loitering-score", type=float, default=0.72)
    avenue.add_argument("--running-loitering-min-stationary-ratio", type=float, default=0.90)
    avenue.add_argument("--running-loitering-max-movement-extent", type=float, default=50.0)
    avenue.add_argument("--running-loitering-max-p90-speed", type=float, default=3.0)
    avenue.add_argument("--loitering-release-frames", type=int, default=1)
    avenue.add_argument("--loitering-model-max-avg-speed", type=float, default=2.2)
    avenue.add_argument("--loitering-model-min-movement-extent", type=float, default=0.0)
    avenue.add_argument("--loitering-model-min-centroid-radius", type=float, default=0.0)
    avenue.add_argument("--loitering-model-max-movement-extent", type=float, default=55.0)
    avenue.add_argument("--loitering-model-max-centroid-radius", type=float, default=28.0)
    avenue.add_argument("--loitering-model-min-stationary-ratio", type=float, default=0.0)
    avenue.add_argument("--loitering-model-min-revisit-ratio", type=float, default=0.0)
    avenue.add_argument("--loitering-model-max-unique-cell-ratio", type=float, default=1.0)
    avenue.add_argument("--loitering-model-min-max-cell-occupancy-ratio", type=float, default=0.0)
    avenue.add_argument("--loitering-model-max-straightness", type=float, default=1.0)
    avenue.add_argument("--loitering-rule-min-stationary-ratio", type=float, default=0.0)
    avenue.add_argument("--loitering-rule-min-revisit-ratio", type=float, default=0.0)
    avenue.add_argument("--loitering-rule-min-movement-extent", type=float, default=0.0)
    avenue.add_argument("--loitering-rule-min-centroid-radius", type=float, default=0.0)
    avenue.add_argument("--loitering-rule-max-straightness", type=float, default=1.0)
    avenue.add_argument("--loitering-rule-max-displacement-ratio", type=float, default=1.0)
    avenue.add_argument("--loitering-max-neighbor-count", type=int, default=-1)
    avenue.add_argument("--loitering-neighbor-radius", type=float, default=80.0)
    avenue.add_argument("--running-model-min-avg-speed", type=float, default=8.0)
    avenue.add_argument("--running-model-min-p90-speed", type=float, default=16.0)
    avenue.add_argument("--running-model-min-movement-extent", type=float, default=120.0)
    avenue.add_argument("--enable-loitering", action="store_true")
    avenue.add_argument("--enable-running", action="store_true")
    avenue.add_argument("--enable-intrusion", action="store_true")
    avenue.add_argument("--enable-cross-line", action="store_true")
    avenue.add_argument("--loiter-frames", type=int, default=168)
    avenue.add_argument("--loiter-radius", type=float, default=40.13)
    avenue.add_argument("--loiter-speed", type=float, default=1.45)
    avenue.add_argument("--running-speed", type=float, default=15.57)
    avenue.add_argument("--running-frames", type=int, default=3)
    avenue.add_argument("--max-videos", type=int, default=None)
    avenue.add_argument("--sequence-ids", nargs="*", default=[])
    avenue.add_argument("--save-demo-frames", action="store_true")
    avenue.add_argument("--demo-dir", default="output/avenue/demo")

    ubnormal = subparsers.add_parser(
        "validate-ubnormal",
        help="Run anomaly validation on UBnormal using the current project config.",
    )
    ubnormal.add_argument("--manifest-path", default="data/processed/ubnormal_official/manifests/test.jsonl")
    ubnormal.add_argument("--config-path", default="config.yaml")
    ubnormal.add_argument("--output-path", default="output/ubnormal/validation_report.json")
    ubnormal.add_argument("--max-videos", type=int, default=None)
    ubnormal.add_argument("--sequence-ids", nargs="*", default=[])
    ubnormal.add_argument("--save-demo-frames", action="store_true")
    ubnormal.add_argument("--demo-dir", default="output/ubnormal/demo")

    pseudo = subparsers.add_parser(
        "generate-avenue-pseudo-labels",
        help="Generate track-level pseudo labels for running/loitering from CUHK Avenue masks.",
    )
    pseudo.add_argument("--avenue-root", default="data/CUHK_Avenue/Avenue Dataset")
    pseudo.add_argument("--ground-truth-root", default="data/CUHK_Avenue/ground_truth_demo/testing_label_mask")
    pseudo.add_argument("--model", default="models/yolov8n.pt")
    pseudo.add_argument("--output-dir", default="output/avenue_pseudo_labels")
    pseudo.add_argument("--device", default=None)
    pseudo.add_argument("--conf-threshold", type=float, default=0.1)
    pseudo.add_argument("--track-high-thresh", type=float, default=0.5)
    pseudo.add_argument("--track-low-thresh", type=float, default=0.1)
    pseudo.add_argument("--new-track-thresh", type=float, default=0.6)
    pseudo.add_argument("--match-thresh", type=float, default=0.8)
    pseudo.add_argument("--max-time-lost", type=int, default=30)
    pseudo.add_argument("--min-box-area", type=float, default=10.0)
    pseudo.add_argument("--loiter-frames", type=int, default=96)
    pseudo.add_argument("--loiter-radius", type=float, default=40.13)
    pseudo.add_argument("--loiter-speed", type=float, default=1.45)
    pseudo.add_argument("--running-speed", type=float, default=15.57)
    pseudo.add_argument("--running-frames", type=int, default=3)
    pseudo.add_argument("--min-track-frames", type=int, default=24)
    pseudo.add_argument("--min-mask-overlap-frames", type=int, default=3)
    pseudo.add_argument("--min-mask-overlap-ratio", type=float, default=0.1)
    pseudo.add_argument("--running-min-high-speed-ratio", type=float, default=0.3)
    pseudo.add_argument("--loiter-radius-multiplier", type=float, default=1.25)
    pseudo.add_argument("--loiter-speed-multiplier", type=float, default=1.2)
    pseudo.add_argument("--running-extent-multiplier", type=float, default=1.5)
    pseudo.add_argument("--max-videos", type=int, default=None)
    pseudo.add_argument("--sequence-ids", nargs="*", default=[])

    behavior_train = subparsers.add_parser(
        "train-behavior",
        help="Train a lightweight trajectory behavior classifier from Avenue pseudo labels.",
    )
    behavior_train.add_argument("--dataset-path", default="output/avenue_pseudo_labels/tracks.jsonl")
    behavior_train.add_argument("--output-dir", default="output/behavior_training")
    behavior_train.add_argument("--run-name", default="avenue_behavior_mlp")
    behavior_train.add_argument("--device", default=None)
    behavior_train.add_argument("--model-type", default="mlp")
    behavior_train.add_argument("--sequence-length", type=int, default=72)
    behavior_train.add_argument("--labels", nargs="+", default=["normal", "running", "loitering"])
    behavior_train.add_argument("--val-ratio", type=float, default=0.2)
    behavior_train.add_argument("--val-split-mode", choices=["auto", "track", "sequence"], default="auto")
    behavior_train.add_argument("--seed", type=int, default=42)
    behavior_train.add_argument("--epochs", type=int, default=150)
    behavior_train.add_argument("--batch-size", type=int, default=32)
    behavior_train.add_argument("--lr", type=float, default=1e-3)
    behavior_train.add_argument("--weight-decay", type=float, default=1e-4)
    behavior_train.add_argument("--loss-type", default="ce")
    behavior_train.add_argument("--focal-gamma", type=float, default=2.0)
    behavior_train.add_argument("--class-balance-beta", type=float, default=0.999)
    behavior_train.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 16])
    behavior_train.add_argument("--dropout", type=float, default=0.1)
    behavior_train.add_argument("--patience", type=int, default=25)

    behavior_eval = subparsers.add_parser(
        "eval-behavior-model",
        help="Evaluate a behavior classifier checkpoint on a track JSONL dataset.",
    )
    behavior_eval.add_argument("--checkpoint-path", required=True)
    behavior_eval.add_argument("--dataset-path", required=True)
    behavior_eval.add_argument("--output-path", default="output/behavior_eval/report.json")
    behavior_eval.add_argument("--device", default="")
    behavior_eval.add_argument("--loitering-min-score", type=float, default=0.595)
    behavior_eval.add_argument("--running-min-score", type=float, default=None)
    behavior_eval.add_argument("--running-loitering-arb-enabled", action=argparse.BooleanOptionalAction, default=True)
    behavior_eval.add_argument("--running-loitering-min-loitering-score", type=float, default=0.695)
    behavior_eval.add_argument("--running-loitering-min-stationary-ratio", type=float, default=0.88)
    behavior_eval.add_argument("--running-loitering-max-movement-extent", type=float, default=55.0)
    behavior_eval.add_argument("--running-loitering-max-p90-speed", type=float, default=3.0)
    behavior_eval.add_argument("--loitering-borderline-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    behavior_eval.add_argument("--loitering-borderline-gate-max-score", type=float, default=0.76)
    behavior_eval.add_argument("--loitering-borderline-gate-min-stationary-ratio", type=float, default=0.70)
    behavior_eval.add_argument("--loitering-borderline-gate-max-movement-extent", type=float, default=70.0)
    behavior_eval.add_argument("--loitering-borderline-gate-max-p90-speed", type=float, default=4.0)
    behavior_eval.add_argument("--loitering-borderline-gate-min-revisit-ratio", type=float, default=0.88)
    behavior_eval.add_argument("--running-borderline-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    behavior_eval.add_argument("--running-borderline-gate-max-score", type=float, default=0.75)
    behavior_eval.add_argument("--running-borderline-gate-min-stationary-ratio", type=float, default=0.80)
    behavior_eval.add_argument("--running-borderline-gate-max-movement-extent", type=float, default=20.0)
    behavior_eval.add_argument("--running-borderline-gate-max-p90-speed", type=float, default=2.0)
    behavior_eval.add_argument("--quality-adaptive-loitering-enabled", action=argparse.BooleanOptionalAction, default=True)
    behavior_eval.add_argument("--quality-adaptive-loitering-long-track-frames", type=float, default=90.0)
    behavior_eval.add_argument("--quality-adaptive-loitering-long-track-max-score", type=float, default=0.82)
    behavior_eval.add_argument("--quality-adaptive-loitering-long-track-min-revisit-ratio", type=float, default=0.88)
    behavior_eval.add_argument("--source-aware-running-gate-enabled", action=argparse.BooleanOptionalAction, default=False)
    behavior_eval.add_argument("--source-aware-rswacv-running-max-score", type=float, default=0.68)
    behavior_eval.add_argument("--source-aware-rswacv-running-max-movement-extent", type=float, default=10.0)
    behavior_eval.add_argument("--source-aware-rswacv-running-max-p90-speed", type=float, default=1.5)

    behavior_expand = subparsers.add_parser(
        "expand-behavior-dataset",
        help="Expand Avenue pseudo labels with positive windows and relaxed abnormal-track promotion.",
    )
    behavior_expand.add_argument("--input-path", default="output/avenue_pseudo_labels/tracks.jsonl")
    behavior_expand.add_argument("--output-dir", default="output/avenue_pseudo_labels_expanded")

    behavior_reconstruct = subparsers.add_parser(
        "reconstruct-behavior-windows",
        help="Reconstruct denser behavior windows from the current merged behavior dataset.",
    )
    behavior_reconstruct.add_argument("--input-path", default="output/behavior_hard_negatives/subset20.jsonl")
    behavior_reconstruct.add_argument("--output-dir", default="output/behavior_reconstructed")
    behavior_reconstruct.add_argument("--max-normal-windows-per-track", type=int, default=3)
    behavior_reconstruct.add_argument("--normal-to-positive-ratio", type=float, default=2.0)
    behavior_reconstruct.add_argument("--seed", type=int, default=42)

    behavior_windows = subparsers.add_parser(
        "build-avenue-behavior-windows",
        help="Construct stricter window-level behavior samples from Avenue mask-supported tracks.",
    )
    behavior_windows.add_argument("--input-path", default="output/avenue_pseudo_labels/tracks.jsonl")
    behavior_windows.add_argument("--output-dir", default="output/behavior_reconstructed/avenue_mask_windows_v1")
    behavior_windows.add_argument("--loiter-window", type=int, default=72)
    behavior_windows.add_argument("--running-window", type=int, default=24)
    behavior_windows.add_argument("--normal-window", type=int, default=48)
    behavior_windows.add_argument("--loiter-stride", type=int, default=12)
    behavior_windows.add_argument("--running-stride", type=int, default=6)
    behavior_windows.add_argument("--normal-stride", type=int, default=24)
    behavior_windows.add_argument("--min-track-frames", type=int, default=24)
    behavior_windows.add_argument("--min-loiter-support-frames", type=int, default=24)
    behavior_windows.add_argument("--min-loiter-support-ratio", type=float, default=0.45)
    behavior_windows.add_argument("--min-loiter-stationary-ratio", type=float, default=0.15)
    behavior_windows.add_argument("--min-loiter-revisit-ratio", type=float, default=0.02)
    behavior_windows.add_argument("--max-loiter-straightness", type=float, default=0.65)
    behavior_windows.add_argument("--min-running-support-frames", type=int, default=8)
    behavior_windows.add_argument("--min-running-support-ratio", type=float, default=0.30)
    behavior_windows.add_argument("--min-running-high-speed-ratio", type=float, default=0.20)
    behavior_windows.add_argument("--min-running-avg-speed", type=float, default=8.0)
    behavior_windows.add_argument("--min-running-p90-speed", type=float, default=15.57)
    behavior_windows.add_argument("--min-running-movement-extent", type=float, default=120.0)
    behavior_windows.add_argument("--max-normal-support-frames", type=int, default=1)
    behavior_windows.add_argument("--max-normal-support-ratio", type=float, default=0.02)
    behavior_windows.add_argument("--max-normal-windows-per-track", type=int, default=2)
    behavior_windows.add_argument("--normal-to-positive-ratio", type=float, default=1.5)
    behavior_windows.add_argument("--seed", type=int, default=42)

    behavior_hardneg = subparsers.add_parser(
        "mine-hard-negatives",
        help="Mine hard negative loitering false positives from Avenue and merge them into the behavior dataset.",
    )
    behavior_hardneg.add_argument("--avenue-root", default="data/CUHK_Avenue/Avenue Dataset")
    behavior_hardneg.add_argument("--ground-truth-root", default="data/CUHK_Avenue/ground_truth_demo/testing_label_mask")
    behavior_hardneg.add_argument("--detector-model", default="output/training/mot17_person_gpu_40e_960_pretrained_w2/weights/best.pt")
    behavior_hardneg.add_argument("--behavior-model-path", default="output/behavior_training/avenue_behavior_mlp_seed2026/best.pt")
    behavior_hardneg.add_argument("--input-dataset", default="output/avenue_pseudo_labels_filtered/tracks_filtered.jsonl")
    behavior_hardneg.add_argument("--output-dir", default="output/behavior_hard_negatives")
    behavior_hardneg.add_argument("--sequence-ids", nargs="*", default=["01", "03", "04", "05", "17", "18"])
    behavior_hardneg.add_argument("--device", default=0)
    behavior_hardneg.add_argument("--conf-threshold", type=float, default=0.1)
    behavior_hardneg.add_argument("--min-track-frames", type=int, default=72)
    behavior_hardneg.add_argument("--min-loiter-alarm-frames", type=int, default=8)
    behavior_hardneg.add_argument("--min-model-loiter-frames", type=int, default=8)
    behavior_hardneg.add_argument("--max-support-ratio", type=float, default=0.05)
    behavior_hardneg.add_argument("--max-overlap-ratio", type=float, default=0.05)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        converter = MOT17ToYOLOConverter(
            PrepareConfig(
                mot_root=args.mot_root,
                output_dir=args.output_dir,
                split_name=args.split_name,
                detector_filter=args.detector,
                min_visibility=args.min_visibility,
                val_ratio=args.val_ratio,
                val_sequences=tuple(args.val_sequences),
                overwrite=args.overwrite,
                dense_small_repeat_factor=args.dense_small_repeat_factor,
                dense_small_max_repeat_frames=args.dense_small_max_repeat_frames,
            dense_small_min_gt_count=args.dense_small_min_gt_count,
            dense_small_min_small_ratio=args.dense_small_min_small_ratio,
            dense_small_max_median_area_ratio=args.dense_small_max_median_area_ratio,
            small_box_area_ratio_thresh=args.small_box_area_ratio_thresh,
            dense_small_crop_enable=args.dense_small_crop_enable,
            dense_small_crop_max_frames=args.dense_small_crop_max_frames,
            dense_small_crop_width_ratio=args.dense_small_crop_width_ratio,
            dense_small_crop_height_ratio=args.dense_small_crop_height_ratio,
            dense_small_crop_min_boxes=args.dense_small_crop_min_boxes,
        )
        )
        result = converter.prepare()
    elif args.command == "calibrate":
        result = calibrate_behavior_thresholds(
            CalibrationConfig(
                mot_root=args.mot_root,
                output_path=args.output_path,
                split_name=args.split_name,
                detector_filter=args.detector,
                min_visibility=args.min_visibility,
            )
        )
    elif args.command == "train":
        result = train_detector(
            TrainConfig(
                data=args.data,
                model=args.model,
                project=args.project,
                name=args.name,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                patience=args.patience,
                device=args.device,
            )
        )
    elif args.command == "validate":
        result = validate_detector(
            ValidateConfig(
                data=args.data,
                weights=args.weights,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                split=args.split,
                workers=args.workers,
            )
        )
    elif args.command == "eval-mot-tracking":
        result = evaluate_mot_tracking(
            MOTTrackingEvalConfig(
                config_path=args.config_path,
                mot_root=args.mot_root,
                output_path=args.output_path,
                save_mot_dir=args.save_mot_dir,
                split_name=args.split_name,
                detector_filter=args.detector,
                sequence_names=tuple(args.sequence_names),
                sequence_runtime_overrides_path=args.sequence_runtime_overrides_path,
                include_classes=tuple(args.include_classes),
                min_visibility=args.min_visibility,
                min_iou=args.min_iou,
                max_frames=args.max_frames,
                apply_runtime_profile=args.apply_runtime_profile,
                model_path=args.model_path,
                device=args.device,
                imgsz=args.imgsz,
                max_det=args.max_det,
                nms_iou=args.nms_iou,
                half=args.half,
                augment=args.augment,
                predict_conf=args.predict_conf,
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                match_thresh=args.match_thresh,
                low_match_thresh=args.low_match_thresh,
                unconfirmed_match_thresh=args.unconfirmed_match_thresh,
                score_fusion_weight=args.score_fusion_weight,
                max_time_lost=args.max_time_lost,
                min_box_area=args.min_box_area,
                appearance_enabled=args.appearance_enabled,
                appearance_weight=args.appearance_weight,
                appearance_ambiguity_margin=args.appearance_ambiguity_margin,
                appearance_feature_mode=args.appearance_feature_mode,
                appearance_hist_bins=tuple(args.appearance_hist_bins) if args.appearance_hist_bins is not None else None,
                appearance_min_box_size=args.appearance_min_box_size,
                appearance_reid_model=args.appearance_reid_model,
                appearance_reid_weights=args.appearance_reid_weights,
                appearance_reid_device=args.appearance_reid_device,
                appearance_reid_input_size=tuple(args.appearance_reid_input_size) if args.appearance_reid_input_size is not None else None,
                appearance_reid_flip_aug=args.appearance_reid_flip_aug,
                appearance_all_valid=args.appearance_all_valid,
                motion_gate_enabled=args.motion_gate_enabled,
                motion_gate_thresh=args.motion_gate_thresh,
                crowd_boost_enabled=args.crowd_boost_enabled,
                crowd_boost_det_count=args.crowd_boost_det_count,
                crowd_match_thresh=args.crowd_match_thresh,
                crowd_low_match_thresh=args.crowd_low_match_thresh,
                crowd_appearance_weight=args.crowd_appearance_weight,
                crowd_boost_min_small_ratio=args.crowd_boost_min_small_ratio,
                crowd_boost_max_median_area_ratio=args.crowd_boost_max_median_area_ratio,
                crowd_boost_small_area_ratio_thresh=args.crowd_boost_small_area_ratio_thresh,
            )
        )
    elif args.command == "validate-avenue":
        result = validate_on_avenue(
            AvenueValidationConfig(
                avenue_root=args.avenue_root,
                ground_truth_root=args.ground_truth_root,
                model=args.model,
                output_path=args.output_path,
                device=args.device,
                imgsz=args.imgsz,
                max_det=args.max_det,
                half=args.half,
                augment=args.augment,
                conf_threshold=args.conf_threshold,
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                match_thresh=args.match_thresh,
                low_match_thresh=args.low_match_thresh,
                unconfirmed_match_thresh=args.unconfirmed_match_thresh,
                score_fusion_weight=args.score_fusion_weight,
                max_time_lost=args.max_time_lost,
                min_box_area=args.min_box_area,
                appearance_enabled=args.appearance_enabled,
                appearance_weight=args.appearance_weight,
                appearance_ambiguity_margin=args.appearance_ambiguity_margin,
                appearance_feature_mode=args.appearance_feature_mode,
                appearance_hist_bins=tuple(args.appearance_hist_bins),
                appearance_min_box_size=args.appearance_min_box_size,
                appearance_reid_model=args.appearance_reid_model,
                appearance_reid_weights=args.appearance_reid_weights,
                appearance_reid_device=args.appearance_reid_device,
                appearance_reid_input_size=tuple(args.appearance_reid_input_size),
                behavior_mode=args.behavior_mode,
                behavior_model_path=args.behavior_model_path,
                behavior_secondary_model_path=args.behavior_secondary_model_path,
                behavior_ensemble_primary_weight=args.behavior_ensemble_primary_weight,
                behavior_ensemble_mode=args.behavior_ensemble_mode,
                behavior_model_score_thresh=args.behavior_model_score_thresh,
                behavior_model_min_frames=args.behavior_model_min_frames,
                behavior_model_max_tracks=args.behavior_model_max_tracks,
                behavior_model_resume_tracks=args.behavior_model_resume_tracks,
                behavior_secondary_max_tracks=args.behavior_secondary_max_tracks,
                behavior_secondary_resume_tracks=args.behavior_secondary_resume_tracks,
                behavior_secondary_loitering_only=args.behavior_secondary_loitering_only,
                behavior_model_eval_interval=args.behavior_model_eval_interval,
                loitering_hybrid_mode=args.loitering_hybrid_mode,
                loitering_model_support_thresh=args.loitering_model_support_thresh,
                loitering_model_score_thresh=args.loitering_model_score_thresh,
                running_model_score_thresh=args.running_model_score_thresh,
                loitering_model_min_frames=args.loitering_model_min_frames,
                loitering_activate_frames=args.loitering_activate_frames,
                loitering_support_activate_frames=args.loitering_support_activate_frames,
                loitering_support_block_running=args.loitering_support_block_running,
                loitering_context_gate_support_only=args.loitering_context_gate_support_only,
                running_loitering_arb_enabled=args.running_loitering_arb_enabled,
                running_loitering_min_loitering_score=args.running_loitering_min_loitering_score,
                running_loitering_min_stationary_ratio=args.running_loitering_min_stationary_ratio,
                running_loitering_max_movement_extent=args.running_loitering_max_movement_extent,
                running_loitering_max_p90_speed=args.running_loitering_max_p90_speed,
                loitering_release_frames=args.loitering_release_frames,
                loitering_model_max_avg_speed=args.loitering_model_max_avg_speed,
                loitering_model_min_movement_extent=args.loitering_model_min_movement_extent,
                loitering_model_min_centroid_radius=args.loitering_model_min_centroid_radius,
                loitering_model_max_movement_extent=args.loitering_model_max_movement_extent,
                loitering_model_max_centroid_radius=args.loitering_model_max_centroid_radius,
                loitering_model_min_stationary_ratio=args.loitering_model_min_stationary_ratio,
                loitering_model_min_revisit_ratio=args.loitering_model_min_revisit_ratio,
                loitering_model_max_unique_cell_ratio=args.loitering_model_max_unique_cell_ratio,
                loitering_model_min_max_cell_occupancy_ratio=args.loitering_model_min_max_cell_occupancy_ratio,
                loitering_model_max_straightness=args.loitering_model_max_straightness,
                loitering_rule_min_stationary_ratio=args.loitering_rule_min_stationary_ratio,
                loitering_rule_min_revisit_ratio=args.loitering_rule_min_revisit_ratio,
                loitering_rule_min_movement_extent=args.loitering_rule_min_movement_extent,
                loitering_rule_min_centroid_radius=args.loitering_rule_min_centroid_radius,
                loitering_rule_max_straightness=args.loitering_rule_max_straightness,
                loitering_rule_max_displacement_ratio=args.loitering_rule_max_displacement_ratio,
                loitering_max_neighbor_count=args.loitering_max_neighbor_count,
                loitering_neighbor_radius=args.loitering_neighbor_radius,
                running_model_min_avg_speed=args.running_model_min_avg_speed,
                running_model_min_p90_speed=args.running_model_min_p90_speed,
                running_model_min_movement_extent=args.running_model_min_movement_extent,
                enable_loitering=args.enable_loitering,
                enable_running=args.enable_running,
                enable_intrusion=args.enable_intrusion,
                enable_cross_line=args.enable_cross_line,
                loiter_frames=args.loiter_frames,
                loiter_radius=args.loiter_radius,
                loiter_speed=args.loiter_speed,
                running_speed=args.running_speed,
                running_frames=args.running_frames,
                max_videos=args.max_videos,
                sequence_ids=tuple(args.sequence_ids),
                save_demo_frames=args.save_demo_frames,
                demo_dir=args.demo_dir,
            )
        )
    elif args.command == "validate-ubnormal":
        result = validate_on_ubnormal(
            UBnormalValidationConfig(
                manifest_path=args.manifest_path,
                config_path=args.config_path,
                output_path=args.output_path,
                max_videos=args.max_videos,
                sequence_ids=tuple(args.sequence_ids),
                save_demo_frames=args.save_demo_frames,
                demo_dir=args.demo_dir,
            )
        )
    elif args.command == "generate-avenue-pseudo-labels":
        result = generate_avenue_pseudo_labels(
            AvenuePseudoLabelConfig(
                avenue_root=args.avenue_root,
                ground_truth_root=args.ground_truth_root,
                model=args.model,
                output_dir=args.output_dir,
                device=args.device,
                conf_threshold=args.conf_threshold,
                track_high_thresh=args.track_high_thresh,
                track_low_thresh=args.track_low_thresh,
                new_track_thresh=args.new_track_thresh,
                match_thresh=args.match_thresh,
                max_time_lost=args.max_time_lost,
                min_box_area=args.min_box_area,
                loiter_frames=args.loiter_frames,
                loiter_radius=args.loiter_radius,
                loiter_speed=args.loiter_speed,
                running_speed=args.running_speed,
                running_frames=args.running_frames,
                min_track_frames=args.min_track_frames,
                min_mask_overlap_frames=args.min_mask_overlap_frames,
                min_mask_overlap_ratio=args.min_mask_overlap_ratio,
                running_min_high_speed_ratio=args.running_min_high_speed_ratio,
                loiter_radius_multiplier=args.loiter_radius_multiplier,
                loiter_speed_multiplier=args.loiter_speed_multiplier,
                running_extent_multiplier=args.running_extent_multiplier,
                max_videos=args.max_videos,
                sequence_ids=tuple(args.sequence_ids),
            )
        )
    elif args.command == "train-behavior":
        result = train_behavior_classifier(
            BehaviorClassifierTrainConfig(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                run_name=args.run_name,
                device=args.device,
                model_type=args.model_type,
                sequence_length=args.sequence_length,
                labels=tuple(args.labels),
                val_ratio=args.val_ratio,
                val_split_mode=args.val_split_mode,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                class_balance_beta=args.class_balance_beta,
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
                patience=args.patience,
            )
        )
    elif args.command == "eval-behavior-model":
        result = evaluate_behavior_model(
            BehaviorModelEvalConfig(
                checkpoint_path=args.checkpoint_path,
                dataset_path=args.dataset_path,
                output_path=args.output_path,
                device=args.device,
                loitering_min_score=args.loitering_min_score,
                running_min_score=args.running_min_score,
                running_loitering_arb_enabled=args.running_loitering_arb_enabled,
                running_loitering_min_loitering_score=args.running_loitering_min_loitering_score,
                running_loitering_min_stationary_ratio=args.running_loitering_min_stationary_ratio,
                running_loitering_max_movement_extent=args.running_loitering_max_movement_extent,
                running_loitering_max_p90_speed=args.running_loitering_max_p90_speed,
                loitering_borderline_gate_enabled=args.loitering_borderline_gate_enabled,
                loitering_borderline_gate_max_score=args.loitering_borderline_gate_max_score,
                loitering_borderline_gate_min_stationary_ratio=args.loitering_borderline_gate_min_stationary_ratio,
                loitering_borderline_gate_max_movement_extent=args.loitering_borderline_gate_max_movement_extent,
                loitering_borderline_gate_max_p90_speed=args.loitering_borderline_gate_max_p90_speed,
                loitering_borderline_gate_min_revisit_ratio=args.loitering_borderline_gate_min_revisit_ratio,
                running_borderline_gate_enabled=args.running_borderline_gate_enabled,
                running_borderline_gate_max_score=args.running_borderline_gate_max_score,
                running_borderline_gate_min_stationary_ratio=args.running_borderline_gate_min_stationary_ratio,
                running_borderline_gate_max_movement_extent=args.running_borderline_gate_max_movement_extent,
                running_borderline_gate_max_p90_speed=args.running_borderline_gate_max_p90_speed,
                quality_adaptive_loitering_enabled=args.quality_adaptive_loitering_enabled,
                quality_adaptive_loitering_long_track_frames=args.quality_adaptive_loitering_long_track_frames,
                quality_adaptive_loitering_long_track_max_score=args.quality_adaptive_loitering_long_track_max_score,
                quality_adaptive_loitering_long_track_min_revisit_ratio=args.quality_adaptive_loitering_long_track_min_revisit_ratio,
                source_aware_running_gate_enabled=args.source_aware_running_gate_enabled,
                source_aware_rswacv_running_max_score=args.source_aware_rswacv_running_max_score,
                source_aware_rswacv_running_max_movement_extent=args.source_aware_rswacv_running_max_movement_extent,
                source_aware_rswacv_running_max_p90_speed=args.source_aware_rswacv_running_max_p90_speed,
            )
        )
    elif args.command == "build-avenue-behavior-windows":
        result = build_avenue_behavior_windows(
            AvenueBehaviorWindowConfig(
                input_path=args.input_path,
                output_dir=args.output_dir,
                loiter_window=args.loiter_window,
                running_window=args.running_window,
                normal_window=args.normal_window,
                loiter_stride=args.loiter_stride,
                running_stride=args.running_stride,
                normal_stride=args.normal_stride,
                min_track_frames=args.min_track_frames,
                min_loiter_support_frames=args.min_loiter_support_frames,
                min_loiter_support_ratio=args.min_loiter_support_ratio,
                min_loiter_stationary_ratio=args.min_loiter_stationary_ratio,
                min_loiter_revisit_ratio=args.min_loiter_revisit_ratio,
                max_loiter_straightness=args.max_loiter_straightness,
                min_running_support_frames=args.min_running_support_frames,
                min_running_support_ratio=args.min_running_support_ratio,
                min_running_high_speed_ratio=args.min_running_high_speed_ratio,
                min_running_avg_speed=args.min_running_avg_speed,
                min_running_p90_speed=args.min_running_p90_speed,
                min_running_movement_extent=args.min_running_movement_extent,
                max_normal_support_frames=args.max_normal_support_frames,
                max_normal_support_ratio=args.max_normal_support_ratio,
                max_normal_windows_per_track=args.max_normal_windows_per_track,
                normal_to_positive_ratio=args.normal_to_positive_ratio,
                seed=args.seed,
            )
        )
    elif args.command == "mine-hard-negatives":
        result = mine_behavior_hard_negatives(
            avenue_root=args.avenue_root,
            ground_truth_root=args.ground_truth_root,
            detector_model=args.detector_model,
            behavior_model_path=args.behavior_model_path,
            input_dataset=args.input_dataset,
            output_dir=args.output_dir,
            sequence_ids=tuple(args.sequence_ids),
            device=args.device,
            conf_threshold=args.conf_threshold,
            min_track_frames=args.min_track_frames,
            min_loiter_alarm_frames=args.min_loiter_alarm_frames,
            min_model_loiter_frames=args.min_model_loiter_frames,
            max_support_ratio=args.max_support_ratio,
            max_overlap_ratio=args.max_overlap_ratio,
        )
    elif args.command == "reconstruct-behavior-windows":
        result = reconstruct_behavior_windows(
            input_path=args.input_path,
            output_dir=args.output_dir,
            max_normal_windows_per_track=args.max_normal_windows_per_track,
            normal_to_positive_ratio=args.normal_to_positive_ratio,
            seed=args.seed,
        )
    else:
        result = expand_behavior_dataset(
            input_path=args.input_path,
            output_dir=args.output_dir,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

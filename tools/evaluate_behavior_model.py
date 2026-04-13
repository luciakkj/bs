from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.behavior_model_eval import evaluate_behavior_model
from training.config import BehaviorModelEvalConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavior classifier checkpoint on a JSONL dataset.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to behavior model checkpoint")
    parser.add_argument("--dataset-path", required=True, help="Path to tracks.jsonl dataset")
    parser.add_argument("--device", default="", help="Torch device override")
    parser.add_argument("--output-path", required=True, help="Path to output metrics JSON")
    parser.add_argument(
        "--loitering-min-score",
        type=float,
        default=0.595,
        help="Recommended minimum probability required to predict 'loitering'.",
    )
    parser.add_argument(
        "--running-min-score",
        type=float,
        default=None,
        help="Optional minimum probability required to predict 'running'.",
    )
    parser.add_argument("--running-loitering-arb-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--running-loitering-min-loitering-score", type=float, default=0.695)
    parser.add_argument("--running-loitering-min-stationary-ratio", type=float, default=0.88)
    parser.add_argument("--running-loitering-max-movement-extent", type=float, default=55.0)
    parser.add_argument("--running-loitering-max-p90-speed", type=float, default=3.0)
    parser.add_argument("--loitering-borderline-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loitering-borderline-gate-max-score", type=float, default=0.76)
    parser.add_argument("--loitering-borderline-gate-min-stationary-ratio", type=float, default=0.70)
    parser.add_argument("--loitering-borderline-gate-max-movement-extent", type=float, default=70.0)
    parser.add_argument("--loitering-borderline-gate-max-p90-speed", type=float, default=4.0)
    parser.add_argument("--loitering-borderline-gate-min-revisit-ratio", type=float, default=0.88)
    parser.add_argument("--running-borderline-gate-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--running-borderline-gate-max-score", type=float, default=0.75)
    parser.add_argument("--running-borderline-gate-min-stationary-ratio", type=float, default=0.80)
    parser.add_argument("--running-borderline-gate-max-movement-extent", type=float, default=20.0)
    parser.add_argument("--running-borderline-gate-max-p90-speed", type=float, default=2.0)
    parser.add_argument("--quality-adaptive-loitering-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-adaptive-loitering-long-track-frames", type=float, default=90.0)
    parser.add_argument("--quality-adaptive-loitering-long-track-max-score", type=float, default=0.82)
    parser.add_argument("--quality-adaptive-loitering-long-track-min-revisit-ratio", type=float, default=0.88)
    parser.add_argument("--source-aware-running-gate-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source-aware-rswacv-running-max-score", type=float, default=0.68)
    parser.add_argument("--source-aware-rswacv-running-max-movement-extent", type=float, default=10.0)
    parser.add_argument("--source-aware-rswacv-running-max-p90-speed", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = evaluate_behavior_model(
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
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

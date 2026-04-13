from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter behavior JSONL samples by label and feature thresholds.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-label", default="loitering")
    parser.add_argument("--min-stationary-ratio", type=float, default=None)
    parser.add_argument("--max-movement-extent", type=float, default=None)
    parser.add_argument("--max-p90-speed", type=float, default=None)
    parser.add_argument("--max-avg-speed", type=float, default=None)
    parser.add_argument("--min-revisit-ratio", type=float, default=None)
    parser.add_argument("--max-straightness", type=float, default=None)
    parser.add_argument("--min-frame-count", type=int, default=None)
    return parser.parse_args()


def _passes(row: dict[str, object], args: argparse.Namespace) -> bool:
    label = str(row.get("primary_label", ""))
    if label != str(args.target_label):
        return True

    features = row.get("features") or {}
    if args.min_stationary_ratio is not None and float(features.get("stationary_ratio", 0.0)) < float(args.min_stationary_ratio):
        return False
    if args.max_movement_extent is not None and float(features.get("movement_extent", 0.0)) > float(args.max_movement_extent):
        return False
    if args.max_p90_speed is not None and float(features.get("p90_speed", 0.0)) > float(args.max_p90_speed):
        return False
    if args.max_avg_speed is not None and float(features.get("avg_speed", 0.0)) > float(args.max_avg_speed):
        return False
    if args.min_revisit_ratio is not None and float(features.get("revisit_ratio", 0.0)) < float(args.min_revisit_ratio):
        return False
    if args.max_straightness is not None and float(features.get("straightness", 1.0)) > float(args.max_straightness):
        return False
    if args.min_frame_count is not None and int(row.get("frame_count", 0)) < int(args.min_frame_count):
        return False
    return True


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [row for row in rows if _passes(row, args)]

    tracks_path = output_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts_before = Counter(str(row.get("primary_label", "")) for row in rows)
    label_counts_after = Counter(str(row.get("primary_label", "")) for row in kept)
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "target_label": str(args.target_label),
        "rows_before": len(rows),
        "rows_after": len(kept),
        "label_counts_before": dict(label_counts_before),
        "label_counts_after": dict(label_counts_after),
        "filters": {
            "min_stationary_ratio": args.min_stationary_ratio,
            "max_movement_extent": args.max_movement_extent,
            "max_p90_speed": args.max_p90_speed,
            "max_avg_speed": args.max_avg_speed,
            "min_revisit_ratio": args.min_revisit_ratio,
            "max_straightness": args.max_straightness,
            "min_frame_count": args.min_frame_count,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

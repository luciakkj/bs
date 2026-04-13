from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from training.behavior_classifier import _load_samples


FEATURE_KEYS = (
    "frame_count",
    "avg_speed",
    "p90_speed",
    "movement_extent",
    "centroid_radius",
    "straightness",
    "stationary_ratio",
    "revisit_ratio",
    "unique_cell_ratio",
    "max_cell_occupancy_ratio",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze loitering false negatives for a behavior model.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--loitering-min-score", type=float, default=0.60)
    parser.add_argument("--running-min-score", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def _apply_label_thresholds(
    predicted_label: str | None,
    probs: dict[str, float],
    labels: tuple[str, ...],
    *,
    loitering_min_score: float | None,
    running_min_score: float | None,
) -> str | None:
    if predicted_label is None:
        return None

    def _fallback(excluded_label: str) -> str | None:
        candidates = [
            (float(score), label_name)
            for label_name, score in probs.items()
            if label_name in labels and label_name != excluded_label
        ]
        if not candidates:
            return predicted_label
        candidates.sort(reverse=True)
        return candidates[0][1]

    if predicted_label == "loitering" and loitering_min_score is not None:
        if float(probs.get("loitering", 0.0)) < float(loitering_min_score):
            return _fallback("loitering")

    if predicted_label == "running" and running_min_score is not None:
        if float(probs.get("running", 0.0)) < float(running_min_score):
            return _fallback("running")

    return predicted_label


def _mean_feature(rows: list[dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        feature_dict = row.get("features", {})
        if key in feature_dict:
            values.append(float(feature_dict[key]))
    return round(float(statistics.fmean(values)), 4) if values else 0.0


def _classify_reason(
    raw_label: str | None,
    final_label: str | None,
    loitering_prob: float,
    loitering_min_score: float,
) -> str:
    if raw_label == "loitering" and final_label != "loitering":
        return "threshold_blocked"
    if final_label == "normal":
        if loitering_prob >= max(0.4, loitering_min_score - 0.1):
            return "borderline_normal_confusion"
        return "weak_loitering_signal"
    if final_label == "running":
        return "running_confusion"
    if final_label is None:
        return "invalid_prediction"
    return "other_confusion"


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = TrajectoryBehaviorClassifier(
        checkpoint_path=checkpoint_path,
        device=str(args.device).strip() or None,
    )
    labels = tuple(classifier.labels)
    samples = _load_samples(dataset_path, labels)

    true_loitering_rows: list[dict[str, object]] = []
    true_positive_rows: list[dict[str, object]] = []
    false_negative_rows: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    predicted_label_counts: Counter[str] = Counter()

    for sample in samples:
        if str(sample.get("primary_label")) != "loitering":
            continue
        true_loitering_rows.append(sample)
        trajectory_payload = sample.get("trajectory", {})
        feature_dict = sample.get("features")

        raw_label, _, probs, _ = classifier.predict_track_info(
            {
                "trajectory": trajectory_payload.get("centers", []),
                "speed_history": trajectory_payload.get("speeds", []),
            },
            target_labels=labels,
        )
        if raw_label is None:
            raw_label, _, probs = classifier.predict_payload(trajectory_payload, feature_dict)

        final_label = _apply_label_thresholds(
            raw_label,
            probs,
            labels,
            loitering_min_score=args.loitering_min_score,
            running_min_score=args.running_min_score,
        )
        if final_label == "loitering":
            true_positive_rows.append(sample)
            continue

        loitering_prob = float(probs.get("loitering", 0.0))
        reason = _classify_reason(
            raw_label,
            final_label,
            loitering_prob,
            float(args.loitering_min_score),
        )
        predicted_label_counts[str(final_label)] += 1
        reason_counts[reason] += 1
        false_negative_rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_track_id": sample.get("source_track_id"),
                "sequence_id": sample.get("sequence_id"),
                "dataset_name": sample.get("dataset_name"),
                "frame_count": int(sample.get("frame_count", 0) or 0),
                "raw_predicted_label": raw_label,
                "final_predicted_label": final_label,
                "reason": reason,
                "probs": {label: round(float(probs.get(label, 0.0)), 4) for label in labels},
                "features": {
                    key: round(float(sample.get("features", {}).get(key, 0.0) or 0.0), 4)
                    for key in FEATURE_KEYS
                },
            }
        )

    false_negative_rows.sort(
        key=lambda row: (
            row["reason"] != "threshold_blocked",
            row["probs"].get("loitering", 0.0),
        ),
        reverse=True,
    )

    feature_comparison = {
        key: {
            "true_positive_mean": _mean_feature(true_positive_rows, key),
            "false_negative_mean": _mean_feature(false_negative_rows, key),
        }
        for key in FEATURE_KEYS
    }

    output = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "loitering_min_score": args.loitering_min_score,
        "running_min_score": args.running_min_score,
        "true_loitering_count": len(true_loitering_rows),
        "true_positive_count": len(true_positive_rows),
        "false_negative_count": len(false_negative_rows),
        "loitering_recall": round(len(true_positive_rows) / max(len(true_loitering_rows), 1), 4),
        "false_negative_predicted_label_counts": dict(predicted_label_counts),
        "false_negative_reason_counts": dict(reason_counts),
        "feature_comparison": feature_comparison,
        "top_false_negatives": false_negative_rows[: max(1, int(args.top_k))],
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

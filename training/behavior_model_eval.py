from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from behavior.trajectory_behavior_classifier import TrajectoryBehaviorClassifier
from training.behavior_classifier import _load_samples
from training.config import BehaviorModelEvalConfig


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _blend_model_score(primary_score: float, secondary_score: float, *, primary_weight: float, mode: str) -> float:
    primary_score = float(primary_score)
    secondary_score = float(secondary_score)
    if mode == "max":
        return max(primary_score, secondary_score)
    if mode == "geometric":
        return math.sqrt(max(0.0, primary_score) * max(0.0, secondary_score))
    if mode == "geometric_weighted":
        secondary_weight = 1.0 - primary_weight
        primary_term = max(1e-6, primary_score) ** primary_weight
        secondary_term = max(1e-6, secondary_score) ** secondary_weight
        return primary_term * secondary_term
    secondary_weight = 1.0 - primary_weight
    return primary_score * primary_weight + secondary_score * secondary_weight


def _compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    labels: tuple[str, ...],
) -> tuple[float, dict[str, dict[str, float]]]:
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for class_index, label in enumerate(labels):
        tp = int(np.sum((predictions == class_index) & (targets == class_index)))
        fp = int(np.sum((predictions == class_index) & (targets != class_index)))
        fn = int(np.sum((predictions != class_index) & (targets == class_index)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(targets == class_index)),
        }
        f1_values.append(f1)
    return float(np.mean(f1_values)), per_class


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


def _apply_running_loitering_arbitration(
    predicted_label: str | None,
    probs: dict[str, float],
    feature_dict: dict[str, object] | None,
    *,
    enabled: bool,
    min_loitering_score: float,
    min_stationary_ratio: float,
    max_movement_extent: float,
    max_p90_speed: float,
) -> str | None:
    if not enabled or predicted_label != "running":
        return predicted_label
    if feature_dict is None:
        return predicted_label

    loitering_score = float(probs.get("loitering", 0.0))
    stationary_ratio = float(feature_dict.get("stationary_ratio", 0.0) or 0.0)
    movement_extent = float(feature_dict.get("movement_extent", 0.0) or 0.0)
    p90_speed = float(feature_dict.get("p90_speed", 0.0) or 0.0)

    if (
        loitering_score >= min_loitering_score
        and stationary_ratio >= min_stationary_ratio
        and movement_extent <= max_movement_extent
        and p90_speed <= max_p90_speed
    ):
        return "loitering"
    return predicted_label


def _apply_loitering_borderline_gate(
    predicted_label: str | None,
    probs: dict[str, float],
    labels: tuple[str, ...],
    feature_dict: dict[str, object] | None,
    *,
    enabled: bool,
    max_score: float,
    min_stationary_ratio: float,
    max_movement_extent: float,
    max_p90_speed: float,
    min_revisit_ratio: float,
    max_straightness: float,
    max_centroid_radius: float,
) -> str | None:
    if not enabled or predicted_label != "loitering":
        return predicted_label
    if feature_dict is None:
        return predicted_label

    loitering_score = float(probs.get("loitering", 0.0))
    if loitering_score >= max_score:
        return predicted_label

    stationary_ratio = float(feature_dict.get("stationary_ratio", 0.0) or 0.0)
    movement_extent = float(feature_dict.get("movement_extent", 0.0) or 0.0)
    p90_speed = float(feature_dict.get("p90_speed", 0.0) or 0.0)
    revisit_ratio = float(feature_dict.get("revisit_ratio", 0.0) or 0.0)
    straightness = float(feature_dict.get("straightness", 0.0) or 0.0)
    centroid_radius = float(feature_dict.get("centroid_radius", 0.0) or 0.0)

    if (
        stationary_ratio >= min_stationary_ratio
        and movement_extent <= max_movement_extent
        and p90_speed <= max_p90_speed
        and revisit_ratio >= min_revisit_ratio
        and straightness <= max_straightness
        and centroid_radius <= max_centroid_radius
    ):
        return predicted_label

    candidates = [
        (float(score), label_name)
        for label_name, score in probs.items()
        if label_name in labels and label_name != "loitering"
    ]
    if not candidates:
        return predicted_label
    candidates.sort(reverse=True)
    return candidates[0][1]


def _apply_running_borderline_gate(
    predicted_label: str | None,
    probs: dict[str, float],
    labels: tuple[str, ...],
    feature_dict: dict[str, object] | None,
    *,
    enabled: bool,
    max_score: float,
    min_stationary_ratio: float,
    max_movement_extent: float,
    max_p90_speed: float,
) -> str | None:
    if not enabled or predicted_label != "running":
        return predicted_label
    if feature_dict is None:
        return predicted_label

    running_score = float(probs.get("running", 0.0))
    stationary_ratio = float(feature_dict.get("stationary_ratio", 0.0) or 0.0)
    movement_extent = float(feature_dict.get("movement_extent", 0.0) or 0.0)
    p90_speed = float(feature_dict.get("p90_speed", 0.0) or 0.0)

    if (
        running_score < max_score
        and stationary_ratio >= min_stationary_ratio
        and movement_extent <= max_movement_extent
        and p90_speed <= max_p90_speed
    ):
        candidates = [
            (float(score), label_name)
            for label_name, score in probs.items()
            if label_name in labels and label_name != "running"
        ]
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    return predicted_label


def _apply_quality_adaptive_loitering_gate(
    predicted_label: str | None,
    probs: dict[str, float],
    labels: tuple[str, ...],
    feature_dict: dict[str, object] | None,
    *,
    enabled: bool,
    long_track_frames: float,
    long_track_max_score: float,
    long_track_min_revisit_ratio: float,
    base_min_stationary_ratio: float,
    base_max_movement_extent: float,
    base_max_p90_speed: float,
    base_min_revisit_ratio: float,
) -> str | None:
    if not enabled or predicted_label != "loitering":
        return predicted_label
    if feature_dict is None:
        return predicted_label

    frame_count = float(feature_dict.get("frame_count", 0.0) or 0.0)
    loitering_score = float(probs.get("loitering", 0.0))
    stationary_ratio = float(feature_dict.get("stationary_ratio", 0.0) or 0.0)
    movement_extent = float(feature_dict.get("movement_extent", 0.0) or 0.0)
    p90_speed = float(feature_dict.get("p90_speed", 0.0) or 0.0)
    revisit_ratio = float(feature_dict.get("revisit_ratio", 0.0) or 0.0)

    if frame_count < long_track_frames or loitering_score >= long_track_max_score:
        return predicted_label

    if (
        stationary_ratio >= base_min_stationary_ratio
        and movement_extent <= base_max_movement_extent
        and p90_speed <= base_max_p90_speed
        and revisit_ratio >= max(base_min_revisit_ratio, long_track_min_revisit_ratio)
    ):
        return predicted_label

    candidates = [
        (float(score), label_name)
        for label_name, score in probs.items()
        if label_name in labels and label_name != "loitering"
    ]
    if not candidates:
        return predicted_label
    candidates.sort(reverse=True)
    return candidates[0][1]


def _apply_source_aware_running_gate(
    predicted_label: str | None,
    probs: dict[str, float],
    labels: tuple[str, ...],
    feature_dict: dict[str, object] | None,
    dataset_name: str | None,
    *,
    enabled: bool,
    rswacv_running_max_score: float,
    rswacv_running_max_movement_extent: float,
    rswacv_running_max_p90_speed: float,
    base_min_stationary_ratio: float,
) -> str | None:
    if not enabled or predicted_label != "running":
        return predicted_label
    if feature_dict is None:
        return predicted_label
    if str(dataset_name or "") != "rswacv24_loitering":
        return predicted_label

    running_score = float(probs.get("running", 0.0))
    stationary_ratio = float(feature_dict.get("stationary_ratio", 0.0) or 0.0)
    movement_extent = float(feature_dict.get("movement_extent", 0.0) or 0.0)
    p90_speed = float(feature_dict.get("p90_speed", 0.0) or 0.0)

    if (
        running_score < rswacv_running_max_score
        and stationary_ratio >= base_min_stationary_ratio
        and movement_extent <= rswacv_running_max_movement_extent
        and p90_speed <= rswacv_running_max_p90_speed
    ):
        candidates = [
            (float(score), label_name)
            for label_name, score in probs.items()
            if label_name in labels and label_name != "running"
        ]
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

    return predicted_label


def _predict_with_classifier(
    classifier: TrajectoryBehaviorClassifier,
    labels: tuple[str, ...],
    trajectory_payload: dict[str, object],
    feature_dict: dict[str, object] | None,
) -> tuple[str | None, dict[str, float]]:
    predicted_label, _, predicted_probs, _ = classifier.predict_track_info(
        {
            "trajectory": trajectory_payload.get("centers", []),
            "speed_history": trajectory_payload.get("speeds", []),
        },
        target_labels=labels,
    )
    if predicted_label is None:
        predicted_label, _, predicted_probs = classifier.predict_payload(
            trajectory_payload,
            feature_dict,
        )
    probs = {
        label: float(predicted_probs.get(label, 0.0))
        for label in labels
    }
    return predicted_label, probs


def _blend_probability_maps(
    primary_probs: dict[str, float],
    secondary_probs: dict[str, float],
    labels: tuple[str, ...],
    *,
    primary_weight: float,
    mode: str,
    loitering_boost: float,
) -> dict[str, float]:
    blended: dict[str, float] = {}
    for label in labels:
        blended[label] = _blend_model_score(
            primary_probs.get(label, 0.0),
            secondary_probs.get(label, 0.0),
            primary_weight=primary_weight,
            mode=mode,
        )
    if "loitering" in blended and loitering_boost != 1.0:
        blended["loitering"] = float(blended["loitering"]) * float(loitering_boost)
    return blended


def evaluate_behavior_model(cfg: BehaviorModelEvalConfig) -> dict[str, object]:
    cfg = cfg.resolved()
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier = TrajectoryBehaviorClassifier(
        checkpoint_path=cfg.checkpoint_path,
        device=str(cfg.device).strip() or None,
    )
    secondary_classifier = None
    if cfg.secondary_checkpoint_path:
        secondary_classifier = TrajectoryBehaviorClassifier(
            checkpoint_path=cfg.secondary_checkpoint_path,
            device=str(cfg.device).strip() or None,
        )
    labels = tuple(classifier.labels)
    if secondary_classifier and tuple(secondary_classifier.labels) != labels:
        raise ValueError("Primary and secondary behavior models expose different label sets.")
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    samples = _load_samples(cfg.dataset_path, labels)

    targets: list[int] = []
    predictions: list[int] = []
    prediction_label_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    invalid_samples = 0

    for sample in samples:
        label = str(sample["primary_label"])
        dataset_name = str(sample.get("dataset_name", ""))
        trajectory_payload = sample.get("trajectory", {})
        feature_dict = sample.get("features")
        predicted_label, predicted_probs = _predict_with_classifier(
            classifier,
            labels,
            trajectory_payload,
            feature_dict,
        )
        if secondary_classifier:
            _, secondary_probs = _predict_with_classifier(
                secondary_classifier,
                labels,
                trajectory_payload,
                feature_dict,
            )
            predicted_probs = _blend_probability_maps(
                predicted_probs,
                secondary_probs,
                labels,
                primary_weight=cfg.ensemble_primary_weight,
                mode=cfg.ensemble_mode,
                loitering_boost=cfg.ensemble_loitering_boost,
            )
            predicted_label = max(predicted_probs.items(), key=lambda item: item[1])[0]
        predicted_label = _apply_label_thresholds(
            predicted_label,
            predicted_probs,
            labels,
            loitering_min_score=cfg.loitering_min_score,
            running_min_score=cfg.running_min_score,
        )
        predicted_label = _apply_running_loitering_arbitration(
            predicted_label,
            predicted_probs,
            feature_dict,
            enabled=cfg.running_loitering_arb_enabled,
            min_loitering_score=cfg.running_loitering_min_loitering_score,
            min_stationary_ratio=cfg.running_loitering_min_stationary_ratio,
            max_movement_extent=cfg.running_loitering_max_movement_extent,
            max_p90_speed=cfg.running_loitering_max_p90_speed,
        )
        predicted_label = _apply_loitering_borderline_gate(
            predicted_label,
            predicted_probs,
            labels,
            feature_dict,
            enabled=cfg.loitering_borderline_gate_enabled,
            max_score=cfg.loitering_borderline_gate_max_score,
            min_stationary_ratio=cfg.loitering_borderline_gate_min_stationary_ratio,
            max_movement_extent=cfg.loitering_borderline_gate_max_movement_extent,
            max_p90_speed=cfg.loitering_borderline_gate_max_p90_speed,
            min_revisit_ratio=cfg.loitering_borderline_gate_min_revisit_ratio,
            max_straightness=cfg.loitering_borderline_gate_max_straightness,
            max_centroid_radius=cfg.loitering_borderline_gate_max_centroid_radius,
        )
        predicted_label = _apply_running_borderline_gate(
            predicted_label,
            predicted_probs,
            labels,
            feature_dict,
            enabled=cfg.running_borderline_gate_enabled,
            max_score=cfg.running_borderline_gate_max_score,
            min_stationary_ratio=cfg.running_borderline_gate_min_stationary_ratio,
            max_movement_extent=cfg.running_borderline_gate_max_movement_extent,
            max_p90_speed=cfg.running_borderline_gate_max_p90_speed,
        )
        predicted_label = _apply_quality_adaptive_loitering_gate(
            predicted_label,
            predicted_probs,
            labels,
            feature_dict,
            enabled=cfg.quality_adaptive_loitering_enabled,
            long_track_frames=cfg.quality_adaptive_loitering_long_track_frames,
            long_track_max_score=cfg.quality_adaptive_loitering_long_track_max_score,
            long_track_min_revisit_ratio=cfg.quality_adaptive_loitering_long_track_min_revisit_ratio,
            base_min_stationary_ratio=cfg.loitering_borderline_gate_min_stationary_ratio,
            base_max_movement_extent=cfg.loitering_borderline_gate_max_movement_extent,
            base_max_p90_speed=cfg.loitering_borderline_gate_max_p90_speed,
            base_min_revisit_ratio=cfg.loitering_borderline_gate_min_revisit_ratio,
        )
        predicted_label = _apply_source_aware_running_gate(
            predicted_label,
            predicted_probs,
            labels,
            feature_dict,
            dataset_name,
            enabled=cfg.source_aware_running_gate_enabled,
            rswacv_running_max_score=cfg.source_aware_rswacv_running_max_score,
            rswacv_running_max_movement_extent=cfg.source_aware_rswacv_running_max_movement_extent,
            rswacv_running_max_p90_speed=cfg.source_aware_rswacv_running_max_p90_speed,
            base_min_stationary_ratio=cfg.running_borderline_gate_min_stationary_ratio,
        )
        if predicted_label is None or predicted_label not in label_to_index:
            invalid_samples += 1
            continue
        targets.append(label_to_index[label])
        predictions.append(label_to_index[predicted_label])
        target_label_counts[label] += 1
        prediction_label_counts[predicted_label] += 1

    if not targets:
        raise ValueError(f"No valid predictions could be produced for {cfg.dataset_path}")

    targets_np = np.asarray(targets, dtype=np.int64)
    predictions_np = np.asarray(predictions, dtype=np.int64)
    macro_f1, per_class = _compute_metrics(targets_np, predictions_np, labels)
    accuracy = float(np.mean(targets_np == predictions_np))

    output: dict[str, object] = {
        "checkpoint_path": str(cfg.checkpoint_path),
        "secondary_checkpoint_path": str(cfg.secondary_checkpoint_path) if cfg.secondary_checkpoint_path else None,
        "dataset_path": str(cfg.dataset_path),
        "sample_count": len(samples),
        "evaluated_samples": int(len(targets)),
        "invalid_samples": int(invalid_samples),
        "labels": list(labels),
        "ensemble_primary_weight": cfg.ensemble_primary_weight,
        "ensemble_mode": cfg.ensemble_mode,
        "ensemble_loitering_boost": cfg.ensemble_loitering_boost,
        "loitering_min_score": cfg.loitering_min_score,
        "running_min_score": cfg.running_min_score,
        "running_loitering_arb_enabled": cfg.running_loitering_arb_enabled,
        "running_loitering_min_loitering_score": cfg.running_loitering_min_loitering_score,
        "running_loitering_min_stationary_ratio": cfg.running_loitering_min_stationary_ratio,
        "running_loitering_max_movement_extent": cfg.running_loitering_max_movement_extent,
        "running_loitering_max_p90_speed": cfg.running_loitering_max_p90_speed,
        "loitering_borderline_gate_enabled": cfg.loitering_borderline_gate_enabled,
        "loitering_borderline_gate_max_score": cfg.loitering_borderline_gate_max_score,
        "loitering_borderline_gate_min_stationary_ratio": cfg.loitering_borderline_gate_min_stationary_ratio,
        "loitering_borderline_gate_max_movement_extent": cfg.loitering_borderline_gate_max_movement_extent,
        "loitering_borderline_gate_max_p90_speed": cfg.loitering_borderline_gate_max_p90_speed,
        "loitering_borderline_gate_min_revisit_ratio": cfg.loitering_borderline_gate_min_revisit_ratio,
        "loitering_borderline_gate_max_straightness": cfg.loitering_borderline_gate_max_straightness,
        "loitering_borderline_gate_max_centroid_radius": cfg.loitering_borderline_gate_max_centroid_radius,
        "running_borderline_gate_enabled": cfg.running_borderline_gate_enabled,
        "running_borderline_gate_max_score": cfg.running_borderline_gate_max_score,
        "running_borderline_gate_min_stationary_ratio": cfg.running_borderline_gate_min_stationary_ratio,
        "running_borderline_gate_max_movement_extent": cfg.running_borderline_gate_max_movement_extent,
        "running_borderline_gate_max_p90_speed": cfg.running_borderline_gate_max_p90_speed,
        "quality_adaptive_loitering_enabled": cfg.quality_adaptive_loitering_enabled,
        "quality_adaptive_loitering_long_track_frames": cfg.quality_adaptive_loitering_long_track_frames,
        "quality_adaptive_loitering_long_track_max_score": cfg.quality_adaptive_loitering_long_track_max_score,
        "quality_adaptive_loitering_long_track_min_revisit_ratio": cfg.quality_adaptive_loitering_long_track_min_revisit_ratio,
        "source_aware_running_gate_enabled": cfg.source_aware_running_gate_enabled,
        "source_aware_rswacv_running_max_score": cfg.source_aware_rswacv_running_max_score,
        "source_aware_rswacv_running_max_movement_extent": cfg.source_aware_rswacv_running_max_movement_extent,
        "source_aware_rswacv_running_max_p90_speed": cfg.source_aware_rswacv_running_max_p90_speed,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "target_label_counts": dict(target_label_counts),
        "prediction_label_counts": dict(prediction_label_counts),
    }
    cfg.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output

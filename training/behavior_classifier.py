from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from behavior.trajectory_behavior_classifier import (
    BehaviorMLP,
    FEATURE_NAMES,
    TEMPORAL_STEP_FEATURE_NAMES,
    TemporalBehaviorCNN,
    TemporalFeatureFusion,
    build_temporal_sequence_array,
    features_from_trajectory_payload,
    vectorize_feature_dict,
)
from training.config import BehaviorClassifierTrainConfig


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_torch_device(device_value: str | int | None) -> torch.device:
    if device_value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_value, int):
        return torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else "cpu")

    text = str(device_value).strip().lower()
    if text.isdigit():
        return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _load_samples(dataset_path: Path, allowed_labels: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        label = sample.get("primary_label")
        if label not in allowed_labels:
            continue
        rows.append(sample)
    if not rows:
        raise ValueError(f"No usable samples found in {dataset_path}")
    return rows


def _resolve_group_key(sample: dict[str, object], group_field: str) -> str:
    if group_field == "sequence_id":
        value = sample.get("sequence_id")
        if value is None or str(value).strip() == "":
            raise ValueError("Sequence-level validation split requires every sample to include a non-empty sequence_id.")
        return str(value)

    if group_field == "source_track_id":
        value = sample.get("source_track_id", sample.get("sample_id"))
        return str(value)

    value = sample.get(group_field, sample.get("sample_id"))
    return str(value)


def _count_labels(rows: list[dict[str, object]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row["primary_label"])] += 1
    return counts


def _split_metadata(
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    *,
    group_field: str,
    strategy: str,
) -> dict[str, object]:
    train_group_ids = sorted({_resolve_group_key(row, group_field) for row in train_rows})
    val_group_ids = sorted({_resolve_group_key(row, group_field) for row in val_rows})
    return {
        "group_field": group_field,
        "strategy": strategy,
        "train_group_count": len(train_group_ids),
        "val_group_count": len(val_group_ids),
        "train_group_ids": train_group_ids,
        "val_group_ids": val_group_ids,
    }


def _stratified_track_group_split(
    samples: list[dict[str, object]],
    labels: tuple[str, ...],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {label: {} for label in labels}
    for sample in samples:
        label = sample["primary_label"]
        group_key = _resolve_group_key(sample, "source_track_id")
        grouped[label].setdefault(group_key, []).append(sample)

    for label, label_groups in grouped.items():
        group_items = list(label_groups.items())
        rng.shuffle(group_items)
        if len(group_items) < 2:
            for _, rows in group_items:
                train_rows.extend(rows)
            continue
        val_group_count = max(1, int(round(len(group_items) * val_ratio)))
        val_group_count = min(val_group_count, len(group_items) - 1)
        for _, rows in group_items[:val_group_count]:
            val_rows.extend(rows)
        for _, rows in group_items[val_group_count:]:
            train_rows.extend(rows)

    if not val_rows:
        val_rows = train_rows[-max(1, len(train_rows) // 5):]
        train_rows = train_rows[:-len(val_rows)]

    return (
        train_rows,
        val_rows,
        _split_metadata(
            train_rows,
            val_rows,
            group_field="source_track_id",
            strategy="label_stratified_track_groups",
        ),
    )


def _sequence_split_score(
    val_counts: Counter,
    train_counts: Counter,
    *,
    labels: tuple[str, ...],
    total_counts: Counter,
    total_samples: int,
    target_ratio: float,
    val_group_count: int,
) -> float | None:
    val_total = int(sum(val_counts.values()))
    if val_total <= 0 or val_total >= total_samples:
        return None

    for label in labels:
        if total_counts[label] <= 0:
            continue
        if val_counts[label] <= 0 or train_counts[label] <= 0:
            return None

    val_sample_ratio = val_total / max(total_samples, 1)
    score = abs(val_sample_ratio - target_ratio) * 80.0
    for label in labels:
        label_total = total_counts[label]
        if label_total <= 0:
            continue
        label_ratio = val_counts[label] / label_total
        score += abs(label_ratio - target_ratio) * 30.0
        minimum_supported_ratio = min(0.5, max(0.08, target_ratio * 0.6))
        if label_ratio < minimum_supported_ratio:
            score += (minimum_supported_ratio - label_ratio) * 20.0
    score += max(0, val_group_count - 3) * 1.5
    return score


def _counter_subtract(base: Counter, delta: Counter) -> Counter:
    result: Counter = Counter(base)
    result.subtract(delta)
    return result


def _sequence_level_split(
    samples: list[dict[str, object]],
    labels: tuple[str, ...],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        group_key = _resolve_group_key(sample, "sequence_id")
        grouped.setdefault(group_key, []).append(sample)

    if len(grouped) < 2:
        raise ValueError("Sequence-level validation split requires at least two unique sequence_id groups.")

    total_counts = _count_labels(samples)
    total_samples = len(samples)
    target_ratio = min(0.5, max(0.05, float(val_ratio)))
    group_ids = sorted(grouped)
    group_counts = {group_id: _count_labels(rows) for group_id, rows in grouped.items()}
    max_exhaustive_groups = 18

    best_subset: tuple[str, ...] | None = None
    best_score: float | None = None

    if len(group_ids) <= max_exhaustive_groups:
        for subset_size in range(1, len(group_ids)):
            for subset in itertools.combinations(group_ids, subset_size):
                val_counts: Counter = Counter()
                for group_id in subset:
                    val_counts.update(group_counts[group_id])
                train_counts = _counter_subtract(total_counts, val_counts)
                score = _sequence_split_score(
                    val_counts,
                    train_counts,
                    labels=labels,
                    total_counts=total_counts,
                    total_samples=total_samples,
                    target_ratio=target_ratio,
                    val_group_count=len(subset),
                )
                if score is None:
                    continue
                if best_score is None or score < best_score - 1e-9 or (
                    abs(score - best_score) <= 1e-9 and subset < best_subset
                ):
                    best_score = score
                    best_subset = subset
    else:
        rng = random.Random(seed)
        remaining_group_ids = list(group_ids)
        rng.shuffle(remaining_group_ids)
        val_group_ids: list[str] = []
        val_counts: Counter = Counter()
        train_counts: Counter = Counter(total_counts)
        target_samples = total_samples * target_ratio

        for label in sorted(labels, key=lambda item: (total_counts[item], item)):
            if val_counts[label] > 0:
                continue
            candidates = []
            for group_id in remaining_group_ids:
                group_count = group_counts[group_id]
                if group_count[label] <= 0:
                    continue
                candidate_train_counts = _counter_subtract(train_counts, group_count)
                if candidate_train_counts[label] <= 0:
                    continue
                candidates.append(
                    (
                        abs((sum(val_counts.values()) + sum(group_count.values())) - target_samples),
                        -group_count[label],
                        group_id,
                    )
                )
            if not candidates:
                continue
            _, _, selected_group_id = min(candidates)
            remaining_group_ids.remove(selected_group_id)
            selected_counts = group_counts[selected_group_id]
            val_group_ids.append(selected_group_id)
            val_counts.update(selected_counts)
            train_counts = _counter_subtract(train_counts, selected_counts)

        while remaining_group_ids:
            current_score = _sequence_split_score(
                val_counts,
                train_counts,
                labels=labels,
                total_counts=total_counts,
                total_samples=total_samples,
                target_ratio=target_ratio,
                val_group_count=len(val_group_ids),
            )
            improving_candidates = []
            for group_id in remaining_group_ids:
                group_count = group_counts[group_id]
                candidate_val_counts = Counter(val_counts)
                candidate_val_counts.update(group_count)
                candidate_train_counts = _counter_subtract(train_counts, group_count)
                candidate_score = _sequence_split_score(
                    candidate_val_counts,
                    candidate_train_counts,
                    labels=labels,
                    total_counts=total_counts,
                    total_samples=total_samples,
                    target_ratio=target_ratio,
                    val_group_count=len(val_group_ids) + 1,
                )
                if candidate_score is None:
                    continue
                if current_score is None or candidate_score < current_score - 1e-9:
                    improving_candidates.append((candidate_score, group_id))
            if not improving_candidates:
                break
            _, selected_group_id = min(improving_candidates)
            remaining_group_ids.remove(selected_group_id)
            selected_counts = group_counts[selected_group_id]
            val_group_ids.append(selected_group_id)
            val_counts.update(selected_counts)
            train_counts = _counter_subtract(train_counts, selected_counts)
            if sum(val_counts.values()) >= target_samples and all(val_counts[label] > 0 for label in labels):
                break

        best_subset = tuple(sorted(val_group_ids))

    if not best_subset:
        raise ValueError("Unable to construct a valid sequence-level validation split with label coverage on both train and val.")

    val_set = set(best_subset)
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for group_id in group_ids:
        destination = val_rows if group_id in val_set else train_rows
        destination.extend(grouped[group_id])

    return (
        train_rows,
        val_rows,
        _split_metadata(
            train_rows,
            val_rows,
            group_field="sequence_id",
            strategy="balanced_sequence_groups",
        ),
    )


def _stratified_group_split(
    samples: list[dict[str, object]],
    labels: tuple[str, ...],
    val_ratio: float,
    seed: int,
    *,
    split_mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if split_mode == "sequence":
        return _sequence_level_split(samples, labels, val_ratio, seed)
    return _stratified_track_group_split(samples, labels, val_ratio, seed)


def _samples_to_arrays(
    samples: list[dict[str, object]],
    label_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    features = []
    targets = []
    sample_infos = []
    for sample in samples:
        sample_features = sample.get("features", {})
        feature_dict = {}
        missing_feature_names = [
            feature_name
            for feature_name in FEATURE_NAMES
            if feature_name != "frame_count" and feature_name not in sample_features
        ]
        if missing_feature_names:
            trajectory = sample.get("trajectory", {})
            derived_features = features_from_trajectory_payload(
                trajectory,
                high_speed_threshold=15.57,
            )
            if derived_features is None:
                raise ValueError(f"Unable to derive missing features for sample: {sample.get('sample_id')}")
        else:
            derived_features = None

        for feature_name in FEATURE_NAMES:
            if feature_name == "frame_count":
                feature_dict[feature_name] = float(sample["frame_count"])
            elif feature_name in sample_features:
                feature_dict[feature_name] = float(sample_features[feature_name])
            else:
                feature_dict[feature_name] = float(derived_features[feature_name])
        features.append(vectorize_feature_dict(feature_dict))
        targets.append(label_to_index[sample["primary_label"]])
        sample_infos.append(sample)
    return (
        np.stack(features).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        sample_infos,
    )


def _samples_to_sequence_arrays(
    samples: list[dict[str, object]],
    label_to_index: dict[str, int],
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    sequences = []
    targets = []
    sample_infos = []
    for sample in samples:
        trajectory = sample.get("trajectory", {})
        sequence_array = build_temporal_sequence_array(
            trajectory,
            sequence_length=sequence_length,
        )
        if sequence_array is None:
            continue
        sequences.append(sequence_array.astype(np.float32))
        targets.append(label_to_index[sample["primary_label"]])
        sample_infos.append(sample)

    if not sequences:
        raise ValueError("No valid temporal samples could be constructed.")
    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        sample_infos,
    )


def _samples_to_fusion_arrays(
    samples: list[dict[str, object]],
    label_to_index: dict[str, int],
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    sequences = []
    static_features = []
    targets = []
    sample_infos = []
    for sample in samples:
        trajectory = sample.get("trajectory", {})
        sequence_array = build_temporal_sequence_array(
            trajectory,
            sequence_length=sequence_length,
        )
        if sequence_array is None:
            continue

        sample_features = sample.get("features", {})
        missing_feature_names = [
            feature_name
            for feature_name in FEATURE_NAMES
            if feature_name != "frame_count" and feature_name not in sample_features
        ]
        if missing_feature_names:
            derived_features = features_from_trajectory_payload(
                trajectory,
                high_speed_threshold=15.57,
            )
            if derived_features is None:
                continue
        else:
            derived_features = None

        feature_dict = {}
        for feature_name in FEATURE_NAMES:
            if feature_name == "frame_count":
                feature_dict[feature_name] = float(sample["frame_count"])
            elif feature_name in sample_features:
                feature_dict[feature_name] = float(sample_features[feature_name])
            else:
                feature_dict[feature_name] = float(derived_features[feature_name])

        sequences.append(sequence_array.astype(np.float32))
        static_features.append(vectorize_feature_dict(feature_dict))
        targets.append(label_to_index[sample["primary_label"]])
        sample_infos.append(sample)

    if not sequences:
        raise ValueError("No valid temporal-fusion samples could be constructed.")
    return (
        np.stack(sequences).astype(np.float32),
        np.stack(static_features).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        sample_infos,
    )


def _compute_macro_f1(targets: np.ndarray, predictions: np.ndarray, num_classes: int) -> tuple[float, dict[str, float]]:
    per_class = {}
    f1_values = []
    for class_index in range(num_classes):
        tp = int(np.sum((predictions == class_index) & (targets == class_index)))
        fp = int(np.sum((predictions == class_index) & (targets != class_index)))
        fn = int(np.sum((predictions != class_index) & (targets == class_index)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[str(class_index)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(targets == class_index)),
        }
        f1_values.append(f1)
    return float(np.mean(f1_values)), per_class


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - target_probs, self.gamma)
        loss = -focal_weight * target_log_probs
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets)
            loss = loss * alpha_weight
        return loss.mean()


def _build_loss(
    loss_type: str,
    *,
    class_counts: np.ndarray,
    device: torch.device,
    gamma: float,
    class_balance_beta: float,
) -> tuple[nn.Module, np.ndarray]:
    normalized_inverse = class_counts.sum() / np.maximum(class_counts * len(class_counts), 1.0)
    alpha_tensor = torch.tensor(normalized_inverse, dtype=torch.float32, device=device)

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=alpha_tensor), normalized_inverse

    if loss_type == "focal":
        return FocalLoss(alpha=alpha_tensor, gamma=gamma), normalized_inverse

    if loss_type == "cb_focal":
        beta = float(class_balance_beta)
        effective_num = 1.0 - np.power(beta, class_counts)
        cb_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        cb_weights = cb_weights / np.maximum(cb_weights.sum(), 1e-8) * len(class_counts)
        cb_tensor = torch.tensor(cb_weights.astype(np.float32), dtype=torch.float32, device=device)
        return FocalLoss(alpha=cb_tensor, gamma=gamma), cb_weights

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def train_behavior_classifier(config: BehaviorClassifierTrainConfig) -> dict[str, object]:
    cfg = config.resolved()
    _set_seed(cfg.seed)

    output_dir = cfg.output_dir / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = tuple(cfg.labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    samples = _load_samples(cfg.dataset_path, labels)
    requested_split_mode = str(cfg.val_split_mode).lower()
    model_type = str(cfg.model_type).lower()
    if requested_split_mode == "auto":
        split_mode = "sequence" if model_type in {"temporal", "temporal_fusion"} else "track"
    else:
        split_mode = requested_split_mode
    train_rows, val_rows, split_info = _stratified_group_split(
        samples,
        labels,
        cfg.val_ratio,
        cfg.seed,
        split_mode=split_mode,
    )

    if model_type == "temporal":
        train_x, train_y, train_infos = _samples_to_sequence_arrays(
            train_rows,
            label_to_index,
            sequence_length=cfg.sequence_length,
        )
        val_x, val_y, _ = _samples_to_sequence_arrays(
            val_rows,
            label_to_index,
            sequence_length=cfg.sequence_length,
        )
        valid_index = len(TEMPORAL_STEP_FEATURE_NAMES) - 1
        valid_mask = train_x[:, :, valid_index] > 0.5
        mean = np.zeros(train_x.shape[-1], dtype=np.float32)
        std = np.ones(train_x.shape[-1], dtype=np.float32)
        for feature_index in range(train_x.shape[-1] - 1):
            feature_values = train_x[:, :, feature_index][valid_mask]
            if feature_values.size == 0:
                continue
            mean[feature_index] = float(feature_values.mean())
            feature_std = float(feature_values.std())
            std[feature_index] = feature_std if feature_std > 1e-6 else 1.0
        train_x[:, :, :-1] = (train_x[:, :, :-1] - mean[:-1]) / std[:-1]
        val_x[:, :, :-1] = (val_x[:, :, :-1] - mean[:-1]) / std[:-1]
        train_dataset = TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_x),
            torch.from_numpy(val_y),
        )
    elif model_type == "temporal_fusion":
        train_seq_x, train_static_x, train_y, train_infos = _samples_to_fusion_arrays(
            train_rows,
            label_to_index,
            sequence_length=cfg.sequence_length,
        )
        val_seq_x, val_static_x, val_y, _ = _samples_to_fusion_arrays(
            val_rows,
            label_to_index,
            sequence_length=cfg.sequence_length,
        )

        valid_index = len(TEMPORAL_STEP_FEATURE_NAMES) - 1
        valid_mask = train_seq_x[:, :, valid_index] > 0.5
        seq_mean = np.zeros(train_seq_x.shape[-1], dtype=np.float32)
        seq_std = np.ones(train_seq_x.shape[-1], dtype=np.float32)
        for feature_index in range(train_seq_x.shape[-1] - 1):
            feature_values = train_seq_x[:, :, feature_index][valid_mask]
            if feature_values.size == 0:
                continue
            seq_mean[feature_index] = float(feature_values.mean())
            feature_std = float(feature_values.std())
            seq_std[feature_index] = feature_std if feature_std > 1e-6 else 1.0
        train_seq_x[:, :, :-1] = (train_seq_x[:, :, :-1] - seq_mean[:-1]) / seq_std[:-1]
        val_seq_x[:, :, :-1] = (val_seq_x[:, :, :-1] - seq_mean[:-1]) / seq_std[:-1]

        mean = train_static_x.mean(axis=0)
        std = train_static_x.std(axis=0)
        std = np.where(std <= 1e-6, 1.0, std)
        train_static_x = (train_static_x - mean) / std
        val_static_x = (val_static_x - mean) / std

        train_dataset = TensorDataset(
            torch.from_numpy(train_seq_x),
            torch.from_numpy(train_static_x),
            torch.from_numpy(train_y),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_seq_x),
            torch.from_numpy(val_static_x),
            torch.from_numpy(val_y),
        )
    else:
        train_x, train_y, train_infos = _samples_to_arrays(train_rows, label_to_index)
        val_x, val_y, _ = _samples_to_arrays(val_rows, label_to_index)

        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std = np.where(std <= 1e-6, 1.0, std)
        train_x = (train_x - mean) / std
        val_x = (val_x - mean) / std

        train_dataset = TensorDataset(
            torch.from_numpy(train_x),
            torch.from_numpy(train_y),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_x),
            torch.from_numpy(val_y),
        )

    class_counts = np.asarray([max(1, np.sum(train_y == idx)) for idx in range(len(labels))], dtype=np.float32)
    sample_weights = np.asarray([1.0 / class_counts[target] for target in train_y], dtype=np.float32)
    hard_negative_multipliers = []
    for sample, target in zip(train_infos, train_y):
        multiplier = 1.0
        label = labels[int(target)]
        if label == "normal":
            hard_negative_score = float(sample.get("features", {}).get("hard_negative_score", 0.0) or 0.0)
            if hard_negative_score > 0.0:
                # Boost hard negatives so the model sees false-positive normals more often,
                # but cap the multiplier to avoid collapsing recall.
                multiplier = min(3.0, 1.0 + np.log1p(hard_negative_score) / 4.0)
        hard_negative_multipliers.append(multiplier)
    sample_weights = sample_weights * np.asarray(hard_negative_multipliers, dtype=np.float32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(cfg.batch_size, len(train_dataset)),
        sampler=WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=max(len(train_dataset), len(labels) * 24),
            replacement=True,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(cfg.batch_size, len(val_dataset)),
        shuffle=False,
    )

    device = _resolve_torch_device(cfg.device)
    if model_type == "temporal":
        model = TemporalBehaviorCNN(
            input_dim=len(TEMPORAL_STEP_FEATURE_NAMES),
            output_dim=len(labels),
            channels=(32, 64),
            dropout=cfg.dropout,
        ).to(device)
    elif model_type == "temporal_fusion":
        model = TemporalFeatureFusion(
            sequence_input_dim=len(TEMPORAL_STEP_FEATURE_NAMES),
            static_input_dim=len(FEATURE_NAMES),
            output_dim=len(labels),
            channels=(32, 64),
            static_hidden_dim=32,
            dropout=cfg.dropout,
        ).to(device)
    else:
        model = BehaviorMLP(
            input_dim=len(FEATURE_NAMES),
            output_dim=len(labels),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(device)

    criterion, loss_weights = _build_loss(
        str(cfg.loss_type).lower(),
        class_counts=class_counts,
        device=device,
        gamma=cfg.focal_gamma,
        class_balance_beta=cfg.class_balance_beta,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = []
    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    def _forward_batch(batch):
        if model_type == "temporal_fusion":
            batch_seq_x, batch_static_x, batch_y = batch
            return (
                model(batch_seq_x.to(device), batch_static_x.to(device)),
                batch_y.to(device),
            )
        batch_x, batch_y = batch
        return (
            model(batch_x.to(device)),
            batch_y.to(device),
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_targets = []
        train_predictions = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, batch_y = _forward_batch(batch)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * batch_y.size(0)
            train_targets.extend(batch_y.detach().cpu().numpy().tolist())
            train_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

        model.eval()
        val_loss_total = 0.0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                logits, batch_y = _forward_batch(batch)
                loss = criterion(logits, batch_y)

                val_loss_total += loss.item() * batch_y.size(0)
                val_targets.extend(batch_y.detach().cpu().numpy().tolist())
                val_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

        train_targets_np = np.asarray(train_targets, dtype=np.int64)
        train_predictions_np = np.asarray(train_predictions, dtype=np.int64)
        val_targets_np = np.asarray(val_targets, dtype=np.int64)
        val_predictions_np = np.asarray(val_predictions, dtype=np.int64)

        train_macro_f1, _ = _compute_macro_f1(train_targets_np, train_predictions_np, len(labels))
        val_macro_f1, _ = _compute_macro_f1(val_targets_np, val_predictions_np, len(labels))

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss_total / max(len(train_dataset), 1), 6),
            "val_loss": round(val_loss_total / max(len(val_dataset), 1), 6),
            "train_macro_f1": round(train_macro_f1, 4),
            "val_macro_f1": round(val_macro_f1, 4),
        }
        history.append(epoch_metrics)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "metrics": epoch_metrics,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Behavior classifier training did not produce a valid checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        if model_type == "temporal_fusion":
            train_logits = model(
                torch.from_numpy(train_seq_x).to(device),
                torch.from_numpy(train_static_x).to(device),
            )
            val_logits = model(
                torch.from_numpy(val_seq_x).to(device),
                torch.from_numpy(val_static_x).to(device),
            )
        else:
            train_logits = model(torch.from_numpy(train_x).to(device))
            val_logits = model(torch.from_numpy(val_x).to(device))

    final_train_predictions = torch.argmax(train_logits, dim=1).cpu().numpy()
    final_val_predictions = torch.argmax(val_logits, dim=1).cpu().numpy()
    train_macro_f1, train_per_class = _compute_macro_f1(train_y, final_train_predictions, len(labels))
    val_macro_f1, val_per_class = _compute_macro_f1(val_y, final_val_predictions, len(labels))

    checkpoint = {
        "model_type": model_type,
        "labels": labels,
        "high_speed_threshold": 15.57,
        "min_frames": 24,
        "score_threshold": 0.55,
        "model_state_dict": model.state_dict(),
    }
    if model_type == "temporal":
        checkpoint.update(
            {
                "sequence_length": int(cfg.sequence_length),
                "temporal_feature_names": TEMPORAL_STEP_FEATURE_NAMES,
                "sequence_normalization": {
                    "mean": mean.astype(np.float32).tolist(),
                    "std": std.astype(np.float32).tolist(),
                },
                "model_config": {
                    "channels": (32, 64),
                    "dropout": cfg.dropout,
                },
                "feature_names": FEATURE_NAMES,
                "normalization": {
                    "mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32).tolist(),
                    "std": np.ones(len(FEATURE_NAMES), dtype=np.float32).tolist(),
                },
            }
        )
    elif model_type == "temporal_fusion":
        checkpoint.update(
            {
                "sequence_length": int(cfg.sequence_length),
                "temporal_feature_names": TEMPORAL_STEP_FEATURE_NAMES,
                "sequence_normalization": {
                    "mean": seq_mean.astype(np.float32).tolist(),
                    "std": seq_std.astype(np.float32).tolist(),
                },
                "feature_names": FEATURE_NAMES,
                "normalization": {
                    "mean": mean.astype(np.float32).tolist(),
                    "std": std.astype(np.float32).tolist(),
                },
                "model_config": {
                    "channels": (32, 64),
                    "static_hidden_dim": 32,
                    "dropout": cfg.dropout,
                },
            }
        )
    else:
        checkpoint.update(
            {
                "feature_names": FEATURE_NAMES,
                "normalization": {
                    "mean": mean.astype(np.float32).tolist(),
                    "std": std.astype(np.float32).tolist(),
                },
                "model_config": {
                    "hidden_dims": cfg.hidden_dims,
                    "dropout": cfg.dropout,
                },
            }
        )

    checkpoint_path = output_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "config": {
            "dataset_path": str(cfg.dataset_path),
            "output_dir": str(output_dir),
            "device": str(device),
            "model_type": model_type,
            "sequence_length": int(cfg.sequence_length),
            "labels": labels,
            "val_ratio": cfg.val_ratio,
            "val_split_mode_requested": requested_split_mode,
            "val_split_mode": split_mode,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "loss_type": cfg.loss_type,
            "focal_gamma": cfg.focal_gamma,
            "class_balance_beta": cfg.class_balance_beta,
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "patience": cfg.patience,
            "loss_weights": [round(float(value), 6) for value in loss_weights],
        },
        "data": {
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "train_class_counts": {label: int(np.sum(train_y == idx)) for label, idx in label_to_index.items()},
            "val_class_counts": {label: int(np.sum(val_y == idx)) for label, idx in label_to_index.items()},
            "split": split_info,
        },
        "best_epoch": best_epoch,
        "train_metrics": {
            "macro_f1": round(train_macro_f1, 4),
            "per_class": {
                labels[int(class_index)]: values
                for class_index, values in train_per_class.items()
            },
        },
        "val_metrics": {
            "macro_f1": round(val_macro_f1, 4),
            "per_class": {
                labels[int(class_index)]: values
                for class_index, values in val_per_class.items()
            },
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "history.json"),
            "metrics": str(output_dir / "metrics.json"),
        },
    }

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics

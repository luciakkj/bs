from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "frame_count",
    "avg_speed",
    "speed_std",
    "max_speed",
    "p90_speed",
    "high_speed_ratio",
    "stationary_ratio",
    "path_length",
    "displacement",
    "movement_extent",
    "centroid_radius",
    "straightness",
    "direction_change_rate",
    "mean_turn_angle",
    "revisit_ratio",
    "unique_cell_ratio",
    "max_cell_occupancy_ratio",
)
TEMPORAL_STEP_FEATURE_NAMES = (
    "dx",
    "dy",
    "speed",
    "dist_to_start",
    "dist_to_centroid",
    "turn_angle",
    "valid_flag",
)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _angle_delta(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
    angle_a = math.atan2(vector_a[1], vector_a[0])
    angle_b = math.atan2(vector_b[1], vector_b[0])
    delta = angle_b - angle_a
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return abs(delta)


def build_feature_vector(
    frame_count: int,
    avg_speed: float,
    speed_std: float,
    max_speed: float,
    p90_speed: float,
    high_speed_ratio: float,
    stationary_ratio: float,
    path_length: float,
    displacement: float,
    movement_extent: float,
    centroid_radius: float,
    straightness: float,
    direction_change_rate: float,
    mean_turn_angle: float,
    revisit_ratio: float,
    unique_cell_ratio: float,
    max_cell_occupancy_ratio: float,
) -> dict[str, float]:
    return {
        "frame_count": float(frame_count),
        "avg_speed": float(avg_speed),
        "speed_std": float(speed_std),
        "max_speed": float(max_speed),
        "p90_speed": float(p90_speed),
        "high_speed_ratio": float(high_speed_ratio),
        "stationary_ratio": float(stationary_ratio),
        "path_length": float(path_length),
        "displacement": float(displacement),
        "movement_extent": float(movement_extent),
        "centroid_radius": float(centroid_radius),
        "straightness": float(straightness),
        "direction_change_rate": float(direction_change_rate),
        "mean_turn_angle": float(mean_turn_angle),
        "revisit_ratio": float(revisit_ratio),
        "unique_cell_ratio": float(unique_cell_ratio),
        "max_cell_occupancy_ratio": float(max_cell_occupancy_ratio),
    }


def features_from_trajectory_payload(
    trajectory_payload: dict[str, object],
    *,
    high_speed_threshold: float = 15.57,
    low_speed_threshold: float = 2.0,
) -> dict[str, float] | None:
    centers = trajectory_payload.get("centers", [])
    if not centers:
        return None

    points = [(float(item[0]), float(item[1])) for item in centers]
    raw_speeds = trajectory_payload.get("speeds", [])
    speeds = [float(value) for value in raw_speeds]
    if not speeds and len(points) >= 2:
        speeds = [_distance(points[idx - 1], points[idx]) for idx in range(1, len(points))]

    motion_vectors = [
        (points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
        for idx in range(1, len(points))
    ]
    turn_angles = []
    for idx in range(1, len(motion_vectors)):
        prev_vector = motion_vectors[idx - 1]
        curr_vector = motion_vectors[idx]
        if _distance((0.0, 0.0), prev_vector) <= 1e-6 or _distance((0.0, 0.0), curr_vector) <= 1e-6:
            continue
        turn_angles.append(_angle_delta(prev_vector, curr_vector))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    path_length = sum(_distance(points[idx - 1], points[idx]) for idx in range(1, len(points)))
    displacement = _distance(points[0], points[-1]) if len(points) >= 2 else 0.0
    movement_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    centroid_radius = max((_distance(point, centroid) for point in points), default=0.0)
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    speed_std = float(np.std(speeds)) if speeds else 0.0
    max_speed = max(speeds, default=0.0)
    p90_speed = float(np.percentile(speeds, 90)) if speeds else 0.0
    high_speed_ratio = _safe_divide(
        sum(1 for value in speeds if value >= high_speed_threshold),
        len(speeds),
    )
    stationary_ratio = _safe_divide(
        sum(1 for value in speeds if value <= low_speed_threshold),
        len(speeds),
    )
    straightness = _safe_divide(displacement, path_length)
    direction_change_rate = _safe_divide(
        sum(1 for angle in turn_angles if angle >= (math.pi / 4.0)),
        len(turn_angles),
    )
    mean_turn_angle = _safe_divide(sum(turn_angles), len(turn_angles))

    # Quantize trajectory points into coarse cells to measure lingering/revisiting behavior.
    cell_size = max(12.0, min(32.0, movement_extent / 4.0 if movement_extent > 0 else 12.0))
    quantized_points = [
        (int(round(point[0] / cell_size)), int(round(point[1] / cell_size)))
        for point in points
    ]
    unique_cells = len(set(quantized_points))
    cell_counts: dict[tuple[int, int], int] = {}
    revisit_events = 0
    for cell in quantized_points:
        previous_count = cell_counts.get(cell, 0)
        if previous_count > 0:
            revisit_events += 1
        cell_counts[cell] = previous_count + 1
    unique_cell_ratio = _safe_divide(unique_cells, len(quantized_points))
    max_cell_occupancy_ratio = _safe_divide(max(cell_counts.values(), default=0), len(quantized_points))
    revisit_ratio = _safe_divide(revisit_events, max(len(quantized_points) - 1, 1))

    return build_feature_vector(
        frame_count=len(points),
        avg_speed=avg_speed,
        speed_std=speed_std,
        max_speed=max_speed,
        p90_speed=p90_speed,
        high_speed_ratio=high_speed_ratio,
        stationary_ratio=stationary_ratio,
        path_length=path_length,
        displacement=displacement,
        movement_extent=movement_extent,
        centroid_radius=centroid_radius,
        straightness=straightness,
        direction_change_rate=direction_change_rate,
        mean_turn_angle=mean_turn_angle,
        revisit_ratio=revisit_ratio,
        unique_cell_ratio=unique_cell_ratio,
        max_cell_occupancy_ratio=max_cell_occupancy_ratio,
    )


def features_from_track_info(track_info: dict[str, object], high_speed_threshold: float = 15.57) -> dict[str, float] | None:
    trajectory = track_info.get("trajectory", [])
    if not trajectory:
        return None

    return features_from_trajectory_payload(
        {
            "centers": trajectory,
            "speeds": track_info.get("speed_history", []),
        },
        high_speed_threshold=high_speed_threshold,
    )


def vectorize_feature_dict(feature_dict: dict[str, float], feature_names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    return np.asarray([float(feature_dict[name]) for name in feature_names], dtype=np.float32)


def build_temporal_sequence_array(
    trajectory_payload: dict[str, object],
    *,
    sequence_length: int = 72,
) -> np.ndarray | None:
    centers = trajectory_payload.get("centers", [])
    if not centers:
        return None

    points = np.asarray([(float(item[0]), float(item[1])) for item in centers], dtype=np.float32)
    frame_count = int(points.shape[0])
    if frame_count <= 0:
        return None

    sequence_length = max(8, int(sequence_length))
    sampled_count = min(frame_count, sequence_length)
    if frame_count > sequence_length:
        sampled_indices = np.linspace(0, frame_count - 1, num=sequence_length)
        sampled_indices = np.clip(np.round(sampled_indices).astype(np.int32), 0, frame_count - 1)
        sampled_points = points[sampled_indices]
        step_gaps = np.diff(sampled_indices, prepend=sampled_indices[0]).astype(np.float32)
    else:
        sampled_indices = np.arange(frame_count, dtype=np.int32)
        sampled_points = points
        step_gaps = np.ones(frame_count, dtype=np.float32)

    step_gaps[step_gaps <= 0] = 1.0
    deltas = np.zeros_like(sampled_points, dtype=np.float32)
    if sampled_points.shape[0] >= 2:
        deltas[1:] = sampled_points[1:] - sampled_points[:-1]
        deltas[1:] = deltas[1:] / step_gaps[1:, None]

    speeds = np.linalg.norm(deltas, axis=1).astype(np.float32)
    start_point = sampled_points[0]
    centroid = np.mean(sampled_points, axis=0)
    dist_to_start = np.linalg.norm(sampled_points - start_point, axis=1).astype(np.float32)
    dist_to_centroid = np.linalg.norm(sampled_points - centroid, axis=1).astype(np.float32)
    turn_angles = np.zeros(sampled_points.shape[0], dtype=np.float32)
    if sampled_points.shape[0] >= 3:
        prev_vectors = deltas[1:-1]
        curr_vectors = deltas[2:]
        for idx, (vector_a, vector_b) in enumerate(zip(prev_vectors, curr_vectors), start=2):
            if np.linalg.norm(vector_a) <= 1e-6 or np.linalg.norm(vector_b) <= 1e-6:
                continue
            turn_angles[idx] = float(_angle_delta((float(vector_a[0]), float(vector_a[1])), (float(vector_b[0]), float(vector_b[1]))))

    sampled_features = np.stack(
        [
            deltas[:, 0],
            deltas[:, 1],
            speeds,
            dist_to_start,
            dist_to_centroid,
            turn_angles / math.pi,
            np.ones(sampled_points.shape[0], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if sampled_count == sequence_length:
        return sampled_features

    padded = np.zeros((sequence_length, len(TEMPORAL_STEP_FEATURE_NAMES)), dtype=np.float32)
    padded[-sampled_count:] = sampled_features
    return padded


class BehaviorMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (32, 16), dropout: float = 0.1):
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TemporalBehaviorCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels: tuple[int, ...] = (32, 64),
        dropout: float = 0.15,
    ):
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_channels = input_dim
        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.encoder = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # [batch, steps, features] -> [batch, features, steps]
        encoded = self.encoder(inputs.transpose(1, 2))
        return self.classifier(encoded)


class TemporalFeatureFusion(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        static_input_dim: int,
        output_dim: int,
        channels: tuple[int, ...] = (32, 64),
        static_hidden_dim: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_channels = sequence_input_dim
        for out_channels in channels:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.encoder = nn.Sequential(*conv_layers)
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels + static_hidden_dim, max(32, static_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(32, static_hidden_dim), output_dim),
        )

    def forward(self, sequence_inputs: torch.Tensor, static_inputs: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_pool(self.encoder(sequence_inputs.transpose(1, 2)))
        static_features = self.static_encoder(static_inputs)
        fused = torch.cat([temporal_features, static_features], dim=1)
        return self.classifier(fused)


class TrajectoryBehaviorClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | None = None,
        min_frames_override: int | None = None,
    ):
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        self.model_type = str(checkpoint.get("model_type", "mlp")).lower()
        self.labels = tuple(checkpoint["labels"])
        self.high_speed_threshold = float(checkpoint.get("high_speed_threshold", 15.57))
        checkpoint_min_frames = int(checkpoint.get("min_frames", 24))
        if min_frames_override is None:
            self.min_frames = checkpoint_min_frames
        else:
            self.min_frames = max(8, int(min_frames_override))
        self.score_threshold = float(checkpoint.get("score_threshold", 0.55))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.model_type in {"temporal", "temporal_fusion"}:
            self.sequence_length = int(checkpoint.get("sequence_length", 72))
            self.temporal_feature_names = tuple(checkpoint.get("temporal_feature_names", TEMPORAL_STEP_FEATURE_NAMES))
            sequence_norm = checkpoint.get("sequence_normalization", {})
            self.sequence_mean = np.asarray(sequence_norm.get("mean", [0.0] * len(self.temporal_feature_names)), dtype=np.float32)
            self.sequence_std = np.asarray(sequence_norm.get("std", [1.0] * len(self.temporal_feature_names)), dtype=np.float32)
            channels = tuple(int(value) for value in checkpoint["model_config"].get("channels", (32, 64)))
            dropout = float(checkpoint["model_config"]["dropout"])
            self.feature_names = tuple(checkpoint.get("feature_names", FEATURE_NAMES))
            static_norm = checkpoint.get("normalization", {})
            self.mean = np.asarray(static_norm.get("mean", [0.0] * len(self.feature_names)), dtype=np.float32)
            self.std = np.asarray(static_norm.get("std", [1.0] * len(self.feature_names)), dtype=np.float32)
            if self.model_type == "temporal_fusion":
                static_hidden_dim = int(checkpoint["model_config"].get("static_hidden_dim", 32))
                self.model = TemporalFeatureFusion(
                    sequence_input_dim=len(self.temporal_feature_names),
                    static_input_dim=len(self.feature_names),
                    output_dim=len(self.labels),
                    channels=channels,
                    static_hidden_dim=static_hidden_dim,
                    dropout=dropout,
                )
            else:
                self.model = TemporalBehaviorCNN(
                    input_dim=len(self.temporal_feature_names),
                    output_dim=len(self.labels),
                    channels=channels,
                    dropout=dropout,
                )
        else:
            self.feature_names = tuple(checkpoint["feature_names"])
            self.mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
            self.std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
            hidden_dims = tuple(int(value) for value in checkpoint["model_config"]["hidden_dims"])
            dropout = float(checkpoint["model_config"]["dropout"])
            self.model = BehaviorMLP(
                input_dim=len(self.feature_names),
                output_dim=len(self.labels),
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _aggregate_window_scores(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        return float(max(scores))

    def _candidate_window_sizes(self, frame_count: int, label_name: str | None = None) -> list[int]:
        base_sizes = {
            self.min_frames,
            24,
            36,
            48,
            72,
            96,
            120,
            168,
            frame_count,
        }
        if label_name == "running":
            base_sizes.update({24, 30, 36, 48, 60, 72})
        elif label_name == "loitering":
            base_sizes.update({72, 84, 96, 120, 144, 168, frame_count})

        sizes = sorted(
            int(size)
            for size in base_sizes
            if self.min_frames <= int(size) <= frame_count
        )

        if label_name == "running":
            running_sizes = [size for size in sizes if size <= 72]
            return running_sizes or sizes
        if label_name == "loitering":
            loitering_sizes = [size for size in sizes if size >= max(self.min_frames, 72)]
            return loitering_sizes or sizes
        return sizes

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        std = np.where(self.std <= 1e-6, 1.0, self.std)
        return (values - self.mean) / std

    def _normalize_temporal_sequence(self, sequence_array: np.ndarray) -> np.ndarray:
        mean = self.sequence_mean.reshape(1, -1)
        std = np.where(self.sequence_std <= 1e-6, 1.0, self.sequence_std).reshape(1, -1)
        normalized = sequence_array.copy()
        if normalized.shape[1] >= 1:
            valid_feature_index = len(self.temporal_feature_names) - 1
            if valid_feature_index > 0:
                normalized[:, :valid_feature_index] = (
                    normalized[:, :valid_feature_index] - mean[:, :valid_feature_index]
                ) / std[:, :valid_feature_index]
            normalized[:, valid_feature_index] = sequence_array[:, valid_feature_index]
        return normalized

    def _predict_probabilities_batch(
        self,
        payloads_by_window: list[tuple[int, dict[str, object]]],
        feature_cache: dict[int, dict[str, float]],
    ) -> dict[int, dict[str, float]]:
        valid_items: list[tuple[int, dict[str, object], dict[str, float]]] = []
        for window_size, payload in payloads_by_window:
            feature_dict = feature_cache.get(int(window_size))
            if feature_dict is None:
                continue
            valid_items.append((int(window_size), payload, feature_dict))

        if not valid_items:
            return {}

        probs_cache: dict[int, dict[str, float]] = {}
        with torch.no_grad():
            if self.model_type not in {"temporal", "temporal_fusion"}:
                feature_batch = np.stack(
                    [
                        self._normalize(vectorize_feature_dict(feature_dict, self.feature_names))
                        for _, _, feature_dict in valid_items
                    ],
                    axis=0,
                ).astype(np.float32)
                logits = self.model(torch.from_numpy(feature_batch).to(self.device))
            else:
                sequence_batch = np.stack(
                    [
                        self._normalize_temporal_sequence(
                            build_temporal_sequence_array(
                                payload,
                                sequence_length=self.sequence_length,
                            )
                        )
                        for _, payload, _ in valid_items
                    ],
                    axis=0,
                ).astype(np.float32)
                sequence_inputs = torch.from_numpy(sequence_batch).to(self.device)
                if self.model_type == "temporal_fusion":
                    static_std = np.where(self.std <= 1e-6, 1.0, self.std)
                    static_batch = np.stack(
                        [
                            ((vectorize_feature_dict(feature_dict, self.feature_names) - self.mean) / static_std).astype(np.float32)
                            for _, _, feature_dict in valid_items
                        ],
                        axis=0,
                    )
                    static_inputs = torch.from_numpy(static_batch).to(self.device)
                    logits = self.model(sequence_inputs, static_inputs)
                else:
                    logits = self.model(sequence_inputs)

            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        for row_index, (window_size, _, _) in enumerate(valid_items):
            probs_cache[window_size] = {
                self.labels[label_index]: float(probabilities[row_index, label_index])
                for label_index in range(len(self.labels))
            }
        return probs_cache

    def predict_features(self, feature_dict: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        values = vectorize_feature_dict(feature_dict, self.feature_names)
        normalized = self._normalize(values)
        inputs = torch.from_numpy(normalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_index = int(np.argmax(probabilities))
        label = self.labels[best_index]
        score = float(probabilities[best_index])
        probs = {
            self.labels[idx]: float(probabilities[idx])
            for idx in range(len(self.labels))
        }
        return label, score, probs

    def predict_payload(self, trajectory_payload: dict[str, object], feature_dict: dict[str, float] | None = None) -> tuple[str | None, float, dict[str, float]]:
        if self.model_type not in {"temporal", "temporal_fusion"}:
            if feature_dict is None:
                feature_dict = features_from_trajectory_payload(
                    trajectory_payload,
                    high_speed_threshold=self.high_speed_threshold,
                )
            if feature_dict is None:
                return None, 0.0, {}
            return self.predict_features(feature_dict)

        sequence_array = build_temporal_sequence_array(
            trajectory_payload,
            sequence_length=self.sequence_length,
        )
        if sequence_array is None:
            return None, 0.0, {}
        if feature_dict is None:
            feature_dict = features_from_trajectory_payload(
                trajectory_payload,
                high_speed_threshold=self.high_speed_threshold,
            )
        if self.model_type == "temporal_fusion" and feature_dict is None:
            return None, 0.0, {}
        normalized = self._normalize_temporal_sequence(sequence_array)

        inputs = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_type == "temporal_fusion":
                static_values = vectorize_feature_dict(feature_dict, self.feature_names)
                static_std = np.where(self.std <= 1e-6, 1.0, self.std)
                static_inputs = torch.from_numpy(((static_values - self.mean) / static_std).astype(np.float32)).unsqueeze(0).to(self.device)
                logits = self.model(inputs, static_inputs)
            else:
                logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        best_index = int(np.argmax(probabilities))
        label = self.labels[best_index]
        score = float(probabilities[best_index])
        probs = {
            self.labels[idx]: float(probabilities[idx])
            for idx in range(len(self.labels))
        }
        return label, score, probs

    def _iter_window_payloads(
        self,
        track_info: dict[str, object],
        label_name: str | None = None,
    ) -> list[tuple[int, dict[str, object]]]:
        trajectory = list(track_info.get("trajectory", []))
        speed_history = list(track_info.get("speed_history", []))
        frame_count = len(trajectory)
        if frame_count < self.min_frames:
            return []

        candidate_sizes = self._candidate_window_sizes(frame_count, label_name=label_name)
        window_payloads: list[tuple[int, dict[str, object]]] = []
        for window_size in candidate_sizes:
            window_payloads.append(
                (
                    int(window_size),
                    {
                        "centers": trajectory[-window_size:],
                        "speeds": speed_history[-window_size:],
                    },
                )
            )
        return window_payloads

    def predict_track_info(
        self,
        track_info: dict[str, object],
        target_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> tuple[str | None, float, dict[str, float], dict[str, object] | None]:
        labels_to_score = [label for label in self.labels if target_labels is None or label in set(target_labels)]
        if not labels_to_score:
            feature_dict = features_from_track_info(
                track_info,
                high_speed_threshold=self.high_speed_threshold,
            )
            return None, 0.0, {}, feature_dict

        all_window_payloads_map: dict[int, dict[str, object]] = {}
        for label_name in labels_to_score:
            for window_size, payload in self._iter_window_payloads(track_info, label_name=label_name):
                all_window_payloads_map[int(window_size)] = payload
        all_window_payloads = sorted(all_window_payloads_map.items(), key=lambda item: item[0])
        window_payloads = all_window_payloads
        if not window_payloads:
            feature_dict = features_from_track_info(
                track_info,
                high_speed_threshold=self.high_speed_threshold,
            )
            return None, 0.0, {}, feature_dict

        feature_cache: dict[int, dict[str, float]] = {}
        for window_size, payload in all_window_payloads:
            feature_dict = features_from_trajectory_payload(
                payload,
                high_speed_threshold=self.high_speed_threshold,
            )
            if feature_dict is None or feature_dict["frame_count"] < self.min_frames:
                continue
            feature_cache[int(window_size)] = feature_dict
        probs_cache = self._predict_probabilities_batch(all_window_payloads, feature_cache)

        best_per_label: dict[str, dict[str, object]] = {
            label_name: {
                "score": 0.0,
                "raw_best_score": 0.0,
                "feature_dict": None,
                "window_size": 0,
                "window_scores": [],
            }
            for label_name in labels_to_score
        }

        for label_name in labels_to_score:
            label_windows = self._iter_window_payloads(track_info, label_name=label_name)
            for window_size, _ in label_windows:
                feature_dict = feature_cache.get(int(window_size))
                probs = probs_cache.get(int(window_size))
                if feature_dict is None or probs is None:
                    continue
                label_score = float(probs.get(label_name, 0.0))
                best_per_label[label_name]["window_scores"].append(label_score)
                if label_score > float(best_per_label[label_name]["raw_best_score"]):
                    best_per_label[label_name] = {
                        "score": label_score,
                        "raw_best_score": label_score,
                        "feature_dict": feature_dict,
                        "window_size": int(window_size),
                        "window_scores": list(best_per_label[label_name].get("window_scores", [])),
                    }

            best_per_label[label_name]["score"] = self._aggregate_window_scores(
                list(best_per_label[label_name].get("window_scores", []))
            )

        probs = {
            label_name: float(values["score"])
            for label_name, values in best_per_label.items()
        }
        best_label = max(probs, key=probs.get, default=None)
        best_score = float(probs.get(best_label, 0.0)) if best_label is not None else 0.0
        best_feature_dict = best_per_label.get(best_label, {}).get("feature_dict") if best_label is not None else None
        best_window_size = int(best_per_label.get(best_label, {}).get("window_size", 0)) if best_label is not None else 0

        return best_label, best_score, probs, {
            "best_label": best_label,
            "best_score": best_score,
            "best_feature_dict": best_feature_dict,
            "best_window_size": best_window_size,
            "window_count": len(window_payloads),
            "per_label": best_per_label,
        }

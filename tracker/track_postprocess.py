from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment


@dataclass(slots=True)
class Tracklet:
    track_id: int
    rows: list[list[float]]

    @property
    def frames(self) -> list[int]:
        return [int(row[0]) for row in self.rows]

    @property
    def start_frame(self) -> int:
        return int(self.rows[0][0])

    @property
    def end_frame(self) -> int:
        return int(self.rows[-1][0])

    @property
    def length(self) -> int:
        return len(self.rows)

    @property
    def first_box(self) -> np.ndarray:
        return np.asarray(self.rows[0][2:6], dtype=np.float32)

    @property
    def last_box(self) -> np.ndarray:
        return np.asarray(self.rows[-1][2:6], dtype=np.float32)

    def center_velocity(self) -> np.ndarray:
        if len(self.rows) < 2:
            return np.zeros(2, dtype=np.float32)
        prev = np.asarray(self.rows[-2][2:6], dtype=np.float32)
        curr = np.asarray(self.rows[-1][2:6], dtype=np.float32)
        prev_center = np.array([prev[0] + prev[2] * 0.5, prev[1] + prev[3] * 0.5], dtype=np.float32)
        curr_center = np.array([curr[0] + curr[2] * 0.5, curr[1] + curr[3] * 0.5], dtype=np.float32)
        return curr_center - prev_center


def _rows_to_tracklets(rows: list[list[float]]) -> list[Tracklet]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row[1])].append(list(row))
    tracklets = []
    for track_id, track_rows in grouped.items():
        ordered = sorted(track_rows, key=lambda item: (int(item[0]), int(item[1])))
        tracklets.append(Tracklet(track_id=track_id, rows=ordered))
    return sorted(tracklets, key=lambda item: item.track_id)


def _center_from_box(box: np.ndarray) -> np.ndarray:
    return np.array([box[0] + box[2] * 0.5, box[1] + box[3] * 0.5], dtype=np.float32)


def _aflink_cost(
    left: Tracklet,
    right: Tracklet,
    max_gap: int,
    max_center_dist: float,
    max_scale_ratio: float,
) -> float:
    gap = right.start_frame - left.end_frame
    if gap <= 0 or gap > max_gap:
        return np.inf
    if left.length < 2 or right.length < 2:
        return np.inf

    left_last = left.last_box
    right_first = right.first_box
    predicted_center = _center_from_box(left_last) + left.center_velocity() * float(gap)
    target_center = _center_from_box(right_first)
    center_dist = float(np.linalg.norm(predicted_center - target_center))
    if center_dist > max_center_dist:
        return np.inf

    left_scale = max(1.0, float(left_last[3]))
    right_scale = max(1.0, float(right_first[3]))
    scale_ratio = max(left_scale, right_scale) / max(1.0, min(left_scale, right_scale))
    if scale_ratio > max_scale_ratio:
        return np.inf

    normalized_center = center_dist / max(left_scale, right_scale)
    gap_penalty = float(gap) / float(max_gap)
    scale_penalty = abs(np.log(scale_ratio))
    return normalized_center + 0.35 * gap_penalty + 0.25 * scale_penalty


def apply_aflink(
    rows: list[list[float]],
    enabled: bool = False,
    max_gap: int = 30,
    max_center_dist: float = 120.0,
    max_scale_ratio: float = 1.8,
    min_track_length: int = 3,
) -> list[list[float]]:
    if not enabled or not rows:
        return rows

    tracklets = [item for item in _rows_to_tracklets(rows) if item.length >= min_track_length]
    if len(tracklets) < 2:
        return rows

    cost_matrix = np.full((len(tracklets), len(tracklets)), np.inf, dtype=np.float32)
    for left_idx, left in enumerate(tracklets):
        for right_idx, right in enumerate(tracklets):
            if left_idx == right_idx:
                continue
            cost_matrix[left_idx, right_idx] = _aflink_cost(
                left,
                right,
                max_gap=max_gap,
                max_center_dist=max_center_dist,
                max_scale_ratio=max_scale_ratio,
            )

    finite_mask = np.isfinite(cost_matrix)
    if not finite_mask.any():
        return rows

    reduced = cost_matrix.copy()
    reduced[~finite_mask] = 1e6
    row_ind, col_ind = linear_sum_assignment(reduced)

    right_to_left: dict[int, int] = {}
    used_left = set()
    used_right = set()
    for row_idx, col_idx in zip(row_ind, col_ind):
        if not np.isfinite(cost_matrix[row_idx, col_idx]):
            continue
        if row_idx in used_left or col_idx in used_right:
            continue
        used_left.add(row_idx)
        used_right.add(col_idx)
        right_to_left[tracklets[col_idx].track_id] = tracklets[row_idx].track_id

    if not right_to_left:
        return rows

    remapped = []
    for row in rows:
        new_row = list(row)
        track_id = int(new_row[1])
        while track_id in right_to_left:
            track_id = right_to_left[track_id]
        new_row[1] = track_id
        remapped.append(new_row)
    return remapped


def apply_gsi(
    rows: list[list[float]],
    enabled: bool = False,
    max_gap: int = 20,
    sigma: float = 1.0,
) -> list[list[float]]:
    if not enabled or not rows:
        return rows

    processed_rows: list[list[float]] = []
    for tracklet in _rows_to_tracklets(rows):
        ordered = sorted(tracklet.rows, key=lambda item: int(item[0]))
        dense_rows: list[list[float]] = []
        for idx, row in enumerate(ordered):
            dense_rows.append(list(row))
            if idx == len(ordered) - 1:
                continue
            next_row = ordered[idx + 1]
            current_frame = int(row[0])
            next_frame = int(next_row[0])
            gap = next_frame - current_frame - 1
            if gap <= 0 or gap > max_gap:
                continue
            for offset in range(1, gap + 1):
                alpha = float(offset) / float(gap + 1)
                interp = list(row)
                interp[0] = current_frame + offset
                for box_idx in range(2, 6):
                    interp[box_idx] = (1.0 - alpha) * float(row[box_idx]) + alpha * float(next_row[box_idx])
                interp[6] = min(float(row[6]), float(next_row[6]))
                dense_rows.append(interp)

        dense_rows = sorted(dense_rows, key=lambda item: int(item[0]))
        if len(dense_rows) >= 3 and sigma > 0.0:
            boxes = np.asarray([row[2:6] for row in dense_rows], dtype=np.float32)
            for column in range(boxes.shape[1]):
                boxes[:, column] = gaussian_filter1d(boxes[:, column], sigma=sigma, mode="nearest")
            for row_idx, row in enumerate(dense_rows):
                for box_idx in range(4):
                    row[2 + box_idx] = float(boxes[row_idx, box_idx])

        processed_rows.extend(dense_rows)

    return sorted(processed_rows, key=lambda item: (int(item[0]), int(item[1])))

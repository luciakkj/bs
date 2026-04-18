from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import count

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.cmc import ECCMotionCompensator
from tracker.kalman_filter import KalmanFilterXYAH
from tracker.reid_encoder import ReIDFeatureExtractor


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def iou_distance(a_tracks, b_tracks):
    if not a_tracks or not b_tracks:
        return np.empty((len(a_tracks), len(b_tracks)), dtype=np.float32)

    cost_matrix = np.zeros((len(a_tracks), len(b_tracks)), dtype=np.float32)
    for i, track_a in enumerate(a_tracks):
        tlbr_a = track_a.tlbr
        for j, track_b in enumerate(b_tracks):
            tlbr_b = track_b.tlbr
            cost_matrix[i, j] = 1.0 - bbox_iou(tlbr_a, tlbr_b)
    return cost_matrix


def fuse_detection_scores(cost_matrix, detections, score_weight=1.0):
    if cost_matrix.size == 0 or not detections or score_weight <= 0:
        return cost_matrix

    weight = float(np.clip(score_weight, 0.0, 1.0))
    det_scores = np.asarray([getattr(det, "score", 1.0) for det in detections], dtype=np.float32)
    det_scores = np.clip(det_scores, 0.0, 1.0)
    score_scale = (1.0 - weight) + weight * det_scores
    similarity = (1.0 - cost_matrix) * score_scale[np.newaxis, :]
    return 1.0 - similarity


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return [], tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    finite_mask = np.isfinite(cost_matrix)
    valid_rows = np.where(finite_mask.any(axis=1))[0]
    valid_cols = np.where(finite_mask.any(axis=0))[0]
    if valid_rows.size == 0 or valid_cols.size == 0:
        return [], tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    reduced = cost_matrix[np.ix_(valid_rows, valid_cols)].copy()
    large_cost = float(max(thresh + 1.0, 1e6))
    reduced[~np.isfinite(reduced)] = large_cost

    row_ind, col_ind = linear_sum_assignment(reduced)
    matches = []
    matched_rows = set()
    matched_cols = set()

    for row, col in zip(row_ind, col_ind):
        original_row = int(valid_rows[row])
        original_col = int(valid_cols[col])
        if cost_matrix[original_row, original_col] > thresh or not np.isfinite(cost_matrix[original_row, original_col]):
            continue
        matches.append((original_row, original_col))
        matched_rows.add(original_row)
        matched_cols.add(original_col)

    unmatched_rows = tuple(idx for idx in range(cost_matrix.shape[0]) if idx not in matched_rows)
    unmatched_cols = tuple(idx for idx in range(cost_matrix.shape[1]) if idx not in matched_cols)
    return matches, unmatched_rows, unmatched_cols


def joint_stracks(a_tracks, b_tracks):
    exists = {}
    result = []
    for track in a_tracks + b_tracks:
        if track.track_id not in exists:
            exists[track.track_id] = True
            result.append(track)
    return result


def sub_stracks(a_tracks, b_tracks):
    b_ids = {track.track_id for track in b_tracks}
    return [track for track in a_tracks if track.track_id not in b_ids]


def remove_duplicate_stracks(a_tracks, b_tracks):
    pdist = iou_distance(a_tracks, b_tracks)
    if pdist.size == 0:
        return a_tracks, b_tracks

    duplicates_a = set()
    duplicates_b = set()
    for row in range(pdist.shape[0]):
        for col in range(pdist.shape[1]):
            if pdist[row, col] < 0.15:
                time_a = a_tracks[row].frame_id - a_tracks[row].start_frame
                time_b = b_tracks[col].frame_id - b_tracks[col].start_frame
                if time_a > time_b:
                    duplicates_b.add(col)
                else:
                    duplicates_a.add(row)

    kept_a = [track for idx, track in enumerate(a_tracks) if idx not in duplicates_a]
    kept_b = [track for idx, track in enumerate(b_tracks) if idx not in duplicates_b]
    return kept_a, kept_b


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


@dataclass
class Detection:
    tlwh: np.ndarray
    score: float
    feature: np.ndarray | None = None

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class STrack:
    _count = count(1)

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.track_id = 0
        self.state = TrackState.New
        self.score = float(score)
        self.tracklet_len = 0
        self.start_frame = 0
        self.frame_id = 0
        self.smooth_feature = None

    @staticmethod
    def next_id():
        return next(STrack._count)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def end_frame(self):
        return self.frame_id

    def activate(self, kalman_filter, frame_id, activated=True):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = bool(activated)
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            new_track.to_xyah(),
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self._update_feature(new_track.smooth_feature)
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            new_track.to_xyah(),
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self._update_feature(new_track.smooth_feature)

    def predict(self):
        if self.mean is None or self.covariance is None:
            return

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def _update_feature(self, feature, momentum=0.8):
        if feature is None:
            return
        feature = np.asarray(feature, dtype=np.float32)
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-6:
            return
        feature = feature / norm
        if self.smooth_feature is None or momentum <= 0.0:
            self.smooth_feature = feature
            return
        blended = momentum * self.smooth_feature + (1.0 - momentum) * feature
        blend_norm = float(np.linalg.norm(blended))
        if blend_norm > 1e-6:
            self.smooth_feature = blended / blend_norm

    def to_xyah(self):
        ret = self._tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def multi_predict(stracks):
        for track in stracks:
            track.predict()


class ByteTrackerLite:
    """
    ByteTrack-style lightweight tracker with:
    - Kalman prediction
    - high/low score two-stage association
    - tracked/lost/removed track state management
    - optional appearance cues only for ambiguous matches
    """

    def __init__(
        self,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        new_track_confirm_frames=1,
        match_thresh=0.8,
        low_match_thresh=0.5,
        unconfirmed_match_thresh=0.7,
        score_fusion_weight=0.0,
        max_time_lost=30,
        min_box_area=10.0,
        appearance_enabled=False,
        appearance_weight=0.25,
        appearance_ambiguity_margin=0.05,
        appearance_all_valid=False,
        appearance_feature_mode="hsv",
        appearance_hist_bins=(8, 4, 4),
        appearance_min_box_size=16,
        appearance_reid_model="mobilenet_v3_small",
        appearance_reid_weights="",
        appearance_reid_device="",
        appearance_reid_input_size=(256, 128),
        appearance_reid_flip_aug=False,
        track_stability_weight=0.0,
        motion_gate_enabled=False,
        motion_gate_thresh=9.4877,
        crowd_boost_enabled=False,
        crowd_boost_det_count=12,
        crowd_match_thresh=None,
        crowd_low_match_thresh=None,
        crowd_new_track_confirm_frames=None,
        crowd_appearance_weight=None,
        crowd_track_stability_weight=None,
        crowd_boost_min_small_ratio=None,
        crowd_boost_max_median_area_ratio=None,
        crowd_boost_small_area_ratio_thresh=0.002,
        cmc_enabled=False,
        cmc_motion_model="affine",
        cmc_ecc_iterations=50,
        cmc_ecc_eps=1e-4,
        cmc_downscale=1.0,
    ):
        self.track_high_thresh = float(track_high_thresh)
        self.track_low_thresh = float(track_low_thresh)
        self.new_track_thresh = float(new_track_thresh)
        self.new_track_confirm_frames = max(1, int(new_track_confirm_frames))
        self.match_thresh = float(match_thresh)
        self.low_match_thresh = float(low_match_thresh)
        self.unconfirmed_match_thresh = float(unconfirmed_match_thresh)
        self.score_fusion_weight = float(score_fusion_weight)
        self.max_time_lost = int(max_time_lost)
        self.min_box_area = float(min_box_area)
        self.appearance_enabled = bool(appearance_enabled)
        self.appearance_weight = float(np.clip(appearance_weight, 0.0, 1.0))
        self.appearance_ambiguity_margin = float(max(0.0, appearance_ambiguity_margin))
        self.appearance_all_valid = bool(appearance_all_valid)
        self.appearance_feature_mode = str(appearance_feature_mode or "hsv").lower()
        bins = tuple(int(value) for value in appearance_hist_bins)
        self.appearance_hist_bins = bins if len(bins) == 3 else (8, 4, 4)
        self.appearance_min_box_size = int(max(4, appearance_min_box_size))
        reid_hw = tuple(int(value) for value in appearance_reid_input_size)
        self.appearance_reid_input_size = reid_hw if len(reid_hw) == 2 else (256, 128)
        self.appearance_reid_model = str(appearance_reid_model or "mobilenet_v3_small")
        self.appearance_reid_weights = str(appearance_reid_weights or "")
        self.appearance_reid_device = str(appearance_reid_device or "")
        self.appearance_reid_flip_aug = bool(appearance_reid_flip_aug)
        self.track_stability_weight = float(np.clip(track_stability_weight, 0.0, 0.2))
        self.motion_gate_enabled = bool(motion_gate_enabled)
        self.motion_gate_thresh = float(max(0.0, motion_gate_thresh))
        self.crowd_boost_enabled = bool(crowd_boost_enabled)
        self.crowd_boost_det_count = max(1, int(crowd_boost_det_count))
        self.crowd_match_thresh = (
            float(crowd_match_thresh)
            if crowd_match_thresh is not None
            else float(self.match_thresh)
        )
        self.crowd_low_match_thresh = (
            float(crowd_low_match_thresh)
            if crowd_low_match_thresh is not None
            else float(self.low_match_thresh)
        )
        self.crowd_new_track_confirm_frames = (
            max(1, int(crowd_new_track_confirm_frames))
            if crowd_new_track_confirm_frames is not None
            else self.new_track_confirm_frames
        )
        self.crowd_appearance_weight = float(
            np.clip(
                self.appearance_weight if crowd_appearance_weight is None else crowd_appearance_weight,
                0.0,
                1.0,
            )
        )
        self.crowd_track_stability_weight = float(
            np.clip(
                self.track_stability_weight
                if crowd_track_stability_weight is None
                else crowd_track_stability_weight,
                0.0,
                0.2,
            )
        )
        self.crowd_boost_min_small_ratio = (
            float(crowd_boost_min_small_ratio)
            if crowd_boost_min_small_ratio is not None
            else None
        )
        self.crowd_boost_max_median_area_ratio = (
            float(crowd_boost_max_median_area_ratio)
            if crowd_boost_max_median_area_ratio is not None
            else None
        )
        self.crowd_boost_small_area_ratio_thresh = float(max(1e-6, crowd_boost_small_area_ratio_thresh))
        self.reid_encoder = None
        self.motion_compensator = ECCMotionCompensator(
            enabled=cmc_enabled,
            motion_model=cmc_motion_model,
            ecc_iterations=cmc_ecc_iterations,
            ecc_eps=cmc_ecc_eps,
            downscale=cmc_downscale,
        )

        self.kalman_filter = KalmanFilterXYAH()
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

    def _to_detections(self, detections):
        converted = []
        for det in detections:
            x1, y1, x2, y2, score = det
            tlwh = STrack.tlbr_to_tlwh([x1, y1, x2, y2])
            if tlwh[2] * tlwh[3] < self.min_box_area:
                continue
            converted.append(Detection(tlwh=tlwh, score=float(score)))
        return converted

    def _extract_hsv_hist_feature(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            list(self.appearance_hist_bins),
            [0, 180, 0, 256, 0, 256],
        )
        if hist is None:
            return None
        feature = hist.astype(np.float32).reshape(-1)
        feature_sum = float(feature.sum())
        if feature_sum <= 1e-6:
            return None
        return feature / feature_sum

    def _get_reid_encoder(self):
        if self.reid_encoder is None:
            self.reid_encoder = ReIDFeatureExtractor(
                model_name=self.appearance_reid_model,
                device=self.appearance_reid_device,
                weights_path=self.appearance_reid_weights,
                input_size=self.appearance_reid_input_size,
                flip_aug=self.appearance_reid_flip_aug,
            )
        return self.reid_encoder

    def _extract_feature(self, frame, tlbr):
        if frame is None:
            return None
        x1, y1, x2, y2 = [int(round(value)) for value in tlbr]
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        if (x2 - x1) < self.appearance_min_box_size or (y2 - y1) < self.appearance_min_box_size:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if self.appearance_feature_mode == "deep_reid":
            return self._get_reid_encoder().encode(crop)
        return self._extract_hsv_hist_feature(crop)

    def _appearance_similarity(self, track_feature, detection_feature):
        if track_feature is None or detection_feature is None:
            return 0.0
        if self.appearance_feature_mode == "deep_reid":
            score = float(np.dot(track_feature, detection_feature))
            return float(np.clip(score, 0.0, 1.0))
        return float(np.minimum(track_feature, detection_feature).sum())

    def _track_stability_score(self, track):
        age = max(int(track.tracklet_len), int(track.frame_id - track.start_frame + 1), 1)
        stability = float(np.log1p(age) / np.log1p(30.0))
        stability = float(np.clip(stability, 0.0, 1.0))
        if track.state == TrackState.Lost:
            stability *= 0.5
        return stability

    def _ensure_detection_features(self, detections, frame, indices):
        if not self.appearance_enabled or frame is None:
            return
        for idx in indices:
            if idx < 0 or idx >= len(detections):
                continue
            if detections[idx].feature is None:
                detections[idx].feature = self._extract_feature(frame, detections[idx].tlbr)

    def _collect_ambiguous_pairs(self, cost_matrix, thresh):
        if cost_matrix.size == 0 or self.appearance_ambiguity_margin <= 0:
            return set(), set()

        ambiguous_rows = set()
        ambiguous_cols = set()
        valid_mask = np.isfinite(cost_matrix) & (cost_matrix <= thresh)

        for row_idx in range(cost_matrix.shape[0]):
            valid_cols = np.where(valid_mask[row_idx])[0]
            if valid_cols.size < 2:
                continue
            row_costs = cost_matrix[row_idx, valid_cols]
            order = np.argsort(row_costs)
            if (row_costs[order[1]] - row_costs[order[0]]) <= self.appearance_ambiguity_margin:
                ambiguous_rows.add(row_idx)
                ambiguous_cols.update(valid_cols[order[:2]].tolist())

        for col_idx in range(cost_matrix.shape[1]):
            valid_rows = np.where(valid_mask[:, col_idx])[0]
            if valid_rows.size < 2:
                continue
            col_costs = cost_matrix[valid_rows, col_idx]
            order = np.argsort(col_costs)
            if (col_costs[order[1]] - col_costs[order[0]]) <= self.appearance_ambiguity_margin:
                ambiguous_cols.add(col_idx)
                ambiguous_rows.update(valid_rows[order[:2]].tolist())

        return ambiguous_rows, ambiguous_cols

    def _apply_appearance_association(
        self,
        cost_matrix,
        tracks,
        detections,
        frame,
        thresh,
        appearance_weight,
        track_stability_weight,
    ):
        if (
            not self.appearance_enabled
            or frame is None
            or cost_matrix.size == 0
            or not tracks
            or not detections
        ):
            return cost_matrix

        valid_mask = np.isfinite(cost_matrix) & (cost_matrix <= thresh)
        if self.appearance_all_valid:
            target_rows = set(np.where(valid_mask.any(axis=1))[0].tolist())
            target_cols = set(np.where(valid_mask.any(axis=0))[0].tolist())
            ambiguous_rows: set[int] = set()
            ambiguous_cols: set[int] = set()
        else:
            ambiguous_rows, ambiguous_cols = self._collect_ambiguous_pairs(cost_matrix, thresh)
            if not ambiguous_rows and not ambiguous_cols:
                return cost_matrix
            target_rows = set(ambiguous_rows)
            target_cols = set(ambiguous_cols)

        if ambiguous_cols:
            for col_idx in ambiguous_cols:
                candidate_rows = np.where(valid_mask[:, col_idx])[0].tolist()
                target_rows.update(candidate_rows)
        if ambiguous_rows:
            for row_idx in ambiguous_rows:
                target_cols.update(np.where(valid_mask[row_idx])[0].tolist())

        if self.appearance_enabled and appearance_weight > 0.0:
            self._ensure_detection_features(detections, frame, sorted(target_cols))
        adjusted = cost_matrix.copy()

        for row_idx in sorted(target_rows):
            track_feature = getattr(tracks[row_idx], "smooth_feature", None)
            if track_feature is None:
                track_feature = None
            track_tlbr = tracks[row_idx].tlbr
            for col_idx in sorted(target_cols):
                if not valid_mask[row_idx, col_idx]:
                    continue
                spatial_similarity = 1.0 - float(cost_matrix[row_idx, col_idx])
                fused_similarity = spatial_similarity

                effective_appearance_weight = 0.0
                if self.appearance_enabled and appearance_weight > 0.0:
                    detection_feature = detections[col_idx].feature
                    if detection_feature is not None and track_feature is not None:
                        effective_appearance_weight = appearance_weight
                        appearance_similarity = self._appearance_similarity(track_feature, detection_feature)
                        fused_similarity = (
                            (1.0 - effective_appearance_weight) * fused_similarity
                            + effective_appearance_weight * appearance_similarity
                        )

                if track_stability_weight > 0.0:
                    stability_bonus = track_stability_weight * self._track_stability_score(tracks[row_idx])
                    fused_similarity = min(1.0, fused_similarity + stability_bonus * spatial_similarity)

                adjusted[row_idx, col_idx] = 1.0 - fused_similarity

        return adjusted

    def _is_crowd_boost_frame(self, detections, frame):
        total_candidate_detections = int(len(detections))
        if not self.crowd_boost_enabled or total_candidate_detections < self.crowd_boost_det_count:
            return False
        if frame is None:
            return True
        if (
            self.crowd_boost_min_small_ratio is None
            and self.crowd_boost_max_median_area_ratio is None
        ):
            return True

        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1.0, float(frame_height * frame_width))
        area_ratios = []
        for detection in detections:
            tlwh = getattr(detection, "tlwh", None)
            if tlwh is None:
                continue
            width = max(1.0, float(tlwh[2]))
            height = max(1.0, float(tlwh[3]))
            area_ratios.append((width * height) / frame_area)
        if not area_ratios:
            return False

        area_ratios = np.asarray(area_ratios, dtype=np.float32)
        small_ratio = float(np.mean(area_ratios <= self.crowd_boost_small_area_ratio_thresh))
        median_area_ratio = float(np.median(area_ratios))
        if self.crowd_boost_min_small_ratio is not None and small_ratio < self.crowd_boost_min_small_ratio:
            return False
        if (
            self.crowd_boost_max_median_area_ratio is not None
            and median_area_ratio > self.crowd_boost_max_median_area_ratio
        ):
            return False
        return True

    def _resolve_frame_params(self, detections, frame):
        if not self._is_crowd_boost_frame(detections, frame):
            return {
                "match_thresh": self.match_thresh,
                "low_match_thresh": self.low_match_thresh,
                "new_track_confirm_frames": self.new_track_confirm_frames,
                "appearance_weight": self.appearance_weight,
                "track_stability_weight": self.track_stability_weight,
            }
        return {
            "match_thresh": self.crowd_match_thresh,
            "low_match_thresh": self.crowd_low_match_thresh,
            "new_track_confirm_frames": self.crowd_new_track_confirm_frames,
            "appearance_weight": self.crowd_appearance_weight,
            "track_stability_weight": self.crowd_track_stability_weight,
        }

    def _make_track_from_detection(self, detection):
        det_track = STrack(detection.tlwh, detection.score)
        det_track.kalman_filter = self.kalman_filter
        det_track._update_feature(detection.feature, momentum=0.0)
        return det_track

    def _apply_motion_gate(self, cost_matrix, tracks, detections):
        if (
            not self.motion_gate_enabled
            or cost_matrix.size == 0
            or not tracks
            or not detections
        ):
            return cost_matrix

        adjusted = cost_matrix.copy()
        measurements = np.asarray(
            [self._make_track_from_detection(det).to_xyah() for det in detections],
            dtype=np.float32,
        )
        for row_idx, track in enumerate(tracks):
            if track.mean is None or track.covariance is None:
                continue
            gating_distances = self.kalman_filter.gating_distance(track.mean, track.covariance, measurements)
            adjusted[row_idx, gating_distances > self.motion_gate_thresh] = np.inf
        return adjusted

    def update(self, detections, frame=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        tentative_stracks = []
        lost_stracks = []
        removed_stracks = []

        detections = self._to_detections(detections)
        high_score_detections = [
            det for det in detections if det.score >= self.track_high_thresh
        ]
        low_score_detections = [
            det for det in detections if self.track_low_thresh <= det.score < self.track_high_thresh
        ]
        frame_params = self._resolve_frame_params(high_score_detections + low_score_detections, frame)
        match_thresh = float(frame_params["match_thresh"])
        low_match_thresh = float(frame_params["low_match_thresh"])
        appearance_weight = float(frame_params["appearance_weight"])
        track_stability_weight = float(frame_params["track_stability_weight"])

        tracked_stracks = []
        unconfirmed = []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        warp_matrix = self.motion_compensator.estimate(frame)
        if warp_matrix is not None:
            for track in strack_pool:
                self.motion_compensator.apply(track, warp_matrix)
            for track in unconfirmed:
                self.motion_compensator.apply(track, warp_matrix)
        STrack.multi_predict(strack_pool)

        dists = iou_distance(strack_pool, high_score_detections)
        dists = self._apply_motion_gate(dists, strack_pool, high_score_detections)
        dists = fuse_detection_scores(dists, high_score_detections, self.score_fusion_weight)
        dists = self._apply_appearance_association(
            dists,
            strack_pool,
            high_score_detections,
            frame,
            match_thresh,
            appearance_weight,
            track_stability_weight,
        )
        matches, u_track, u_detection = linear_assignment(dists, thresh=match_thresh)

        for track_idx, det_idx in matches:
            track = strack_pool[track_idx]
            det_track = self._make_track_from_detection(high_score_detections[det_idx])

            if track.state == TrackState.Tracked:
                track.update(det_track, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det_track, self.frame_id, new_id=False)
                refind_stracks.append(track)

        remaining_tracked = [
            strack_pool[idx]
            for idx in u_track
            if strack_pool[idx].state == TrackState.Tracked
        ]
        dists_low = iou_distance(remaining_tracked, low_score_detections)
        dists_low = self._apply_motion_gate(dists_low, remaining_tracked, low_score_detections)
        dists_low = self._apply_appearance_association(
            dists_low,
            remaining_tracked,
            low_score_detections,
            frame,
            low_match_thresh,
            appearance_weight,
            track_stability_weight,
        )
        matches_low, u_track_low, u_low_detection = linear_assignment(dists_low, thresh=low_match_thresh)

        for track_idx, det_idx in matches_low:
            track = remaining_tracked[track_idx]
            det_track = self._make_track_from_detection(low_score_detections[det_idx])
            track.update(det_track, self.frame_id)
            activated_stracks.append(track)

        for idx in u_track_low:
            track = remaining_tracked[idx]
            track.mark_lost()
            lost_stracks.append(track)

        unmatched_high = [high_score_detections[idx] for idx in u_detection]
        dists_unc = iou_distance(unconfirmed, unmatched_high)
        dists_unc = self._apply_motion_gate(dists_unc, unconfirmed, unmatched_high)
        dists_unc = fuse_detection_scores(dists_unc, unmatched_high, self.score_fusion_weight)
        dists_unc = self._apply_appearance_association(
            dists_unc,
            unconfirmed,
            unmatched_high,
            frame,
            self.unconfirmed_match_thresh,
            appearance_weight,
            track_stability_weight,
        )
        matches_unc, u_unconfirmed, u_detection = linear_assignment(
            dists_unc,
            thresh=self.unconfirmed_match_thresh,
        )

        for track_idx, det_idx in matches_unc:
            track = unconfirmed[track_idx]
            det_track = self._make_track_from_detection(unmatched_high[det_idx])
            track.update(det_track, self.frame_id)
            activated_stracks.append(track)

        for idx in u_unconfirmed:
            track = unconfirmed[idx]
            track.mark_removed()
            removed_stracks.append(track)

        unmatched_high = [unmatched_high[idx] for idx in u_detection]
        for det in unmatched_high:
            if det.score < self.new_track_thresh:
                continue

            track = self._make_track_from_detection(det)
            if self.new_track_confirm_frames <= 1:
                track.activate(self.kalman_filter, self.frame_id, activated=True)
                activated_stracks.append(track)
            else:
                track.activate(self.kalman_filter, self.frame_id, activated=False)
                tentative_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [
            track for track in self.tracked_stracks
            if track.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, tentative_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, removed_stracks)

        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks,
            self.lost_stracks,
        )

        outputs = []
        for track in self.tracked_stracks:
            if not track.is_activated or track.state != TrackState.Tracked:
                continue

            x1, y1, x2, y2 = track.tlbr
            outputs.append(
                [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                    int(track.track_id),
                    float(track.score),
                ]
            )
        return outputs

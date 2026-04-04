from __future__ import annotations

from collections import defaultdict, deque
import math

from behavior.trajectory_behavior_classifier import features_from_trajectory_payload


class AbnormalDetector:
    def __init__(
        self,
        roi=(200, 120, 500, 420),
        behavior_mode="rules",
        enable_intrusion=False,
        enable_cross_line=False,
        enable_loitering=False,
        enable_running=False,
        intrusion_frames=1,
        cross_line=None,
        loiter_frames=168,
        loiter_radius=40.0,
        loiter_speed=3.0,
        running_speed=15.5,
        running_frames=3,
        history_size=180,
        inactive_patience_frames=2,
        behavior_classifier=None,
        secondary_behavior_classifier=None,
        behavior_ensemble_primary_weight=1.0,
        behavior_ensemble_mode="weighted",
        behavior_model_score_thresh=0.55,
        behavior_model_min_frames=24,
        behavior_model_max_tracks=32,
        behavior_model_resume_tracks=26,
        behavior_secondary_max_tracks=24,
        behavior_secondary_resume_tracks=18,
        behavior_secondary_loitering_only=True,
        behavior_model_eval_interval=3,
        loitering_hybrid_mode="union",
        loitering_model_support_thresh=0.0,
        loitering_model_score_thresh=0.97,
        running_model_score_thresh=0.85,
        loitering_model_min_frames=72,
        loitering_activate_frames=1,
        loitering_support_activate_frames=0,
        loitering_support_block_running=False,
        loitering_context_gate_support_only=False,
        loitering_release_frames=1,
        loitering_model_max_avg_speed=2.2,
        loitering_model_min_movement_extent=0.0,
        loitering_model_min_centroid_radius=0.0,
        loitering_model_max_movement_extent=55.0,
        loitering_model_max_centroid_radius=28.0,
        loitering_model_min_stationary_ratio=0.0,
        loitering_model_min_revisit_ratio=0.0,
        loitering_model_max_unique_cell_ratio=1.0,
        loitering_model_min_max_cell_occupancy_ratio=0.0,
        loitering_model_max_straightness=1.0,
        loitering_rule_min_stationary_ratio=0.0,
        loitering_rule_min_revisit_ratio=0.0,
        loitering_rule_min_movement_extent=0.0,
        loitering_rule_min_centroid_radius=0.0,
        loitering_rule_max_straightness=1.0,
        loitering_rule_max_displacement_ratio=1.0,
        loitering_max_neighbor_count=-1,
        loitering_neighbor_radius=80.0,
        running_model_min_avg_speed=8.0,
        running_model_min_p90_speed=16.0,
        running_model_min_movement_extent=120.0,
    ):
        self.roi = tuple(roi) if roi else None
        self.cross_line = self._normalize_line(cross_line)
        self.behavior_mode = str(behavior_mode or "rules").lower()
        self.enable_intrusion = bool(enable_intrusion)
        self.enable_cross_line = bool(enable_cross_line)
        self.enable_loitering = bool(enable_loitering)
        self.enable_running = bool(enable_running)
        self.intrusion_frames = max(1, int(intrusion_frames))
        self.loiter_frames = max(1, int(loiter_frames))
        self.loiter_radius = float(loiter_radius)
        self.loiter_speed = float(loiter_speed)
        self.running_speed = float(running_speed)
        self.running_frames = max(1, int(running_frames))
        self.history_size = max(10, int(history_size))
        self.inactive_patience_frames = max(0, int(inactive_patience_frames))
        self.behavior_classifier = behavior_classifier
        self.secondary_behavior_classifier = secondary_behavior_classifier
        self.behavior_ensemble_primary_weight = min(1.0, max(0.0, float(behavior_ensemble_primary_weight)))
        self.behavior_ensemble_mode = str(behavior_ensemble_mode or "weighted").lower()
        self.behavior_model_score_thresh = float(behavior_model_score_thresh)
        self.behavior_model_min_frames = max(1, int(behavior_model_min_frames))
        self.behavior_model_max_tracks = int(behavior_model_max_tracks)
        self.behavior_model_resume_tracks = int(behavior_model_resume_tracks)
        self.behavior_secondary_max_tracks = int(behavior_secondary_max_tracks)
        self.behavior_secondary_resume_tracks = int(behavior_secondary_resume_tracks)
        self.behavior_secondary_loitering_only = bool(behavior_secondary_loitering_only)
        self.behavior_model_eval_interval = max(1, int(behavior_model_eval_interval))
        self.loitering_hybrid_mode = str(loitering_hybrid_mode or "union").lower()
        self.loitering_model_support_thresh = float(loitering_model_support_thresh)
        self.loitering_model_score_thresh = float(loitering_model_score_thresh)
        self.running_model_score_thresh = float(running_model_score_thresh)
        self.loitering_model_min_frames = max(1, int(loitering_model_min_frames))
        self.loitering_activate_frames = max(1, int(loitering_activate_frames))
        support_activate_frames = int(loitering_support_activate_frames)
        if support_activate_frames <= 0:
            support_activate_frames = self.loitering_activate_frames
        self.loitering_support_activate_frames = max(self.loitering_activate_frames, support_activate_frames)
        self.loitering_support_block_running = bool(loitering_support_block_running)
        self.loitering_context_gate_support_only = bool(loitering_context_gate_support_only)
        self.loitering_release_frames = max(1, int(loitering_release_frames))
        self.loitering_model_max_avg_speed = float(loitering_model_max_avg_speed)
        self.loitering_model_min_movement_extent = float(loitering_model_min_movement_extent)
        self.loitering_model_min_centroid_radius = float(loitering_model_min_centroid_radius)
        self.loitering_model_max_movement_extent = float(loitering_model_max_movement_extent)
        self.loitering_model_max_centroid_radius = float(loitering_model_max_centroid_radius)
        self.loitering_model_min_stationary_ratio = float(loitering_model_min_stationary_ratio)
        self.loitering_model_min_revisit_ratio = float(loitering_model_min_revisit_ratio)
        self.loitering_model_max_unique_cell_ratio = float(loitering_model_max_unique_cell_ratio)
        self.loitering_model_min_max_cell_occupancy_ratio = float(loitering_model_min_max_cell_occupancy_ratio)
        self.loitering_model_max_straightness = float(loitering_model_max_straightness)
        self.loitering_rule_min_stationary_ratio = float(loitering_rule_min_stationary_ratio)
        self.loitering_rule_min_revisit_ratio = float(loitering_rule_min_revisit_ratio)
        self.loitering_rule_min_movement_extent = float(loitering_rule_min_movement_extent)
        self.loitering_rule_min_centroid_radius = float(loitering_rule_min_centroid_radius)
        self.loitering_rule_max_straightness = float(loitering_rule_max_straightness)
        self.loitering_rule_max_displacement_ratio = float(loitering_rule_max_displacement_ratio)
        self.loitering_max_neighbor_count = int(loitering_max_neighbor_count)
        self.loitering_neighbor_radius = float(loitering_neighbor_radius)
        self.running_model_min_avg_speed = float(running_model_min_avg_speed)
        self.running_model_min_p90_speed = float(running_model_min_p90_speed)
        self.running_model_min_movement_extent = float(running_model_min_movement_extent)

        self.track_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.speed_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.roi_dwell_frames = defaultdict(int)
        self.intrusion_active = defaultdict(bool)
        self.loitering_active = defaultdict(bool)
        self.loitering_positive_streak = defaultdict(int)
        self.loitering_negative_streak = defaultdict(int)
        self.running_active = defaultdict(bool)
        self.track_missing_frames = defaultdict(int)
        self.track_last_frame = {}
        self.display_history_size = 30
        self.frame_index = 0
        self.model_result_cache = {}
        self.frame_model_active = self._model_enabled()
        self.frame_secondary_model_active = self._secondary_model_enabled()

    @staticmethod
    def _normalize_line(line):
        if not line:
            return None
        if len(line) != 2:
            return None
        p1, p2 = line
        if len(p1) != 2 or len(p2) != 2:
            return None
        return (
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
        )

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _point_in_roi(self, point):
        if not self.roi:
            return False
        x, y = point
        x1, y1, x2, y2 = self.roi
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _line_side(point, line):
        (x1, y1), (x2, y2) = line
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _crossed_line(self, prev_point, curr_point):
        if self.cross_line is None or prev_point is None:
            return False
        side_prev = self._line_side(prev_point, self.cross_line)
        side_curr = self._line_side(curr_point, self.cross_line)
        if abs(side_prev) < 1e-6 or abs(side_curr) < 1e-6:
            return False
        return side_prev * side_curr < 0

    def _cleanup_inactive_tracks(self, active_track_ids):
        active_track_ids = set(active_track_ids)
        for track_id in active_track_ids:
            self.track_missing_frames[track_id] = 0
            self.track_last_frame[track_id] = self.frame_index

        tracked_maps = (
            self.track_history,
            self.speed_history,
            self.roi_dwell_frames,
            self.intrusion_active,
            self.loitering_active,
            self.loitering_positive_streak,
            self.loitering_negative_streak,
            self.running_active,
            self.model_result_cache,
            self.track_missing_frames,
            self.track_last_frame,
        )
        known_track_ids = set()
        for mapping in tracked_maps:
            known_track_ids.update(mapping.keys())

        for track_id in known_track_ids:
            if track_id in active_track_ids:
                continue
            self.track_missing_frames[track_id] = int(self.track_missing_frames.get(track_id, 0)) + 1
            if self.track_missing_frames[track_id] <= self.inactive_patience_frames:
                continue
            for mapping in tracked_maps:
                mapping.pop(track_id, None)

    def _rule_enabled(self):
        return self.behavior_mode in {"rules", "hybrid"}

    def _model_enabled(self):
        return self.behavior_mode in {"model", "hybrid"} and self.behavior_classifier is not None

    def _secondary_model_enabled(self):
        return self.behavior_mode in {"model", "hybrid"} and self.secondary_behavior_classifier is not None

    def _frame_model_enabled(self, active_track_count):
        if not self._model_enabled():
            return False
        active_track_count = int(active_track_count)
        if self.behavior_model_max_tracks < 0:
            self.frame_model_active = True
            return True

        if self.frame_model_active:
            if active_track_count > self.behavior_model_max_tracks:
                self.frame_model_active = False
        else:
            resume_tracks = self.behavior_model_resume_tracks
            if resume_tracks < 0 or active_track_count <= resume_tracks:
                self.frame_model_active = True
        return self.frame_model_active

    def _frame_secondary_model_enabled(self, active_track_count):
        if not self._secondary_model_enabled():
            return False
        if not self._frame_model_enabled(active_track_count):
            self.frame_secondary_model_active = False
            return False
        active_track_count = int(active_track_count)
        if self.behavior_secondary_max_tracks < 0:
            self.frame_secondary_model_active = True
            return True

        if self.frame_secondary_model_active:
            if active_track_count > self.behavior_secondary_max_tracks:
                self.frame_secondary_model_active = False
        else:
            resume_tracks = self.behavior_secondary_resume_tracks
            if resume_tracks < 0 or active_track_count <= resume_tracks:
                self.frame_secondary_model_active = True
        return self.frame_secondary_model_active

    @staticmethod
    def _append_alarm(alarms, track_id, alarm_type):
        if any(int(existing_track_id) == int(track_id) and str(existing_alarm_type) == str(alarm_type) for existing_track_id, existing_alarm_type in alarms):
            return
        alarms.append((track_id, alarm_type))

    def _passes_loitering_model_gate(self, model_score, model_features, score_threshold=None):
        if model_features is None:
            return False
        if score_threshold is None:
            score_threshold = self.loitering_model_score_thresh
        return (
            float(model_score) >= float(score_threshold)
            and float(model_features.get("frame_count", 0.0)) >= self.loitering_model_min_frames
            and float(model_features.get("avg_speed", 0.0)) <= self.loitering_model_max_avg_speed
            and float(model_features.get("movement_extent", 0.0)) >= self.loitering_model_min_movement_extent
            and float(model_features.get("centroid_radius", 0.0)) >= self.loitering_model_min_centroid_radius
            and float(model_features.get("movement_extent", 0.0)) <= self.loitering_model_max_movement_extent
            and float(model_features.get("centroid_radius", 0.0)) <= self.loitering_model_max_centroid_radius
            and float(model_features.get("stationary_ratio", 0.0)) >= self.loitering_model_min_stationary_ratio
            and float(model_features.get("revisit_ratio", 0.0)) >= self.loitering_model_min_revisit_ratio
            and float(model_features.get("unique_cell_ratio", 1.0)) <= self.loitering_model_max_unique_cell_ratio
            and float(model_features.get("max_cell_occupancy_ratio", 0.0)) >= self.loitering_model_min_max_cell_occupancy_ratio
            and float(model_features.get("straightness", 1.0)) <= self.loitering_model_max_straightness
        )

    def _passes_loitering_context_gate(self, nearby_track_count):
        return self.loitering_max_neighbor_count < 0 or int(nearby_track_count) <= self.loitering_max_neighbor_count

    def _passes_loitering_rule_pattern_gate(self, feature_dict):
        if feature_dict is None:
            return False
        return (
            float(feature_dict.get("stationary_ratio", 0.0)) >= self.loitering_rule_min_stationary_ratio
            and float(feature_dict.get("revisit_ratio", 0.0)) >= self.loitering_rule_min_revisit_ratio
            and float(feature_dict.get("movement_extent", 0.0)) >= self.loitering_rule_min_movement_extent
            and float(feature_dict.get("centroid_radius", 0.0)) >= self.loitering_rule_min_centroid_radius
            and float(feature_dict.get("straightness", 1.0)) <= self.loitering_rule_max_straightness
        )

    def _passes_running_model_gate(self, model_score, model_features):
        if model_features is None:
            return False
        return (
            float(model_score) >= self.running_model_score_thresh
            and float(model_features.get("avg_speed", 0.0)) >= self.running_model_min_avg_speed
            and float(model_features.get("p90_speed", 0.0)) >= self.running_model_min_p90_speed
            and float(model_features.get("movement_extent", 0.0)) >= self.running_model_min_movement_extent
        )

    def _should_score_loitering_model(self, avg_speed, frame_count):
        if not self.enable_loitering:
            return False
        if frame_count < self.loitering_model_min_frames:
            return False
        max_candidate_avg_speed = max(
            6.0,
            self.loitering_model_max_avg_speed * 3.0,
            self.loiter_speed * 3.0,
        )
        return float(avg_speed) <= float(max_candidate_avg_speed)

    def _should_score_running_model(self, speed, avg_speed, speed_history):
        if not self.enable_running:
            return False
        if len(speed_history) < self.running_frames:
            return False
        recent_max_speed = max(speed_history[-self.running_frames:], default=0.0)
        min_candidate_speed = min(
            8.0,
            self.running_speed * 0.5,
            self.running_model_min_p90_speed * 0.5,
        )
        return max(float(speed), float(avg_speed), float(recent_max_speed)) >= float(min_candidate_speed)

    def _blend_model_score(self, primary_score, secondary_score):
        primary_score = float(primary_score)
        secondary_score = float(secondary_score)
        if self.behavior_ensemble_mode == "max":
            return max(primary_score, secondary_score)
        if self.behavior_ensemble_mode == "geometric":
            return math.sqrt(max(0.0, primary_score) * max(0.0, secondary_score))
        if self.behavior_ensemble_mode == "geometric_weighted":
            primary_weight = float(self.behavior_ensemble_primary_weight)
            secondary_weight = 1.0 - primary_weight
            primary_term = max(1e-6, primary_score) ** primary_weight
            secondary_term = max(1e-6, secondary_score) ** secondary_weight
            return primary_term * secondary_term
        primary_weight = float(self.behavior_ensemble_primary_weight)
        secondary_weight = 1.0 - primary_weight
        return primary_score * primary_weight + secondary_score * secondary_weight

    def update(self, tracks):
        self.frame_index += 1
        alarms = []
        track_infos = {}
        active_track_ids = []
        active_track_count = len(tracks)
        frame_model_enabled = self._frame_model_enabled(active_track_count)
        frame_secondary_model_enabled = self._frame_secondary_model_enabled(active_track_count)
        enable_neighbor_gate = (
            frame_model_enabled
            and self.enable_loitering
            and self.loitering_max_neighbor_count >= 0
        )
        centers_by_track_id = {}
        if enable_neighbor_gate:
            centers_by_track_id = {
                int(track[4]): self._center([int(track[0]), int(track[1]), int(track[2]), int(track[3])])
                for track in tracks
            }

        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            track_id = int(track_id)
            active_track_ids.append(track_id)

            box = [int(x1), int(y1), int(x2), int(y2)]
            center = self._center(box)
            nearby_track_count = 0

            trajectory = self.track_history[track_id]
            prev_center = trajectory[-1] if trajectory else None
            last_seen_frame = self.track_last_frame.get(track_id, self.frame_index)
            trajectory.append(center)

            speed = 0.0
            if prev_center is not None:
                frame_gap = max(1, int(self.frame_index) - int(last_seen_frame))
                speed = self._distance(center, prev_center) / float(frame_gap)

            speed_hist = self.speed_history[track_id]
            speed_hist.append(speed)
            avg_speed = sum(speed_hist) / len(speed_hist) if speed_hist else 0.0

            inside_roi = self._point_in_roi(center)
            if self.roi:
                if inside_roi:
                    self.roi_dwell_frames[track_id] += 1
                else:
                    self.roi_dwell_frames[track_id] = 0
                    self.intrusion_active[track_id] = False
                    self.loitering_active[track_id] = False
            else:
                self.roi_dwell_frames[track_id] = 0
                self.intrusion_active[track_id] = False

            dwell_frames = self.roi_dwell_frames[track_id]

            intrusion_now = bool(
                self.enable_intrusion
                and inside_roi
                and dwell_frames >= self.intrusion_frames
            )
            if intrusion_now:
                self._append_alarm(alarms, track_id, "intrusion")
                self.intrusion_active[track_id] = True
            else:
                self.intrusion_active[track_id] = False

            if self.enable_cross_line and self._crossed_line(prev_center, center):
                alarms.append((track_id, "cross_line"))

            rule_loitering_now = False
            move_range = 0.0
            displacement = 0.0
            rule_loitering_features = None
            if self.enable_loitering and self._rule_enabled() and len(trajectory) >= self.loiter_frames:
                recent_points = list(trajectory)[-self.loiter_frames:]
                if len(recent_points) >= self.loiter_frames:
                    xs = [p[0] for p in recent_points]
                    ys = [p[1] for p in recent_points]
                    move_range = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                    displacement = self._distance(recent_points[0], recent_points[-1])
                    recent_speeds = list(speed_hist)[-self.loiter_frames:]
                    recent_avg_speed = sum(recent_speeds) / len(recent_speeds) if recent_speeds else 0.0
                    max_rule_displacement = self.loiter_radius * max(0.0, self.loitering_rule_max_displacement_ratio)
                    rule_loitering_features = features_from_trajectory_payload(
                        {
                            "centers": recent_points,
                            "speeds": recent_speeds,
                        },
                        low_speed_threshold=self.loiter_speed,
                        high_speed_threshold=self.running_speed,
                    )
                    rule_loitering_now = (
                        move_range <= self.loiter_radius
                        and recent_avg_speed <= self.loiter_speed
                        and displacement <= max_rule_displacement
                        and self._passes_loitering_rule_pattern_gate(rule_loitering_features)
                    )

            rule_running_now = False
            if self.enable_running and self._rule_enabled() and len(speed_hist) >= self.running_frames:
                recent_speeds = list(speed_hist)[-self.running_frames:]
                rule_running_now = all(speed_value >= self.running_speed for speed_value in recent_speeds)

            cache_entry = self.model_result_cache.get(track_id, {})
            track_infos[track_id] = {
                "trajectory": list(trajectory)[-self.display_history_size:],
                "bbox": box,
                "speed": speed,
                "avg_speed": avg_speed,
                "speed_history": list(speed_hist)[-self.display_history_size:],
                "frame_count": len(trajectory),
                "active_track_count": int(active_track_count),
                "inside_roi": inside_roi,
                "dwell_frames": dwell_frames,
                "move_range": move_range,
                "displacement": displacement,
                "rule_loitering_features": rule_loitering_features,
                "nearby_track_count": int(nearby_track_count),
                "conf": float(conf),
                "intrusion_active": bool(self.intrusion_active[track_id]),
                "rule_loitering": bool(rule_loitering_now),
                "rule_running": bool(rule_running_now),
                "loitering_positive_streak": int(self.loitering_positive_streak[track_id]),
                "loitering_negative_streak": int(self.loitering_negative_streak[track_id]),
                "model_behavior_label": None,
                "model_behavior_score": 0.0,
                "model_behavior_probs": {},
                "model_features": None,
                "model_window_size": 0,
                "model_loitering_score": 0.0,
                "model_running_score": 0.0,
                "secondary_model_behavior_label": None,
                "secondary_model_behavior_score": 0.0,
                "secondary_model_behavior_probs": {},
                "secondary_model_loitering_score": 0.0,
                "secondary_model_running_score": 0.0,
                "ensemble_loitering_score": 0.0,
                "ensemble_running_score": 0.0,
                "model_loitering_support": False,
                "support_only_loitering": False,
                "frame_model_enabled": bool(frame_model_enabled),
                "frame_secondary_model_enabled": bool(frame_secondary_model_enabled),
                "model_cache_hit": False,
            }

            model_loitering_now = False
            model_running_now = False
            model_loitering_support = False
            target_labels = []
            if self._should_score_loitering_model(avg_speed, len(trajectory)):
                target_labels.append("loitering")
            if self._should_score_running_model(speed, avg_speed, list(speed_hist)):
                target_labels.append("running")

            if enable_neighbor_gate and "loitering" in target_labels:
                nearby_track_count = sum(
                    1
                    for other_track_id, other_center in centers_by_track_id.items()
                    if int(other_track_id) != track_id and self._distance(center, other_center) <= self.loitering_neighbor_radius
                )
                track_infos[track_id]["nearby_track_count"] = int(nearby_track_count)

            if frame_model_enabled and len(trajectory) >= self.behavior_model_min_frames and target_labels:
                should_refresh_model = True
                cached_labels = set(cache_entry.get("target_labels", []))
                if cache_entry:
                    eval_interval = self.behavior_model_eval_interval
                    frame_due = ((self.frame_index + track_id) % eval_interval) == 0
                    if cached_labels == set(target_labels) and not frame_due:
                        should_refresh_model = False

                if should_refresh_model:
                    model_track_info = {
                        **track_infos[track_id],
                        "trajectory": list(trajectory),
                        "speed_history": list(speed_hist),
                    }
                    model_label, model_score, model_probs, model_metadata = self.behavior_classifier.predict_track_info(
                        model_track_info,
                        target_labels=target_labels,
                    )
                    model_features = None
                    model_window_size = 0
                    if isinstance(model_metadata, dict):
                        model_features = model_metadata.get("best_feature_dict")
                        model_window_size = int(model_metadata.get("best_window_size", 0) or 0)
                    loitering_model_entry = {}
                    running_model_entry = {}
                    if isinstance(model_metadata, dict):
                        per_label = model_metadata.get("per_label", {})
                        loitering_model_entry = per_label.get("loitering", {}) or {}
                        running_model_entry = per_label.get("running", {}) or {}

                    loitering_model_score = float(loitering_model_entry.get("score", model_probs.get("loitering", 0.0) or 0.0))
                    loitering_model_features = loitering_model_entry.get("feature_dict")
                    running_model_score = float(running_model_entry.get("score", model_probs.get("running", 0.0) or 0.0))
                    running_model_features = running_model_entry.get("feature_dict")

                    secondary_label = None
                    secondary_score = 0.0
                    secondary_probs = {}
                    secondary_loitering_score = 0.0
                    secondary_running_score = 0.0
                    ensemble_loitering_score = loitering_model_score
                    ensemble_running_score = running_model_score
                    secondary_target_labels = list(target_labels)
                    if self.behavior_secondary_loitering_only:
                        secondary_target_labels = [
                            label_name for label_name in secondary_target_labels if label_name == "loitering"
                        ]

                    if frame_secondary_model_enabled and secondary_target_labels:
                        secondary_label, secondary_score, secondary_probs, secondary_metadata = self.secondary_behavior_classifier.predict_track_info(
                            model_track_info,
                            target_labels=secondary_target_labels,
                        )
                        secondary_loitering_entry = {}
                        secondary_running_entry = {}
                        if isinstance(secondary_metadata, dict):
                            per_label = secondary_metadata.get("per_label", {})
                            secondary_loitering_entry = per_label.get("loitering", {}) or {}
                            secondary_running_entry = per_label.get("running", {}) or {}

                        secondary_loitering_score = float(secondary_loitering_entry.get("score", secondary_probs.get("loitering", 0.0) or 0.0))
                        secondary_running_score = float(secondary_running_entry.get("score", secondary_probs.get("running", 0.0) or 0.0))
                        ensemble_loitering_score = self._blend_model_score(loitering_model_score, secondary_loitering_score)
                        ensemble_running_score = self._blend_model_score(running_model_score, secondary_running_score)

                    cache_entry = {
                        "target_labels": list(target_labels),
                        "model_behavior_label": model_label,
                        "model_behavior_score": float(model_score),
                        "model_behavior_probs": model_probs,
                        "model_features": model_features,
                        "model_window_size": model_window_size,
                        "model_loitering_score": loitering_model_score,
                        "model_running_score": running_model_score,
                        "secondary_model_behavior_label": secondary_label,
                        "secondary_model_behavior_score": float(secondary_score),
                        "secondary_model_behavior_probs": secondary_probs,
                        "secondary_model_loitering_score": secondary_loitering_score,
                        "secondary_model_running_score": secondary_running_score,
                        "ensemble_loitering_score": ensemble_loitering_score,
                        "ensemble_running_score": ensemble_running_score,
                        "loitering_model_features": loitering_model_features,
                        "running_model_features": running_model_features,
                    }
                    self.model_result_cache[track_id] = cache_entry
                else:
                    track_infos[track_id]["model_cache_hit"] = True

                track_infos[track_id]["model_behavior_label"] = cache_entry.get("model_behavior_label")
                track_infos[track_id]["model_behavior_score"] = float(cache_entry.get("model_behavior_score", 0.0))
                track_infos[track_id]["model_behavior_probs"] = cache_entry.get("model_behavior_probs", {})
                track_infos[track_id]["model_features"] = cache_entry.get("model_features")
                track_infos[track_id]["model_window_size"] = int(cache_entry.get("model_window_size", 0) or 0)
                track_infos[track_id]["model_loitering_score"] = float(cache_entry.get("model_loitering_score", 0.0))
                track_infos[track_id]["model_running_score"] = float(cache_entry.get("model_running_score", 0.0))
                track_infos[track_id]["secondary_model_behavior_label"] = cache_entry.get("secondary_model_behavior_label")
                track_infos[track_id]["secondary_model_behavior_score"] = float(cache_entry.get("secondary_model_behavior_score", 0.0))
                track_infos[track_id]["secondary_model_behavior_probs"] = cache_entry.get("secondary_model_behavior_probs", {})
                track_infos[track_id]["secondary_model_loitering_score"] = float(cache_entry.get("secondary_model_loitering_score", 0.0))
                track_infos[track_id]["secondary_model_running_score"] = float(cache_entry.get("secondary_model_running_score", 0.0))
                track_infos[track_id]["ensemble_loitering_score"] = float(cache_entry.get("ensemble_loitering_score", 0.0))
                track_infos[track_id]["ensemble_running_score"] = float(cache_entry.get("ensemble_running_score", 0.0))

                loitering_model_features = cache_entry.get("loitering_model_features")
                running_model_features = cache_entry.get("running_model_features")
                model_loitering_support = (
                    self._passes_loitering_model_gate(
                        track_infos[track_id]["ensemble_loitering_score"],
                        loitering_model_features,
                        score_threshold=self.loitering_model_support_thresh,
                    )
                    and self._passes_loitering_context_gate(nearby_track_count)
                )
                track_infos[track_id]["model_loitering_support"] = bool(model_loitering_support)
                model_context_gate_ok = (
                    True
                    if self.loitering_context_gate_support_only
                    else self._passes_loitering_context_gate(nearby_track_count)
                )
                if self._passes_loitering_model_gate(
                    track_infos[track_id]["ensemble_loitering_score"],
                    loitering_model_features,
                ) and model_context_gate_ok:
                    model_loitering_now = True
                if self._passes_running_model_gate(track_infos[track_id]["ensemble_running_score"], running_model_features):
                    model_running_now = True

            loitering_now = False
            support_only_loitering_now = False
            if self.enable_loitering:
                if self.behavior_mode == "model":
                    loitering_now = model_loitering_now
                elif self.behavior_mode == "hybrid":
                    if self.loitering_hybrid_mode == "model_only":
                        loitering_now = model_loitering_now
                    elif self.loitering_hybrid_mode == "model_support":
                        support_only_loitering_now = bool(
                            rule_loitering_now
                            and model_loitering_support
                            and not model_loitering_now
                            and (
                                not self.loitering_support_block_running
                                or (not rule_running_now and not model_running_now)
                            )
                        )
                        loitering_now = model_loitering_now or support_only_loitering_now
                    elif self.loitering_hybrid_mode == "rules_only":
                        loitering_now = rule_loitering_now
                    else:
                        loitering_now = rule_loitering_now or model_loitering_now
                else:
                    loitering_now = rule_loitering_now
            track_infos[track_id]["support_only_loitering"] = bool(support_only_loitering_now)

            if loitering_now:
                self.loitering_positive_streak[track_id] += 1
                self.loitering_negative_streak[track_id] = 0
            else:
                self.loitering_positive_streak[track_id] = 0
                self.loitering_negative_streak[track_id] += 1

            if (
                loitering_now
                and not self.loitering_active[track_id]
                and self.loitering_positive_streak[track_id] >= (
                    self.loitering_support_activate_frames
                    if support_only_loitering_now
                    else self.loitering_activate_frames
                )
            ):
                self._append_alarm(alarms, track_id, "loitering")
                self.loitering_active[track_id] = True
            elif (
                not loitering_now
                and self.loitering_active[track_id]
                and self.loitering_negative_streak[track_id] >= self.loitering_release_frames
            ):
                self.loitering_active[track_id] = False

            track_infos[track_id]["loitering_positive_streak"] = int(self.loitering_positive_streak[track_id])
            track_infos[track_id]["loitering_negative_streak"] = int(self.loitering_negative_streak[track_id])
            track_infos[track_id]["loitering_active"] = bool(self.loitering_active[track_id])

            running_now = False
            if self.enable_running:
                if self.behavior_mode == "model":
                    running_now = model_running_now
                elif self.behavior_mode == "hybrid":
                    running_now = rule_running_now or model_running_now
                else:
                    running_now = rule_running_now

            if running_now and not self.running_active[track_id]:
                self._append_alarm(alarms, track_id, "running")
                self.running_active[track_id] = True
            elif not running_now:
                self.running_active[track_id] = False

            track_infos[track_id]["running_active"] = bool(self.running_active[track_id])
            track_infos[track_id]["is_alarm"] = any(alarm_track_id == track_id for alarm_track_id, _ in alarms)

        self._cleanup_inactive_tracks(active_track_ids)
        return alarms, track_infos

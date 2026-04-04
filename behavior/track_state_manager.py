from collections import deque
import math


class TrackStateManager:
    def __init__(self, max_history=30, max_missed_frames=30):
        self.max_history = max_history
        self.max_missed_frames = max_missed_frames
        self.tracks = {}

    def _create_track_state(self, track_id, bbox, center, frame_id):
        return {
            "track_id": track_id,
            "class_name": "person",

            "bbox": bbox,
            "center": center,
            "prev_center": None,

            "center_history": deque([center], maxlen=self.max_history),
            "bbox_history": deque([bbox], maxlen=self.max_history),
            "frame_history": deque([frame_id], maxlen=self.max_history),

            "speed": 0.0,
            "speed_history": deque(maxlen=self.max_history),
            "avg_speed": 0.0,

            "inside_intrusion_roi": False,
            "inside_loitering_roi": False,
            "enter_intrusion_frame": None,
            "enter_loitering_frame": None,

            "loiter_duration": 0,
            "loiter_radius": 0.0,

            "last_frame_id": frame_id,
            "missed_frames": 0,

            "last_alarm_frame": {},
            "triggered_events": set(),
        }

    def update_track(self, track_id, bbox, frame_id):
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if track_id not in self.tracks:
            self.tracks[track_id] = self._create_track_state(track_id, bbox, center, frame_id)
            return self.tracks[track_id]

        state = self.tracks[track_id]

        state["prev_center"] = state["center"]
        state["bbox"] = bbox
        state["center"] = center

        state["bbox_history"].append(bbox)
        state["center_history"].append(center)
        state["frame_history"].append(frame_id)

        if state["prev_center"] is not None:
            dx = center[0] - state["prev_center"][0]
            dy = center[1] - state["prev_center"][1]
            speed = math.sqrt(dx * dx + dy * dy)
        else:
            speed = 0.0

        state["speed"] = speed
        state["speed_history"].append(speed)

        if len(state["speed_history"]) > 0:
            state["avg_speed"] = sum(state["speed_history"]) / len(state["speed_history"])
        else:
            state["avg_speed"] = 0.0

        state["last_frame_id"] = frame_id
        state["missed_frames"] = 0

        return state

    def mark_missing_and_cleanup(self, active_track_ids):
        active_track_ids = set(active_track_ids)
        to_delete = []

        for track_id, state in self.tracks.items():
            if track_id not in active_track_ids:
                state["missed_frames"] += 1
                if state["missed_frames"] > self.max_missed_frames:
                    to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

    def get_track_state(self, track_id):
        return self.tracks.get(track_id)

    def get_all_states(self):
        return self.tracks
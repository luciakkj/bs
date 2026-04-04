import os
import cv2
import queue
import threading


class SnapshotService:
    def __init__(self, output_root="output/snaps", run_id="default", roi=None):
        self.output_dir = os.path.join(output_root, f"run_{run_id}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.roi = roi

        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _draw_roi(self, image):
        if not self.roi:
            return

        x1, y1, x2, y2 = self.roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            image,
            "ROI",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_alarm_box(self, image, bbox, track_id, alarm_type):
        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = [int(v) for v in bbox]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"ID {int(track_id)} | {alarm_type}"
        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20

        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_header(self, image, alarm_type, track_id):
        header = f"ALARM: {alarm_type} | TRACK ID: {int(track_id)}"
        cv2.putText(
            image,
            header,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def _build_annotated_frame(self, frame, alarm_type, track_id, bbox):
        image = frame.copy()

        self._draw_roi(image)
        self._draw_alarm_box(image, bbox, track_id, alarm_type)
        self._draw_header(image, alarm_type, track_id)

        return image

    def _worker_loop(self):
        while not self.stop_event.is_set() or not self.task_queue.empty():
            try:
                path, image = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                cv2.imwrite(path, image)
            finally:
                self.task_queue.task_done()

    def save(self, frame, alarm_type, track_id, bbox=None, frame_timestamp=None):
        annotated = self._build_annotated_frame(
            frame=frame,
            alarm_type=alarm_type,
            track_id=track_id,
            bbox=bbox,
        )

        suffix = frame_timestamp if frame_timestamp is not None else "frame"
        filename = f"{alarm_type}_tid{int(track_id)}_{suffix}.jpg"
        path = os.path.join(self.output_dir, filename)

        # 异步写盘：这里只入队，不阻塞主循环
        self.task_queue.put((path, annotated))

        return path

    def close(self):
        self.stop_event.set()
        self.task_queue.join()
        self.worker.join(timeout=1.0)
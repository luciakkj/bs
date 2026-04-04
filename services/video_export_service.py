import re
from pathlib import Path

import cv2


class VideoExportService:
    def __init__(
        self,
        output_root="output/videos",
        run_id="default",
        source_name="source",
        fps=25.0,
        codec="mp4v",
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.source_name = self._sanitize_name(source_name)
        self.fps = max(1.0, float(fps))
        self.codec = str(codec or "mp4v")[:4]
        self.output_path = str(self.output_root / f"run_{self.run_id}_{self.source_name}.mp4")
        self.writer = None
        self.frame_size = None

    @staticmethod
    def _sanitize_name(value):
        text = str(value or "source").strip()
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return sanitized or "source"

    def _ensure_writer(self, frame):
        if self.writer is not None:
            return

        height, width = frame.shape[:2]
        self.frame_size = (int(width), int(height))
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            float(self.fps),
            self.frame_size,
        )
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"Unable to open video writer: {self.output_path}")

    def write(self, frame):
        self._ensure_writer(frame)
        if self.writer is None:
            return

        height, width = frame.shape[:2]
        if self.frame_size != (int(width), int(height)):
            raise ValueError(
                f"Video frame size changed from {self.frame_size} to {(int(width), int(height))}."
            )
        self.writer.write(frame)

    def open(self, frame):
        self._ensure_writer(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

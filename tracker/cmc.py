from __future__ import annotations

import cv2
import numpy as np


class ECCMotionCompensator:
    """Lightweight camera motion compensation inspired by BoT-SORT GMC/CMC."""

    def __init__(
        self,
        enabled: bool = False,
        motion_model: str = "affine",
        ecc_iterations: int = 50,
        ecc_eps: float = 1e-4,
        downscale: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.motion_model = str(motion_model or "affine").lower()
        self.ecc_iterations = max(10, int(ecc_iterations))
        self.ecc_eps = max(1e-7, float(ecc_eps))
        self.downscale = float(np.clip(downscale, 0.1, 1.0))
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def _prepare_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale >= 0.999:
            return gray
        width = max(32, int(round(gray.shape[1] * self.downscale)))
        height = max(32, int(round(gray.shape[0] * self.downscale)))
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    def estimate(self, frame: np.ndarray | None) -> np.ndarray | None:
        if not self.enabled or frame is None:
            return None

        gray = self._prepare_gray(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        warp_mode = cv2.MOTION_AFFINE if self.motion_model == "affine" else cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ecc_iterations,
            self.ecc_eps,
        )

        try:
            # Estimate transform from previous frame to current frame.
            _, warp_matrix = cv2.findTransformECC(
                gray,
                self.prev_gray,
                warp_matrix,
                warp_mode,
                criteria,
                None,
                1,
            )
        except cv2.error:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = gray
        if self.downscale < 0.999:
            warp_matrix = warp_matrix.astype(np.float32)
            warp_matrix[:, 2] /= self.downscale
        return warp_matrix.astype(np.float32)

    @staticmethod
    def _transform_covariance(covariance: np.ndarray, linear: np.ndarray, scale: float) -> np.ndarray:
        transform = np.eye(8, dtype=np.float32)
        transform[0:2, 0:2] = linear
        transform[4:6, 4:6] = linear
        transform[3, 3] = scale
        transform[7, 7] = scale
        return np.linalg.multi_dot((transform, covariance, transform.T)).astype(np.float32)

    @staticmethod
    def apply(track, warp_matrix: np.ndarray | None) -> None:
        if warp_matrix is None or track.mean is None or track.covariance is None:
            return

        linear = warp_matrix[:, :2].astype(np.float32)
        translation = warp_matrix[:, 2].astype(np.float32)
        scale = float(np.mean(np.linalg.norm(linear, axis=0)))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0

        center = track.mean[:2].astype(np.float32)
        velocity = track.mean[4:6].astype(np.float32)

        track.mean[:2] = linear @ center + translation
        track.mean[4:6] = linear @ velocity
        track.mean[3] = float(max(1.0, track.mean[3] * scale))
        track.mean[7] = float(track.mean[7] * scale)
        track.covariance = ECCMotionCompensator._transform_covariance(track.covariance, linear, scale)

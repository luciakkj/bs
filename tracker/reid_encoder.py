from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ReIDFeatureExtractor:
    _MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str | None = None,
        weights_path: str | None = None,
        input_size: tuple[int, int] = (256, 128),
    ) -> None:
        self.model_name = str(model_name or "mobilenet_v3_small").lower()
        self.device = self._resolve_device(device)
        self.weights_path = str(weights_path or "").strip()
        self.input_size = (
            max(32, int(input_size[0])),
            max(16, int(input_size[1])),
        )
        self.model = self._load_model()
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _resolve_device(self, device: str | None) -> torch.device:
        device_str = str(device or "").strip()
        if not device_str:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device_str.isdigit():
            return torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
        try:
            return torch.device(device_str)
        except Exception:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> torch.nn.Module:
        cache_key = (self.model_name, str(self.device), self.weights_path)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if self.model_name != "mobilenet_v3_small":
            raise ValueError(f"Unsupported ReID model: {self.model_name}")

        if self.weights_path:
            model = mobilenet_v3_small(weights=None)
            state_dict = torch.load(self.weights_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        backbone = torch.nn.Sequential(model.features, model.avgpool)
        backbone.eval()
        backbone.to(self.device)
        self._MODEL_CACHE[cache_key] = backbone
        return backbone

    def encode(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None

        resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features = self.model(tensor)
            if features.ndim > 2:
                features = torch.flatten(features, 1)
            features = F.normalize(features, p=2, dim=1)

        vector = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if vector.size == 0:
            return None
        return vector

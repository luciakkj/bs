from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)


class ReIDFeatureExtractor:
    _MODEL_CACHE: dict[tuple[str, str, str], torch.nn.Module] = {}

    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str | None = None,
        weights_path: str | None = None,
        input_size: tuple[int, int] = (256, 128),
        flip_aug: bool = False,
    ) -> None:
        self.model_name = str(model_name or "mobilenet_v3_small").lower()
        self.device = self._resolve_device(device)
        self.weights_path = str(weights_path or "").strip()
        self.input_size = (
            max(32, int(input_size[0])),
            max(16, int(input_size[1])),
        )
        self.flip_aug = bool(flip_aug)
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

        if self.model_name == "mobilenet_v3_small":
            if self.weights_path:
                model = mobilenet_v3_small(weights=None)
                state_dict = torch.load(self.weights_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
            else:
                model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            backbone = torch.nn.Sequential(model.features, model.avgpool)
        elif self.model_name == "resnet50":
            if self.weights_path:
                model = resnet50(weights=None)
                state_dict = torch.load(self.weights_path, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
            else:
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            backbone = torch.nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
            )
        else:
            raise ValueError(f"Unsupported ReID model: {self.model_name}")

        backbone.eval()
        backbone.to(self.device)
        self._MODEL_CACHE[cache_key] = backbone
        return backbone

    def encode(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None

        resized = cv2.resize(crop, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        variants = [rgb]
        if self.flip_aug:
            variants.append(np.ascontiguousarray(rgb[:, ::-1, :]))
        batch = np.stack([(image - self.mean) / self.std for image in variants], axis=0)
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(self.device)

        with torch.inference_mode():
            features = self.model(tensor)
            if features.ndim > 2:
                features = torch.flatten(features, 1)
            features = F.normalize(features, p=2, dim=1)
            features = F.normalize(features.mean(dim=0, keepdim=True), p=2, dim=1)

        vector = features.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if vector.size == 0:
            return None
        return vector

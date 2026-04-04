from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from training.config import TrainConfig, ValidateConfig


def train_detector(config: TrainConfig) -> dict[str, object]:
    cfg = config.resolved()
    cfg.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(cfg.model))
    results = model.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=True,
    )

    save_dir = Path(getattr(results, "save_dir", cfg.project / cfg.name))
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    return {
        "save_dir": str(save_dir.resolve()),
        "best_weights": str(best_path.resolve()) if best_path.exists() else None,
        "last_weights": str(last_path.resolve()) if last_path.exists() else None,
    }


def validate_detector(config: ValidateConfig) -> dict[str, object]:
    cfg = config.resolved()
    model = YOLO(str(cfg.weights))
    metrics = model.val(
        data=str(cfg.data),
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        split=cfg.split,
        workers=cfg.workers,
    )

    box = getattr(metrics, "box", None)
    return {
        "weights": str(cfg.weights),
        "data": str(cfg.data),
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(getattr(box, "mp", 0.0) or 0.0),
        "recall": float(getattr(box, "mr", 0.0) or 0.0),
    }

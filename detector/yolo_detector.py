from ultralytics import YOLO


class YOLODetector:
    """Lightweight YOLO inference wrapper for person detection."""

    def __init__(
        self,
        model_path="models/yolov8n.pt",
        conf=0.1,
        classes=None,
        device=None,
        imgsz=None,
        max_det=None,
        nms_iou=None,
        half=False,
        augment=False,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = [0] if classes is None else classes
        self.device = device
        self.imgsz = int(imgsz) if imgsz else None
        self.max_det = int(max_det) if max_det else None
        self.nms_iou = float(nms_iou) if nms_iou is not None else None
        self.half = bool(half)
        self.augment = bool(augment)

    def _build_predict_kwargs(self, conf):
        predict_kwargs = {
            "conf": conf,
            "classes": self.classes,
            "device": self.device,
            "verbose": False,
        }
        if self.imgsz:
            predict_kwargs["imgsz"] = self.imgsz
        if self.max_det:
            predict_kwargs["max_det"] = self.max_det
        if self.nms_iou is not None:
            predict_kwargs["iou"] = self.nms_iou
        if self.half:
            predict_kwargs["half"] = self.half
        if self.augment:
            predict_kwargs["augment"] = self.augment
        return predict_kwargs

    def _predict(self, frame, conf):
        results = self.model.predict(
            frame,
            **self._build_predict_kwargs(conf),
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else []

        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, box[:4])
            detections.append([x1, y1, x2, y2, float(score)])
        return detections

    def detect(self, frame):
        return self._predict(frame, self.conf)

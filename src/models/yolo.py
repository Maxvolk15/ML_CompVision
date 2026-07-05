from __future__ import annotations

import torch

from src.models.base import BaseDetector, register_model


@register_model("yolo")
class YOLODetector(BaseDetector):
    """YOLOv8 (one-stage, anchor-free), Ultralytics. Обучается через fit()"""

    def __init__(self, model_cfg, full_cfg):
        super().__init__(model_cfg, full_cfg)
        from ultralytics import YOLO
        arch = model_cfg.get("arch", "yolov8n")
        self.yolo = YOLO(f"{arch}.pt" if model_cfg.get("pretrained", True) else f"{arch}.yaml")
        self.img_size = model_cfg.get("img_size", 640)

    def _fraction(self) -> float:
        """Доля обучающих изображений (аналог data.subset для остальных моделей)"""
        if self.model_cfg.get("fraction") is not None:
            return float(self.model_cfg["fraction"])
        subset = self.full_cfg["data"].get("subset")
        if not subset:
            return 1.0
        import os
        d = self.full_cfg["data"]["train_images"]
        try:
            n = sum(1 for f in os.listdir(d)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
        except OSError:
            return 1.0
        return min(1.0, int(subset) / n) if n else 1.0

    def fit(self, data_yaml: str, name: str = "yolo", **kwargs):
        from pathlib import Path
        tr = self.full_cfg["training"]
        pref = self.full_cfg["project"]["device"]
        device = "cpu" if (pref == "cpu" or not torch.cuda.is_available()) else 0
        project = str(Path(self.full_cfg["project"]["output_dir"]).resolve())
        results = self.yolo.train(
            data=data_yaml, epochs=kwargs.get("epochs", tr["epochs"]),
            batch=tr["batch_size"], imgsz=self.img_size, lr0=tr["lr"],
            optimizer=tr["optimizer"].upper(), seed=self.full_cfg["project"]["seed"],
            device=device, amp=tr.get("amp", True), fraction=self._fraction(),
            workers=self.full_cfg["project"].get("num_workers", 0),
            project=project, name=name, exist_ok=True)
        try:
            return {"mAP": float(results.box.map), "mAP_50": float(results.box.map50)}
        except Exception:
            return {"mAP": None, "mAP_50": None}

    def forward(self, images, targets=None):
        return self.yolo.model(images)

    @torch.no_grad()
    def predict(self, images):
        out = []
        for r in self.yolo.predict(images, imgsz=self.img_size, verbose=False):
            b = r.boxes
            out.append({"boxes": b.xyxy, "labels": b.cls.long(), "scores": b.conf})
        return out

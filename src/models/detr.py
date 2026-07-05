from __future__ import annotations

import torch

from src.models.base import BaseDetector, register_model


@register_model("detr")
class DETRDetector(BaseDetector):
    """DETR (transformer), HuggingFace transformers.

    Общий Trainer передаёт список изображений и torchvision-таргеты, а DETR ждёт
    батч-тензор pixel_values и метки в формате {class_labels, boxes(cx,cy,w,h норм.)}
    — конвертация выполняется здесь.
    """

    supports_amp = False  # в fp16 обучение DETR расходится до NaN — только fp32

    def __init__(self, model_cfg, full_cfg):
        super().__init__(model_cfg, full_cfg)
        from transformers import DetrForObjectDetection, DetrImageProcessor
        self.arch = model_cfg.get("arch", "facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            self.arch, num_labels=full_cfg["data"]["num_classes"],
            ignore_mismatched_sizes=True)
        self.processor = DetrImageProcessor.from_pretrained(self.arch)
        self.score_threshold = full_cfg["evaluation"].get("score_threshold", 0.05)

    @staticmethod
    def _stack(images):
        return torch.stack(list(images)) if isinstance(images, (list, tuple)) else images

    def _to_detr_labels(self, targets, h, w):
        labels = []
        for t in targets:
            b = t["boxes"]
            if b.numel():
                cx = (b[:, 0] + b[:, 2]) / 2 / w
                cy = (b[:, 1] + b[:, 3]) / 2 / h
                bw = (b[:, 2] - b[:, 0]) / w
                bh = (b[:, 3] - b[:, 1]) / h
                boxes = torch.stack([cx, cy, bw, bh], dim=1)
            else:
                boxes = b.reshape(0, 4)
            labels.append({"class_labels": t["labels"], "boxes": boxes})
        return labels

    def forward(self, images, targets=None):
        pixel_values = self._stack(images)
        if targets is not None:
            h, w = pixel_values.shape[-2:]
            out = self.model(pixel_values=pixel_values,
                             labels=self._to_detr_labels(targets, h, w))
            return {"loss": out.loss}
        return self.model(pixel_values=pixel_values)

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        pixel_values = self._stack(images)
        outputs = self.model(pixel_values=pixel_values)
        h, w = pixel_values.shape[-2:]
        sizes = torch.tensor([[h, w]] * pixel_values.shape[0])
        res = self.processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=self.score_threshold)
        return [{"boxes": r["boxes"], "labels": r["labels"], "scores": r["scores"]} for r in res]

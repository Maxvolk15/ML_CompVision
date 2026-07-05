from __future__ import annotations

import torch

from src.models.base import BaseDetector, register_model


@register_model("ssd")
class SSDDetector(BaseDetector):
    """SSD300 (one-stage), torchvision."""

    def __init__(self, model_cfg, full_cfg):
        super().__init__(model_cfg, full_cfg)
        from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
        from torchvision.models.detection.ssd import SSDClassificationHead

        num_classes = full_cfg["data"]["num_classes"] + 1
        pretrained = model_cfg.get("pretrained", True)
        self.model = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None)
        in_channels = [m.in_channels for m in self.model.head.classification_head.module_list]
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        self.model.head.classification_head = SSDClassificationHead(
            in_channels, num_anchors, num_classes)

    def forward(self, images, targets=None):
        if targets is not None:
            targets = [{**t, "labels": t["labels"] + 1} for t in targets]
            return self.model(images, targets)
        return self.model(images)

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        return [{"boxes": o["boxes"], "labels": o["labels"] - 1, "scores": o["scores"]}
                for o in self.model(images)]

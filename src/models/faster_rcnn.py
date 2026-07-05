from __future__ import annotations

import torch

from src.models.base import BaseDetector, register_model


@register_model("faster_rcnn")
class FasterRCNNDetector(BaseDetector):
    """Faster R-CNN (two-stage), torchvision."""

    def __init__(self, model_cfg, full_cfg):
        super().__init__(model_cfg, full_cfg)
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        num_classes = full_cfg["data"]["num_classes"] + 1
        pretrained = model_cfg.get("pretrained", True)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = fasterrcnn_resnet50_fpn(
            weights=weights, weights_backbone="DEFAULT" if pretrained else None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

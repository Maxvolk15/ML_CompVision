from __future__ import annotations

import torch

from src.models.base import BaseDetector, register_model


@register_model("efficientdet")
class EfficientDetDetector(BaseDetector):
    """EfficientDet-D0 (пакет effdet); при его отсутствии — RetinaNet (torchvision)."""

    def __init__(self, model_cfg, full_cfg):
        super().__init__(model_cfg, full_cfg)
        num_classes = full_cfg["data"]["num_classes"]
        pretrained = model_cfg.get("pretrained", True)
        try:
            from effdet import create_model
            self.backend = "effdet"
            self.model = create_model(
                model_cfg.get("arch", "tf_efficientdet_d0"), bench_task="train",
                num_classes=num_classes, pretrained=pretrained,
                image_size=(model_cfg.get("img_size", 512),) * 2)
        except ImportError:
            from functools import partial
            import torch.nn as nn
            from torchvision.models.detection import (
                retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
            from torchvision.models.detection.retinanet import RetinaNetClassificationHead
            self.backend = "retinanet"
            self.model = retinanet_resnet50_fpn_v2(
                weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None)
            num_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
            self.model.head.classification_head = RetinaNetClassificationHead(
                self.model.backbone.out_channels, num_anchors, num_classes,
                norm_layer=partial(nn.GroupNorm, 32))

    def forward(self, images, targets=None):
        if self.backend == "retinanet":
            return self.model(images, targets) if targets is not None else self.model(images)
        return self.model(images, targets)

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        if self.backend == "retinanet":
            return [{"boxes": o["boxes"], "labels": o["labels"], "scores": o["scores"]}
                    for o in self.model(images)]
        return [{"boxes": d[:, :4], "scores": d[:, 4], "labels": d[:, 5].long()}
                for d in self.model(images)]

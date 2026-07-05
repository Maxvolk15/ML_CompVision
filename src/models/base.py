from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

try:
    import torch.nn as nn
    _Base = nn.Module
except ImportError:
    _Base = object

_REGISTRY: Dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    def wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return wrap


def build_model(name: str, cfg: Dict[str, Any]) -> "BaseDetector":
    from src.models import yolo, faster_rcnn, ssd, efficientdet, detr  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"Неизвестная модель '{name}'. Доступны: {list(_REGISTRY)}")
    return _REGISTRY[name](cfg["models"][name], cfg)


class BaseDetector(_Base, ABC):
    """Единый интерфейс: forward(images, targets) -> loss/предсказания, predict(images)."""

    supports_amp = True  # можно ли обучать в смешанной точности (fp16); DETR — нет

    def __init__(self, model_cfg: Dict[str, Any], full_cfg: Dict[str, Any]):
        super().__init__()
        self.model_cfg = model_cfg
        self.full_cfg = full_cfg
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, images, targets=None): ...

    @abstractmethod
    def predict(self, images) -> List[Dict[str, Any]]: ...
